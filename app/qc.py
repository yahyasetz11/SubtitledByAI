"""QC agent: sliding-window LLM pass to verify and fix timing + translation."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from app import audio
from app.providers import call_with_retries
from app.subtitle import _dashed_line_count
from app.transcribe import Utterance

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

AUDIO_BUFFER_SECONDS = 1.5

_JP_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_PRONOUN_RE = re.compile(r"\b(saya|anda|gue|lo)\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

ACOUSTIC_FLAG_REASONS = {
    "timing_fallback",
    "in_silence",
    "overlap",
    "rate_outlier",
    "speaker_overflow",
}


def _chunk_field(chunk, key: str):
    return chunk[key] if isinstance(chunk, dict) else getattr(chunk, key)


def _is_in_silence(start: float, end: float, silences: list) -> bool:
    dur = end - start
    if dur < 0.7:
        return False
    for s in silences:
        if s.start <= start + 0.05 and end <= s.end + 0.05:
            return True
    return False


def audit_local(
    rows: list[dict],
    *,
    audio_path: Path | None = None,
    timing_report: list[dict] | None = None,
    chunk_offsets: Sequence[dict | audio.PlannedChunk] | None = None,
    cps_threshold: float = 25.0,
) -> dict[int, list[str]]:
    """Free deterministic QC audit. Returns {row_id: [reason_codes]}."""
    if not rows:
        return {}

    flags: dict[int, list[str]] = {}

    def _flag(row_id: int, code: str) -> None:
        lst = flags.setdefault(row_id, [])
        if code not in lst:
            lst.append(code)

    sorted_rows = sorted(rows, key=lambda r: (float(r["start"]), float(r["end"])))

    # 1. Identify utterances belonging to timing fallback chunks
    fallback_chunks = {
        e["chunk"]
        for e in (timing_report or [])
        if e.get("status") == "fallback" and "chunk" in e
    }
    if fallback_chunks and chunk_offsets:
        norm_chunks = sorted(
            [
                (_chunk_field(c, "index"), float(_chunk_field(c, "start")), float(_chunk_field(c, "end")))
                for c in chunk_offsets
            ],
            key=lambda x: x[1],
        )
        first_index, first_start, _ = norm_chunks[0]
        for row in sorted_rows:
            rstart = float(row["start"])
            if rstart < first_start:
                if first_index in fallback_chunks:
                    _flag(int(row["id"]), "timing_fallback")
                continue
            for idx, (cindex, cstart, cend) in enumerate(norm_chunks):
                if cstart <= rstart < cend or idx == len(norm_chunks) - 1:
                    if cindex in fallback_chunks:
                        _flag(int(row["id"]), "timing_fallback")
                    break

    # 2. Acoustic silence check (only when audio_path is available)
    silences = []
    if audio_path is not None and audio_path.exists():
        try:
            silences = audio.detect_silences(audio_path, -30)
        except Exception:
            silences = []

    prev_end: float | None = None
    for row in sorted_rows:
        rid = int(row["id"])
        start = float(row["start"])
        end = float(row["end"])
        dur = max(0.01, end - start)
        ja = str(row.get("ja") or "")
        id_text = str(row.get("id_text") or "")

        if silences and _is_in_silence(start, end, silences):
            _flag(rid, "in_silence")

        if prev_end is not None and prev_end > start + 0.05:
            _flag(rid, "overlap")
        prev_end = max(prev_end, end) if prev_end is not None else end

        ja_compact = _SPACE_RE.sub("", ja)
        jp_rate = len(ja_compact) / dur
        if end <= start or (dur >= 0.4 and jp_rate > 18.0) or (dur > 3.0 and jp_rate < 1.2):
            _flag(rid, "rate_outlier")

        ja_turns = _dashed_line_count(ja)
        id_turns = _dashed_line_count(id_text)
        if ja_turns >= 3 or id_turns >= 3:
            _flag(rid, "speaker_overflow")

        if _JP_KANA_RE.search(id_text):
            _flag(rid, "untranslated_jp")

        if len(id_text.strip()) / dur > cps_threshold:
            _flag(rid, "high_cps")

        if _PRONOUN_RE.search(id_text):
            _flag(rid, "pronoun_violation")

    return flags


def slice_audio_for_batch(
    audio_path: Path,
    batch: list[dict],
    out_path: Path,
) -> tuple[Path, float]:
    """Cut audio to cover batch utterances ± buffer. Returns (path, slice_start)."""
    start = max(0.0, batch[0]["start"] - AUDIO_BUFFER_SECONDS)
    end = batch[-1]["end"] + AUDIO_BUFFER_SECONDS
    audio.cut_chunk(audio_path, out_path, start, end)
    return out_path, start


def build_qc_system(context_md: str, members_md: str,
                    additional_context: str | None = None) -> str:
    result = (
        "Kamu adalah QC editor subtitle JP->ID untuk variety show "
        "Soko Magattara, Sakurazaka?. Tugasmu: verifikasi timing dan terjemahan "
        "setiap ujaran dalam batch menggunakan audio yang dilampirkan dan konteks "
        "subtitle lengkap yang diberikan. Patuhi style guide dan ejaan nama persis "
        "seperti referensi berikut.\n\n"
        "=== KONTEKS & STYLE GUIDE ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )
    if additional_context:
        result += "\n\n=== KONTEKS VIDEO ===\n" + additional_context
    return result


def build_qc_user_prompt(
    batch: list[dict],
    context_window: list[dict],
    slice_start: float,
) -> str:
    context_lines = "\n".join(
        f"[{i['id']}] {i['start']:.2f}-{i['end']:.2f}s | {i['ja']} | {i.get('id_text','')}"
        for i in context_window
    )
    batch_payload = json.dumps([
        {
            "id": i["id"],
            "start": round(i["start"] - slice_start, 3),
            "end": round(i["end"] - slice_start, 3),
            "ja": i["ja"],
            "id_text": i.get("id_text", ""),
        }
        for i in batch
    ], ensure_ascii=False)

    return (
        "=== KONTEKS SUBTITLE (jangan ubah) ===\n"
        + context_lines
        + "\n\n=== BATCH UNTUK DIPERIKSA ===\n"
        "Audio terlampir dimulai dari detik " + f"{slice_start:.2f}" + " dalam video asli.\n"
        "Timestamp start/end di bawah relatif terhadap awal audio terlampir.\n\n"
        + batch_payload + "\n\n"
        "Untuk setiap item dalam batch, periksa:\n"
        "1. timing_ok: apakah start/end sesuai dengan suara di audio?\n"
        "2. translation_ok: apakah id_text akurat dan sesuai style guide?\n\n"
        "Balas HANYA array JSON dengan jumlah item SAMA PERSIS, format per item:\n"
        '{"id": <id sama>, "timing_ok": true|false, '
        '"translation_ok": true|false, '
        '"fixed_translation": "<terjemahan baru atau string kosong jika ok>", '
        '"issue": "<deskripsi masalah atau string kosong>"}'
    )


def parse_qc_response(text: str) -> list[dict]:
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("output bukan array JSON")
    return data


def apply_qc_results(
    batch_rows: list[dict],
    qc_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge QC fixes back into batch rows. Returns (fixed_rows, issues)."""
    by_id = {r["id"]: r for r in qc_results}
    fixed: list[dict] = []
    issues: list[dict] = []

    for row in batch_rows:
        qc = by_id.get(row["id"])
        out = dict(row)
        if qc is None:
            fixed.append(out)
            continue

        if not qc.get("translation_ok", True):
            fix = (qc.get("fixed_translation") or "").strip()
            if fix:
                out["id_text"] = fix
                issues.append({
                    "id": row["id"],
                    "type": "translation",
                    "original": row.get("id_text", ""),
                    "fixed": fix,
                    "issue": qc.get("issue", ""),
                })

        if not qc.get("timing_ok", True):
            issues.append({
                "id": row["id"],
                "type": "timing",
                "start": row["start"],
                "end": row["end"],
                "issue": qc.get("issue", ""),
            })

        fixed.append(out)
    return fixed, issues


def run_qc_all(
    gemini,
    rows: list[dict],
    audio_path: Path,
    tmp_dir: Path,
    system: str,
    tracker,
    batch_size: int = 20,
    window_size: int = 10,
    on_batch=None,
) -> tuple[list[dict], list[dict]]:
    """Run QC over all rows in sliding-window batches.

    Returns (fixed_rows, all_issues).
    """
    all_fixed: list[dict] = list(rows)
    all_issues: list[dict] = []
    n = len(rows)
    n_batches = (n + batch_size - 1) // batch_size
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx in range(n_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, n)
        batch = rows[start_i:end_i]

        ctx_start = max(0, start_i - window_size)
        ctx_end = min(n, end_i + window_size)
        context_window = rows[ctx_start:ctx_end]

        slice_path = tmp_dir / f"qc_slice_{batch_idx:03d}.mp3"
        try:
            _, slice_start = slice_audio_for_batch(audio_path, batch, slice_path)
        except Exception:
            if on_batch:
                on_batch(batch_idx + 1, n_batches)
            continue

        audio_ref = gemini.upload_audio(slice_path)
        user = build_qc_user_prompt(batch, context_window, slice_start)

        try:
            response = call_with_retries(
                lambda: gemini.generate(system=system, user=user, audio=audio_ref)
            )
            if tracker is not None:
                tracker.add(gemini.model, response.input_tokens, response.output_tokens)
            qc_results = parse_qc_response(response.text)
            fixed_batch, issues = apply_qc_results(batch, qc_results)
        except Exception:
            fixed_batch = batch
            issues = []

        for i, row in enumerate(fixed_batch):
            all_fixed[start_i + i] = row
        all_issues.extend(issues)

        try:
            slice_path.unlink(missing_ok=True)
        except Exception:
            pass

        if on_batch:
            on_batch(batch_idx + 1, n_batches)

    return all_fixed, all_issues


def save_qc(rows: list[dict], issues: list[dict], out_dir: Path) -> None:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(exist_ok=True)
    (qc_dir / "utterances.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (qc_dir / "report.json").write_text(
        json.dumps(issues, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_qc(out_dir: Path) -> list[dict]:
    path = out_dir / "qc" / "utterances.json"
    return json.loads(path.read_text(encoding="utf-8"))
