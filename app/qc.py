"""QC agent: sliding-window LLM pass to verify and fix timing + translation."""
from __future__ import annotations

import json
import math
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
    if text is None:
        raise ValueError("output kosong")
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


TEXT_QC_SINGLE_LIMIT = 250
TEXT_QC_WINDOW = 200
TEXT_QC_OVERLAP = 30


def build_text_qc_user_prompt(
    rows: list[dict],
    glossary: dict[str, str],
    audit_flags: dict[int, list[str]],
    *,
    check_ids: set[int] | None = None,
) -> str:
    gloss_block = (
        "\n".join(f"- {jp} -> {id_val}" for jp, id_val in glossary.items())
        if glossary else "(belum ada istilah khusus)"
    )
    items = []
    for r in rows:
        rid = int(r["id"])
        dur = round(max(0.0, float(r["end"]) - float(r["start"])), 2)
        entry = {
            "id": rid,
            "dur": dur,
            "ja": str(r.get("ja") or ""),
            "id_text": str(r.get("id_text") or ""),
        }
        if check_ids is not None and rid not in check_ids:
            entry["context_only"] = True
        flags = audit_flags.get(rid) if audit_flags else None
        if flags:
            entry["audit_flags"] = flags
        items.append(entry)

    return (
        "=== GLOSARIUM EPISODE (WAJIB KONSISTEN) ===\n"
        + gloss_block
        + "\n\n=== TRANSKRIP & TERJEMAHAN ===\n"
        + json.dumps(items, ensure_ascii=False)
        + "\n\nTugasmu (HANYA kembalikan item yang perlu diperbaiki atau dicek ulang audionya):\n"
        "1. Seragamkan terjemahan istilah berulang/segmen/nama acara di seluruh episode.\n"
        "2. Perbaiki style guide (pakai aku/kamu, tanpa sisa kana/kanji Jepang, ringkas baris >25 CPS).\n"
        "3. Pastikan maksimal 2 pembicara per baris (format '- Baris 1\\n- Baris 2').\n"
        "4. Tandai needs_audio_check=true jika teks 'ja' tampak salah dengar, aneh secara konteks, atau bertumpuk 3+ pembicara.\n"
        '5. Item dengan "context_only": true hanyalah konteks baca-saja dan JANGAN dikembalikan.\n\n'
        "Balas HANYA array JSON berisi item bermasalah saja (boleh [] jika semua baik):\n"
        '[{"id": <id>, "fixed_id_text": "<terjemahan baru atau string kosong>", '
        '"needs_audio_check": true|false, "reason": "<alasan singkat>"}]'
    )


def _safe_item_id(raw_id) -> int | None:
    if raw_id is None or isinstance(raw_id, (bool, list, dict)):
        return None
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float):
        return int(raw_id) if raw_id.is_integer() else None
    if isinstance(raw_id, str):
        try:
            return int(raw_id.strip())
        except ValueError:
            return None
    return None


def apply_text_qc_results(
    rows: list[dict],
    qc_items: list[dict],
) -> tuple[list[dict], list[dict], set[int]]:
    by_id: dict[int, dict] = {}
    for item in qc_items:
        if not isinstance(item, dict):
            continue
        rid = _safe_item_id(item.get("id"))
        if rid is None:
            continue
        raw_fix = item.get("fixed_id_text")
        fix = raw_fix.strip() if isinstance(raw_fix, str) else ""
        needs_audio = item.get("needs_audio_check") is True
        raw_reason = item.get("reason")
        reason = raw_reason.strip() if isinstance(raw_reason, str) else str(raw_reason or "").strip()

        if rid not in by_id:
            by_id[rid] = {
                "fixed_id_text": fix,
                "needs_audio_check": needs_audio,
                "reason": reason,
            }
        else:
            prev = by_id[rid]
            if fix:
                prev["fixed_id_text"] = fix
            if needs_audio:
                prev["needs_audio_check"] = True
            if reason:
                if not prev["reason"]:
                    prev["reason"] = reason
                elif reason not in prev["reason"].split("; "):
                    prev["reason"] = f"{prev['reason']}; {reason}"

    updated: list[dict] = []
    issues: list[dict] = []
    audio_ids: set[int] = set()

    for row in rows:
        out = dict(row)
        rid = int(row["id"])
        item = by_id.get(rid)
        if item is not None:
            fix = item["fixed_id_text"]
            reason = item["reason"]
            if fix and fix != str(row.get("id_text") or "").strip():
                issues.append({
                    "id": rid,
                    "pass": "text_qc",
                    "type": "translation",
                    "original": str(row.get("id_text") or ""),
                    "fixed": fix,
                    "reason": reason,
                })
                out["id_text"] = fix
            if item["needs_audio_check"]:
                audio_ids.add(rid)
                issues.append({
                    "id": rid,
                    "pass": "text_qc",
                    "type": "audio_flag",
                    "reason": reason or "ja/timing dicurigai perlu dicek audio",
                })
        updated.append(out)
    return updated, issues, audio_ids


def _is_finite_number(val) -> bool:
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(val)


def _load_text_qc_checkpoint(checkpoint_path: Path | None, expected_rows: list[dict]):
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    try:
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict):
        return None
    rows = saved.get("rows")
    issues = saved.get("issues")
    audio_ids = saved.get("audio_ids")
    if not isinstance(rows, list) or not isinstance(issues, list) or not isinstance(audio_ids, list):
        return None
    if not all(isinstance(r, dict) for r in rows):
        return None
    expected_ids = [int(r["id"]) for r in expected_rows]
    if [_safe_item_id(r.get("id")) for r in rows] != expected_ids:
        return None
    if not all(_is_finite_number(r.get("start")) and _is_finite_number(r.get("end")) for r in rows):
        return None
    if not all(isinstance(i, dict) for i in issues):
        return None
    parsed_ids: set[int] = set()
    for x in audio_ids:
        rid = _safe_item_id(x)
        if rid is None:
            return None
        parsed_ids.add(rid)
    if not (parsed_ids <= set(expected_ids)):
        return None
    return rows, issues, parsed_ids


def run_text_qc(
    rows: list[dict],
    *,
    system: str,
    glossary: dict[str, str],
    audit_flags: dict[int, list[str]],
    tracker,
    text_runner,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], list[dict], set[int]]:
    """Run Pass 1B whole-episode text consistency QC."""
    if not rows:
        return [], [], set()

    if len({int(r["id"]) for r in rows}) != len(rows):
        raise ValueError("id baris duplikat dalam rows")

    cached = _load_text_qc_checkpoint(checkpoint_path, rows)
    if cached is not None:
        return cached

    if len(rows) <= TEXT_QC_SINGLE_LIMIT:
        windows: list[tuple[int, int, set[int] | None]] = [(0, len(rows), None)]
    else:
        windows = []
        step = TEXT_QC_WINDOW - TEXT_QC_OVERLAP
        for start in range(0, len(rows), step):
            end = min(start + TEXT_QC_WINDOW, len(rows))
            if start == 0:
                check_ids = None
            else:
                check_ids = {int(r["id"]) for r in rows[start + TEXT_QC_OVERLAP:end]}
            windows.append((start, end, check_ids))
            if end >= len(rows):
                break

    current = [dict(r) for r in rows]
    all_issues: list[dict] = []
    all_audio_ids: set[int] = set()

    for start, end, check_ids in windows:
        win_rows = current[start:end]
        prompt = build_text_qc_user_prompt(win_rows, glossary, audit_flags, check_ids=check_ids)
        response, model_used = text_runner(system, prompt)
        if tracker is not None:
            tracker.add(model_used, response.input_tokens, response.output_tokens)

        try:
            qc_items = parse_qc_response(response.text)
        except (json.JSONDecodeError, ValueError) as exc:
            retry_prompt = (
                f"Output sebelumnya SALAH ({exc}). Balas HANYA array JSON sesuai skema.\n\n"
                + prompt
            )
            response, model_used = text_runner(system, retry_prompt)
            if tracker is not None:
                tracker.add(model_used, response.input_tokens, response.output_tokens)
            try:
                qc_items = parse_qc_response(response.text)
            except (json.JSONDecodeError, ValueError) as retry_exc:
                all_issues.append({
                    "pass": "text_qc",
                    "type": "warning",
                    "reason": f"Respons Pass 1B tidak valid ({retry_exc})",
                })
                continue

        allowed_ids = check_ids if check_ids is not None else {int(r["id"]) for r in win_rows}
        filtered = []
        for it in qc_items:
            if not isinstance(it, dict):
                continue
            rid = _safe_item_id(it.get("id"))
            if rid is not None and rid in allowed_ids:
                filtered.append(it)
        current, win_issues, win_audio = apply_text_qc_results(current, filtered)
        all_issues.extend(win_issues)
        all_audio_ids |= win_audio

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({
                "rows": current,
                "issues": all_issues,
                "audio_ids": sorted(all_audio_ids),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return current, all_issues, all_audio_ids


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
