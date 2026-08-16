"""QC agent: sliding-window LLM pass to verify and fix timing + translation."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from app import audio
from app.providers import call_with_retries
from app.transcribe import Utterance

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

AUDIO_BUFFER_SECONDS = 1.5


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
