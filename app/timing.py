"""Timing correction: re-anchor utterance timestamps by sending each chunk back to Gemini."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from app.providers import call_with_retries
from app.transcribe import Utterance

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def build_timing_system(context_md: str, members_md: str) -> str:
    return (
        "Kamu adalah timing corrector untuk subtitle variety show Jepang "
        "(Soko Magattara, Sakurazaka?). Tugasmu HANYA memperbaiki timestamp "
        "start/end setiap ujaran agar sesuai dengan audio yang dilampirkan. "
        "JANGAN ubah teks ja sama sekali.\n\n"
        "=== KONTEKS ACARA ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )


def build_timing_user_prompt(utterances: list[Utterance]) -> str:
    payload = json.dumps(
        [{"id": u.id, "start": round(u.start, 3), "end": round(u.end, 3), "ja": u.ja}
         for u in utterances],
        ensure_ascii=False,
    )
    return (
        "Audio terlampir adalah satu chunk dari video. Timestamp berikut adalah "
        "hasil transkripsi otomatis yang mungkin sedikit meleset.\n\n"
        "Dengarkan audio dan perbaiki start/end setiap ujaran agar tepat sesuai "
        "kapan suara benar-benar terdengar. Timestamp relatif terhadap awal audio ini.\n\n"
        + payload + "\n\n"
        "Balas HANYA array JSON dengan jumlah item SAMA PERSIS, format per item:\n"
        '{"id": <id sama>, "start": <detik>, "end": <detik>}\n'
        "Aturan:\n"
        "- id HARUS sama persis dengan input.\n"
        "- Jumlah item HARUS sama persis.\n"
        "- start harus < end.\n"
        "- Jangan ubah urutan.\n"
        "- Jangan tambah atau hapus item."
    )


def parse_timing_response(text: str, expected_ids: list[int]) -> dict[int, tuple[float, float]] | None:
    """Parse Gemini's corrected timing. Returns {id: (start, end)} or None on failure."""
    try:
        cleaned = _FENCE_RE.sub("", text.strip())
        data = json.loads(cleaned)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        if not isinstance(data, list):
            return None
        result: dict[int, tuple[float, float]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            item_id = int(item["id"])
            start = float(item["start"])
            end = float(item["end"])
            if end <= start:
                end = start + 0.5
            result[item_id] = (start, end)
        if set(result.keys()) != set(expected_ids):
            return None
        return result
    except Exception:
        return None


def correct_chunk_timing(
    gemini,
    chunk_path: Path,
    utterances: list[Utterance],
    chunk_start: float,
    tracker,
) -> tuple[list[Utterance], bool]:
    """Send chunk audio + current timestamps to Gemini; get corrected timestamps back.

    Utterances have absolute timestamps. chunk_start is subtracted before sending
    (so Gemini sees timestamps relative to the chunk audio), then re-added after.

    Returns (corrected_utterances, success). success=False keeps original timestamps.
    """
    rel_utts = [replace(u, start=round(u.start - chunk_start, 3),
                        end=round(u.end - chunk_start, 3))
                for u in utterances]

    expected_ids = [u.id for u in rel_utts]
    user = build_timing_user_prompt(rel_utts)

    try:
        audio_ref = gemini.upload_audio(chunk_path)
        response = call_with_retries(
            lambda: gemini.generate(system="", user=user, audio=audio_ref)
        )
        if tracker is not None:
            tracker.add(gemini.model, response.input_tokens, response.output_tokens)

        corrected_map = parse_timing_response(response.text, expected_ids)
        if corrected_map is None:
            return utterances, False

        result: list[Utterance] = []
        for u in utterances:
            rel_start, rel_end = corrected_map[u.id]
            result.append(replace(
                u,
                start=round(chunk_start + rel_start, 3),
                end=round(chunk_start + rel_end, 3),
            ))
        return result, True

    except Exception:
        return utterances, False


def reanchor_all(
    utterances: list[Utterance],
    chunks: list,
    chunks_dir: Path,
    gemini,
    tracker,
    on_progress=None,
) -> tuple[list[Utterance], list[dict]]:
    """Re-anchor all utterances by re-sending each chunk to Gemini.

    Returns (corrected_utterances, report).
    """
    chunk_map: dict[int, list[Utterance]] = {c.index: [] for c in chunks}
    for u in utterances:
        for chunk in chunks:
            if chunk.start <= u.start < chunk.end or chunk == chunks[-1]:
                chunk_map[chunk.index].append(u)
                break

    result: list[Utterance] = []
    report: list[dict] = []
    n = len(chunks)

    for chunk in chunks:
        chunk_utts = chunk_map[chunk.index]
        if not chunk_utts:
            report.append({"chunk": chunk.index, "status": "skipped", "reason": "no_utterances"})
            if on_progress:
                on_progress(chunk.index, n)
            continue

        chunk_path = chunks_dir / f"chunk_{chunk.index:03d}.mp3"
        if not chunk_path.exists():
            report.append({"chunk": chunk.index, "status": "fallback", "reason": "chunk_file_missing"})
            result.extend(chunk_utts)
            if on_progress:
                on_progress(chunk.index, n)
            continue

        corrected, success = correct_chunk_timing(
            gemini, chunk_path, chunk_utts, chunk.start, tracker
        )

        if success:
            report.append({"chunk": chunk.index, "status": "corrected", "count": len(corrected)})
        else:
            report.append({"chunk": chunk.index, "status": "fallback", "reason": "gemini_parse_error"})

        result.extend(corrected)
        if on_progress:
            on_progress(chunk.index, n)

    result.sort(key=lambda u: u.start)
    result = [replace(u, id=i) for i, u in enumerate(result, start=1)]
    return result, report


def save_timing(utterances: list[Utterance], report: list[dict], out_dir: Path) -> None:
    timing_dir = out_dir / "timing"
    timing_dir.mkdir(exist_ok=True)
    (timing_dir / "utterances.json").write_text(
        json.dumps([u.to_dict() for u in utterances], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (timing_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_timing(out_dir: Path) -> list[Utterance]:
    path = out_dir / "timing" / "utterances.json"
    return [Utterance.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
