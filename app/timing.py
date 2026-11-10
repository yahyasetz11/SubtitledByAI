"""Timing correction: re-anchor utterance timestamps by sending each chunk back to Gemini."""
from __future__ import annotations

import json
import math
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


def parse_timing_response(
    text: str,
    expected_ids: list[int],
    max_duration: float | None = None,
) -> dict[int, tuple[float, float]] | None:
    """Parse Gemini's corrected timing. Returns {id: (start, end)} or None on failure."""
    if text is None:
        return None
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
            raw_id = item["id"]
            raw_start = item["start"]
            raw_end = item["end"]
            if (
                isinstance(raw_id, bool)
                or isinstance(raw_start, bool)
                or isinstance(raw_end, bool)
            ):
                return None
            if isinstance(raw_id, float) and not raw_id.is_integer():
                return None
            item_id = int(raw_id)
            if item_id in result:
                return None
            start = float(raw_start)
            end = float(raw_end)
            if not (math.isfinite(start) and math.isfinite(end)) or start < 0.0:
                return None
            if end <= start:
                end = start + 0.5
            if max_duration is not None and (
                start > max_duration + 0.5 or end > max_duration + 2.0
            ):
                return None
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
    audio_runner=None,
    max_duration: float | None = None,
) -> tuple[list[Utterance], bool]:
    """Send chunk audio + current timestamps to Gemini; get corrected timestamps back.

    Raises API/transport failures (after fallback exhaustion in audio_runner).
    Returns (utterances, False) only on malformed JSON / ID mismatch.
    """
    if not utterances:
        return [], True

    rel_utts = [replace(u, start=round(u.start - chunk_start, 3),
                        end=round(u.end - chunk_start, 3))
                for u in utterances]

    expected_ids = [u.id for u in rel_utts]
    user = build_timing_user_prompt(rel_utts)

    if audio_runner is not None:
        response, model_used = audio_runner(chunk_path, "", user)
    else:
        audio_ref = gemini.upload_audio(chunk_path)
        response = call_with_retries(
            lambda: gemini.generate(system="", user=user, audio=audio_ref)
        )
        model_used = gemini.model

    if tracker is not None:
        tracker.add(model_used, response.input_tokens, response.output_tokens)

    corrected_map = parse_timing_response(
        response.text, expected_ids, max_duration=max_duration
    )
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


def _unpack_checkpoint(raw, expected_count: int) -> tuple[list[Utterance], dict] | None:
    if not isinstance(raw, dict):
        return None
    raw_utts = raw.get("utterances")
    report_entry = raw.get("report_entry")
    if not isinstance(raw_utts, list) or not isinstance(report_entry, dict):
        return None
    status = report_entry.get("status")
    if not isinstance(status, str) or not status:
        return None

    utts: list[Utterance] = []
    for item in raw_utts:
        if not isinstance(item, dict):
            return None
        if any(isinstance(item.get(k), bool) for k in ("id", "start", "end")):
            return None
        try:
            u = Utterance.from_dict(item)
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(u.start) and math.isfinite(u.end)):
            return None
        utts.append(u)

    if expected_count > 0 and status != "skipped" and len(utts) != expected_count:
        return None
    return utts, report_entry


def reanchor_all(
    utterances: list[Utterance],
    chunks: list,
    chunks_dir: Path,
    gemini,
    tracker,
    on_progress=None,
    audio_runner=None,
    load_checkpoint=None,
    save_checkpoint=None,
) -> tuple[list[Utterance], list[dict]]:
    """Re-anchor all utterances by re-sending each chunk to Gemini."""
    if not chunks:
        sorted_utts = sorted(utterances, key=lambda u: u.start)
        return [replace(u, id=i) for i, u in enumerate(sorted_utts, start=1)], []

    chunk_map: dict[int, list[Utterance]] = {c.index: [] for c in chunks}
    for u in utterances:
        if u.start < chunks[0].start:
            chunk_map[chunks[0].index].append(u)
            continue
        for idx, chunk in enumerate(chunks):
            if chunk.start <= u.start < chunk.end or idx == len(chunks) - 1:
                chunk_map[chunk.index].append(u)
                break

    result: list[Utterance] = []
    report: list[dict] = []
    n = len(chunks)

    for chunk in chunks:
        chunk_utts = chunk_map[chunk.index]
        raw_ckpt = load_checkpoint(chunk.index) if load_checkpoint else None
        unpacked = (
            _unpack_checkpoint(raw_ckpt, len(chunk_utts))
            if raw_ckpt is not None
            else None
        )

        if unpacked is not None:
            chunk_out, entry = unpacked
        else:
            if not chunk_utts:
                chunk_out = []
                entry = {"chunk": chunk.index, "status": "skipped", "reason": "no_utterances"}
            else:
                chunk_path = chunks_dir / f"chunk_{chunk.index:03d}.mp3"
                if not chunk_path.exists():
                    chunk_out = list(chunk_utts)
                    entry = {"chunk": chunk.index, "status": "fallback", "reason": "chunk_file_missing"}
                else:
                    chunk_out, success = correct_chunk_timing(
                        gemini,
                        chunk_path,
                        chunk_utts,
                        chunk.start,
                        tracker,
                        audio_runner=audio_runner,
                        max_duration=max(0.0, chunk.end - chunk.start),
                    )
                    if success:
                        entry = {"chunk": chunk.index, "status": "corrected", "count": len(chunk_out)}
                    else:
                        entry = {"chunk": chunk.index, "status": "fallback", "reason": "gemini_parse_error"}

            if save_checkpoint:
                save_checkpoint(
                    chunk.index,
                    {
                        "utterances": [u.to_dict() for u in chunk_out],
                        "report_entry": entry,
                    },
                )

        result.extend(chunk_out)
        report.append(entry)
        if on_progress:
            on_progress(chunk.index, n)

    result.sort(key=lambda u: u.start)
    result = [replace(u, id=i) for i, u in enumerate(result, start=1)]
    return result, report


def save_timing(rows: list[dict] | list[Utterance], report: list[dict], out_dir: Path) -> None:
    timing_dir = out_dir / "timing"
    timing_dir.mkdir(exist_ok=True)
    serialized = [u.to_dict() if isinstance(u, Utterance) else u for u in rows]
    (timing_dir / "utterances.json").write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (timing_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_timing(out_dir: Path) -> list[dict]:
    path = out_dir / "timing" / "utterances.json"
    return json.loads(path.read_text(encoding="utf-8"))
