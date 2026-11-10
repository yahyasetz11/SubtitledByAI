"""QC agent: sliding-window LLM pass to verify and fix timing + translation."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

from app import audio
from app.providers import call_with_retries
from app.subtitle import _dashed_line_count

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


MAX_SLICE_SPAN_SECONDS = 90.0
MAX_SLICE_GAP_SECONDS = 15.0


def build_audio_qc_windows(
    rows: list[dict],
    *,
    flagged_ids: set[int],
    deep_audio_qc: bool = False,
    batch_size: int = 20,
    window_size: int = 10,
) -> list[dict]:
    """Return list of {'batch': [...], 'context_window': [...]} descriptors."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: (float(r["start"]), float(r["end"])))
    n = len(ordered)
    if deep_audio_qc:
        batch_size = max(1, int(batch_size))
        window_size = max(0, int(window_size))
        windows = []
        for start_i in range(0, n, batch_size):
            end_i = min(start_i + batch_size, n)
            ctx_start = max(0, start_i - window_size)
            ctx_end = min(n, end_i + window_size)
            windows.append({
                "batch": ordered[start_i:end_i],
                "context_window": ordered[ctx_start:ctx_end],
            })
        return windows

    flagged_indices = [i for i, r in enumerate(ordered) if int(r["id"]) in flagged_ids]
    if not flagged_indices:
        return []

    groups: list[list[int]] = [[flagged_indices[0]]]
    group_start = float(ordered[flagged_indices[0]]["start"])
    group_end = float(ordered[flagged_indices[0]]["end"])
    for idx in flagged_indices[1:]:
        cur_start = float(ordered[idx]["start"])
        cur_end = float(ordered[idx]["end"])
        gap = cur_start - group_end
        new_end = max(group_end, cur_end)
        span = new_end - group_start
        if gap <= MAX_SLICE_GAP_SECONDS and span <= MAX_SLICE_SPAN_SECONDS:
            groups[-1].append(idx)
            group_end = new_end
        else:
            groups.append([idx])
            group_start = cur_start
            group_end = cur_end

    windows = []
    for grp in groups:
        lo, hi = grp[0], grp[-1] + 1
        ctx_start = max(0, lo - 5)
        ctx_end = min(n, hi + 5)
        windows.append({
            "batch": [ordered[i] for i in grp],
            "context_window": ordered[ctx_start:ctx_end],
        })
    return windows


def build_audio_qc_user_prompt(
    batch: list[dict],
    context_window: list[dict],
    slice_start: float,
    glossary: dict[str, str] | None = None,
) -> str:
    context_lines = "\n".join(
        f"[{i['id']}] {float(i['start']):.2f}-{float(i['end']):.2f}s | "
        f"{i.get('ja') or ''} | {i.get('id_text') or ''}"
        for i in context_window
    )
    batch_payload = json.dumps([
        {
            "id": int(i["id"]),
            "start": round(float(i["start"]) - slice_start, 3),
            "end": round(float(i["end"]) - slice_start, 3),
            "ja": str(i.get("ja") or ""),
            "id_text": str(i.get("id_text") or ""),
        }
        for i in batch
    ], ensure_ascii=False)
    gloss_text = (
        "\n".join(f"- {jp} -> {id_val}" for jp, id_val in (glossary or {}).items())
        if glossary else "(kosong)"
    )

    return (
        "=== GLOSARIUM EPISODE ===\n"
        + gloss_text
        + "\n\n=== KONTEKS SUBTITLE SEKITAR (jangan ubah) ===\n"
        + context_lines
        + "\n\n=== ITEM UNTUK DIAUDIT DENGAN AUDIO ===\n"
        f"Audio terlampir dimulai dari detik {slice_start:.2f} video asli.\n"
        "Timestamp start/end di bawah relatif terhadap awal audio terlampir.\n\n"
        + batch_payload
        + "\n\nDengarkan audio dengan teliti. Untuk item yang bermasalah saja, perbaiki langsung:\n"
        "- fixed_ja: perbaiki jika Jepang salah dengar/nama member salah (kosongkan jika sudah benar)\n"
        "- fixed_id_text: perbaiki terjemahan agar akurat & sesuai glosarium (kosongkan jika sudah benar)\n"
        "- fixed_start / fixed_end: angka detik relatif terhadap audio terlampir jika timing meleset (atau null)\n"
        "- split: jika 1 item berisi 3+ pembicara, pecah menjadi 2-4 objek "
        '[{"start": <detik>, "end": <detik>, "ja": "...", "id_text": "..."}] (maks 2 pembicara per item)\n\n'
        "Balas HANYA array JSON (boleh [] jika semua sudah akurat):\n"
        '[{"id": <id>, "fixed_ja": "", "fixed_id_text": "", '
        '"fixed_start": null, "fixed_end": null, "split": null, "reason": "..."}]'
    )


def _merge_audio_qc_items(qc_results: list[dict]) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    for item in qc_results:
        if not isinstance(item, dict):
            continue
        rid = _safe_item_id(item.get("id"))
        if rid is None:
            continue
        raw_ja = item.get("fixed_ja")
        f_ja = raw_ja.strip() if isinstance(raw_ja, str) else ""
        raw_id = item.get("fixed_id_text")
        f_id = raw_id.strip() if isinstance(raw_id, str) else ""
        raw_reason = item.get("reason")
        reason = raw_reason.strip() if isinstance(raw_reason, str) else str(raw_reason or "").strip()
        has_timing = "fixed_start" in item or "fixed_end" in item
        has_split = "split" in item and item["split"] is not None

        if rid not in by_id:
            by_id[rid] = {
                "fixed_ja": f_ja,
                "fixed_id_text": f_id,
                "fixed_start": item.get("fixed_start"),
                "fixed_end": item.get("fixed_end"),
                "split": item.get("split") if has_split else None,
                "reason": reason,
            }
        else:
            prev = by_id[rid]
            if f_ja:
                prev["fixed_ja"] = f_ja
            if f_id:
                prev["fixed_id_text"] = f_id
            if has_timing and (item.get("fixed_start") is not None or item.get("fixed_end") is not None):
                prev["fixed_start"] = item.get("fixed_start")
                prev["fixed_end"] = item.get("fixed_end")
            if has_split:
                prev["split"] = item.get("split")
            if reason:
                if not prev["reason"]:
                    prev["reason"] = reason
                elif reason not in prev["reason"].split("; "):
                    prev["reason"] = f"{prev['reason']}; {reason}"
    return by_id


def apply_audio_qc_results(
    batch_rows: list[dict],
    qc_results: list[dict],
    *,
    slice_start: float,
    slice_duration: float,
) -> tuple[list[dict], list[dict]]:
    """Apply audio QC fixes (ja, id_text, timing, splits) to batch_rows."""
    by_id = _merge_audio_qc_items(qc_results)
    fixed: list[dict] = []
    issues: list[dict] = []
    max_rel = slice_duration + 0.5

    for row in batch_rows:
        rid = int(row["id"])
        qc = by_id.get(rid)
        if qc is None:
            fixed.append(dict(row))
            continue

        reason = qc["reason"]
        raw_type = row.get("type")
        row_type = raw_type.strip().lower() if isinstance(raw_type, str) else "dialogue"
        if row_type not in ("dialogue", "narration"):
            row_type = "dialogue"

        split_items = qc.get("split")
        if isinstance(split_items, list):
            valid_parts: list[dict] = []
            for part in split_items[:4]:
                if not isinstance(part, dict):
                    continue
                raw_ps = part.get("start")
                raw_pe = part.get("end")
                if not (_is_finite_number(raw_ps) and _is_finite_number(raw_pe)):
                    continue
                ps = float(raw_ps)
                pe = float(raw_pe)
                raw_pja = part.get("ja")
                raw_pid = part.get("id_text")
                p_ja = raw_pja.strip() if isinstance(raw_pja, str) else ""
                p_id = raw_pid.strip() if isinstance(raw_pid, str) else ""
                if 0.0 <= ps < pe <= max_rel and p_ja and p_id:
                    valid_parts.append({
                        "id": rid,
                        "start": round(slice_start + ps, 3),
                        "end": round(slice_start + pe, 3),
                        "type": row_type,
                        "ja": p_ja,
                        "id_text": p_id,
                    })
            if len(valid_parts) >= 2:
                fixed.extend(valid_parts)
                issues.append({
                    "id": rid,
                    "pass": "audio_qc",
                    "type": "split",
                    "original": str(row.get("id_text") or ""),
                    "fixed": f"Dipecah menjadi {len(valid_parts)} baris",
                    "reason": reason,
                })
                continue
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "warning",
                "reason": f"Split QC tidak valid untuk id={rid} (kurang dari 2 bagian valid)",
            })

        out = dict(row)
        f_ja = qc["fixed_ja"]
        if f_ja and f_ja != str(row.get("ja") or "").strip():
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "ja",
                "original": str(row.get("ja") or ""),
                "fixed": f_ja,
                "reason": reason,
            })
            out["ja"] = f_ja

        f_id = qc["fixed_id_text"]
        if f_id and f_id != str(row.get("id_text") or "").strip():
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "translation",
                "original": str(row.get("id_text") or ""),
                "fixed": f_id,
                "reason": reason,
            })
            out["id_text"] = f_id

        fs = qc.get("fixed_start")
        fe = qc.get("fixed_end")
        if fs is not None and fe is not None:
            if not (_is_finite_number(fs) and _is_finite_number(fe)):
                issues.append({
                    "id": rid,
                    "pass": "audio_qc",
                    "type": "warning",
                    "reason": f"Timestamp QC tidak valid diabaikan untuk id={rid}",
                })
            else:
                rel_s, rel_e = float(fs), float(fe)
                if 0.0 <= rel_s < rel_e <= max_rel:
                    abs_s = round(slice_start + rel_s, 3)
                    abs_e = round(slice_start + rel_e, 3)
                    orig_s = float(row["start"])
                    orig_e = float(row["end"])
                    if abs_s != round(orig_s, 3) or abs_e != round(orig_e, 3):
                        issues.append({
                            "id": rid,
                            "pass": "audio_qc",
                            "type": "timing",
                            "original": f"{orig_s:.2f}-{orig_e:.2f}s",
                            "fixed": f"{abs_s:.2f}-{abs_e:.2f}s",
                            "reason": reason,
                        })
                        out["start"] = abs_s
                        out["end"] = abs_e
                else:
                    issues.append({
                        "id": rid,
                        "pass": "audio_qc",
                        "type": "warning",
                        "reason": f"Timestamp QC di luar batas slice diabaikan ({rel_s:.2f}-{rel_e:.2f}s)",
                    })
        elif (fs is None) != (fe is None):
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "warning",
                "reason": f"Timestamp QC tidak lengkap diabaikan untuk id={rid}",
            })

        fixed.append(out)
    return fixed, issues


def _load_audio_qc_checkpoint(
    checkpoint_path: Path | None,
    expected_batch_ids: list[int],
) -> tuple[list[dict], list[dict]] | None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    try:
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict):
        return None
    raw_batch_ids = saved.get("batch_ids")
    if not isinstance(raw_batch_ids, list):
        return None
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in raw_batch_ids):
        return None
    if raw_batch_ids != expected_batch_ids:
        return None
    fixed = saved.get("fixed")
    issues = saved.get("issues")
    if not isinstance(fixed, list) or not isinstance(issues, list):
        return None
    if not all(isinstance(r, dict) for r in fixed):
        return None
    expected_set = set(expected_batch_ids)
    fixed_ids: set[int] = set()
    for r in fixed:
        rid = _safe_item_id(r.get("id"))
        if rid not in expected_set:
            return None
        if not (_is_finite_number(r.get("start")) and _is_finite_number(r.get("end"))):
            return None
        fixed_ids.add(rid)
    if fixed_ids != expected_set:
        return None
    if not all(isinstance(i, dict) for i in issues):
        return None
    return fixed, issues


def _summarize_issues(
    issues: list[dict],
    *,
    total_rows: int,
    local_flags: int = 0,
    audio_slices_checked: int = 0,
    checked_audio_ids: set[int] | None = None,
) -> dict:
    checked = checked_audio_ids or set()
    return {
        "total_rows": total_rows,
        "local_flags": local_flags,
        "audio_slices_checked": audio_slices_checked,
        "translation_fixes": sum(1 for i in issues if i.get("type") == "translation"),
        "ja_fixes": sum(1 for i in issues if i.get("type") == "ja"),
        "timing_fixes": sum(1 for i in issues if i.get("type") == "timing"),
        "splits": sum(1 for i in issues if i.get("type") == "split"),
        "unresolved_warnings": sum(
            1 for i in issues
            if i.get("type") == "warning"
            or (i.get("type") == "audio_flag" and _safe_item_id(i.get("id")) not in checked)
        ),
    }


def run_qc_all(
    gemini,
    rows: list[dict],
    audio_path: Path | None,
    tmp_dir: Path,
    system: str,
    tracker,
    batch_size: int = 20,
    window_size: int = 10,
    on_batch=None,
    *,
    glossary: dict[str, str] | None = None,
    timing_report: list[dict] | None = None,
    chunk_offsets: Sequence[dict | audio.PlannedChunk] | None = None,
    cps_threshold: float = 25.0,
    deep_audio_qc: bool = False,
    text_runner=None,
    audio_runner=None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Two-Pass Hybrid QC: Pass 1A local audit + Pass 1B text QC + Pass 2 targeted/deep audio QC."""
    mode = "text_only" if audio_path is None else ("deep" if deep_audio_qc else "hybrid")
    if not rows:
        return [], {
            "mode": mode,
            "deep_audio_qc": deep_audio_qc,
            "summary": _summarize_issues([], total_rows=0),
            "issues": [],
        }

    if len({int(r["id"]) for r in rows}) != len(rows):
        raise ValueError("id baris duplikat dalam rows")

    if text_runner is None and gemini is not None:
        def text_runner(sys_p, usr_p):
            resp = call_with_retries(lambda: gemini.generate(system=sys_p, user=usr_p))
            return resp, gemini.model

    if audio_runner is None and gemini is not None:
        def audio_runner(slice_p, sys_p, usr_p):
            ref = gemini.upload_audio(slice_p)
            resp = call_with_retries(lambda: gemini.generate(system=sys_p, user=usr_p, audio=ref))
            return resp, gemini.model

    # Pass 1A: free local audit
    audit_flags = audit_local(
        rows,
        audio_path=audio_path,
        timing_report=timing_report,
        chunk_offsets=chunk_offsets,
        cps_threshold=cps_threshold,
    )

    # Pass 1B: episode-wide text consistency QC
    pass1_ckpt = (checkpoint_dir / "qc_pass1.json") if checkpoint_dir is not None else None
    if text_runner is not None:
        working_rows, pass1_issues, text_audio_ids = run_text_qc(
            rows,
            system=system,
            glossary=glossary or {},
            audit_flags=audit_flags,
            tracker=tracker,
            text_runner=text_runner,
            checkpoint_path=pass1_ckpt,
        )
    else:
        working_rows = [dict(r) for r in rows]
        pass1_issues = []
        text_audio_ids = set()

    # Pass 2: targeted or deep audio QC
    all_issues = list(pass1_issues)
    acoustic_flagged_ids = {
        rid for rid, reasons in audit_flags.items()
        if any(r in ACOUSTIC_FLAG_REASONS for r in reasons)
    }
    flagged_ids = acoustic_flagged_ids | text_audio_ids

    audio_slices_checked = 0
    checked_audio_ids: set[int] = set()
    if audio_path is not None and audio_runner is not None:
        windows = build_audio_qc_windows(
            working_rows,
            flagged_ids=flagged_ids,
            deep_audio_qc=deep_audio_qc,
            batch_size=batch_size,
            window_size=window_size,
        )
        if windows:
            tmp_dir.mkdir(parents=True, exist_ok=True)
        n_win = len(windows)
        by_id_replacements: dict[int, list[dict]] = {}

        for w_idx, win in enumerate(windows):
            batch = win["batch"]
            batch_ids = [int(r["id"]) for r in batch]
            ckpt_file = (
                checkpoint_dir / f"qc_audio_{w_idx:03d}.json"
                if checkpoint_dir is not None else None
            )
            loaded_ckpt = _load_audio_qc_checkpoint(ckpt_file, batch_ids)
            if loaded_ckpt is not None:
                slice_fixed, slice_issues = loaded_ckpt
            else:
                slice_path = tmp_dir / f"qc_slice_{w_idx:03d}.mp3"
                slice_start = max(0.0, min(float(r["start"]) for r in batch) - AUDIO_BUFFER_SECONDS)
                slice_end = max(float(r["end"]) for r in batch) + AUDIO_BUFFER_SECONDS
                slice_duration = max(0.5, slice_end - slice_start)
                try:
                    audio.cut_chunk(audio_path, slice_path, slice_start, slice_end)
                    user = build_audio_qc_user_prompt(
                        batch, win["context_window"], slice_start, glossary=glossary,
                    )
                    response, model_used = audio_runner(slice_path, system, user)
                    if tracker is not None:
                        tracker.add(model_used, response.input_tokens, response.output_tokens)
                    try:
                        qc_results = parse_qc_response(response.text)
                    except (json.JSONDecodeError, ValueError) as exc:
                        retry_user = (
                            f"Output sebelumnya SALAH ({exc}). Balas HANYA array JSON sesuai skema.\n\n"
                            + user
                        )
                        response, model_used = audio_runner(slice_path, system, retry_user)
                        if tracker is not None:
                            tracker.add(model_used, response.input_tokens, response.output_tokens)
                        try:
                            qc_results = parse_qc_response(response.text)
                        except (json.JSONDecodeError, ValueError) as retry_exc:
                            slice_fixed = [dict(r) for r in batch]
                            slice_issues = [{
                                "pass": "audio_qc",
                                "type": "warning",
                                "reason": f"Respons Audio QC tidak valid ({retry_exc})",
                            }]
                        else:
                            slice_fixed, slice_issues = apply_audio_qc_results(
                                batch, qc_results,
                                slice_start=slice_start, slice_duration=slice_duration,
                            )
                    else:
                        slice_fixed, slice_issues = apply_audio_qc_results(
                            batch, qc_results,
                            slice_start=slice_start, slice_duration=slice_duration,
                        )
                finally:
                    slice_path.unlink(missing_ok=True)

                if ckpt_file is not None:
                    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_file.write_text(
                        json.dumps({
                            "batch_ids": batch_ids,
                            "fixed": slice_fixed,
                            "issues": slice_issues,
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            audio_slices_checked += 1
            checked_audio_ids.update(batch_ids)
            all_issues.extend(slice_issues)

            grouped: dict[int, list[dict]] = {bid: [] for bid in batch_ids}
            for sf in slice_fixed:
                sf_id = _safe_item_id(sf.get("id"))
                if sf_id in grouped:
                    grouped[sf_id].append(sf)
            for bid in batch_ids:
                if not grouped[bid]:
                    grouped[bid] = [next(dict(r) for r in batch if int(r["id"]) == bid)]
            by_id_replacements.update(grouped)

            if on_batch:
                on_batch(w_idx + 1, n_win)

        rebuilt: list[dict] = []
        for r in working_rows:
            rid = int(r["id"])
            if rid in by_id_replacements:
                rebuilt.extend(by_id_replacements[rid])
            else:
                rebuilt.append(r)
        working_rows = rebuilt

    # Sort and re-index sequentially
    working_rows.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    final_rows = [{**r, "id": i} for i, r in enumerate(working_rows, start=1)]

    report = {
        "mode": mode,
        "deep_audio_qc": deep_audio_qc,
        "summary": _summarize_issues(
            all_issues,
            total_rows=len(final_rows),
            local_flags=len(audit_flags),
            audio_slices_checked=audio_slices_checked,
            checked_audio_ids=checked_audio_ids,
        ),
        "issues": all_issues,
    }
    return final_rows, report


def save_qc(rows: list[dict], report_or_issues: dict | list[dict], out_dir: Path) -> None:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(report_or_issues, list):
        report = {
            "mode": "hybrid",
            "deep_audio_qc": False,
            "summary": _summarize_issues(report_or_issues, total_rows=len(rows)),
            "issues": report_or_issues,
        }
    else:
        report = report_or_issues
    (qc_dir / "utterances.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (qc_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_qc(out_dir: Path) -> list[dict]:
    path = out_dir / "qc" / "utterances.json"
    return json.loads(path.read_text(encoding="utf-8"))
