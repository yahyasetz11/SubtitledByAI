"""JP -> ID translation in numbered JSON batches (§6b)."""
from __future__ import annotations

import json
import re

from app.providers import call_with_retries
from app.transcribe import Utterance

BATCH_SIZE = 100  # merged utterances per request (spec: "per chunk"; see plan notes)


class TranslateError(Exception):
    """Translation failed after retry + per-line fallback."""


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_translation(text: str) -> dict[int, str]:
    """Model output -> {id: id_text}. Tolerates fences and wrapper dicts."""
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("output bukan array JSON")
    result: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item["id"])
            id_text = str(item["id_text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if id_text:
            result[item_id] = id_text
    return result


def validate_alignment(mapping: dict[int, str],
                       expected_ids: list[int]) -> str | None:
    """None when aligned; otherwise a human-readable error for the retry prompt."""
    expected = set(expected_ids)
    got = set(mapping)
    problems = []
    if missing := sorted(expected - got):
        problems.append(f"id hilang: {missing}")
    if extra := sorted(got - expected):
        problems.append(f"id tidak dikenal: {extra}")
    if len(mapping) != len(expected):
        problems.append(f"jumlah item {len(mapping)} != {len(expected)}")
    return "; ".join(problems) if problems else None


def build_translate_system(context_md: str, members_md: str) -> str:
    return (
        "Kamu adalah penerjemah subtitle JP->ID untuk variety show "
        "Soko Magattara, Sakurazaka?. Patuhi style guide, glossary, dan "
        "ejaan nama persis seperti referensi berikut.\n\n"
        "=== KONTEKS & STYLE GUIDE ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )


def _build_user_prompt(utterances: list[Utterance],
                       error_feedback: str | None = None) -> str:
    payload = json.dumps([{"id": u.id, "ja": u.ja} for u in utterances],
                         ensure_ascii=False)
    prompt = (
        "Terjemahkan tiap item berikut ke Bahasa Indonesia sesuai style guide.\n"
        "Balas HANYA array JSON dengan id yang SAMA PERSIS dan jumlah item "
        "yang SAMA PERSIS, format per item: "
        '{"id": <id sama>, "id_text": "<terjemahan>"}.\n'
        "Jangan menggabung, memecah, menambah, atau membuang item.\n\n" + payload
    )
    if error_feedback:
        prompt = (f"Output sebelumnya SALAH ({error_feedback}). Perbaiki dan "
                  f"ikuti instruksi dengan tepat.\n\n" + prompt)
    return prompt


def _request_mapping(client, utterances, system, tracker,
                     error_feedback=None) -> dict[int, str]:
    response = call_with_retries(
        lambda: client.generate(system=system,
                                user=_build_user_prompt(utterances,
                                                        error_feedback)))
    if tracker is not None:
        tracker.add(client.model, response.input_tokens, response.output_tokens)
    try:
        return parse_translation(response.text)
    except (json.JSONDecodeError, ValueError):
        return {}


def translate_batch(client, utterances: list[Utterance], system: str,
                    tracker) -> dict[int, str]:
    expected_ids = [u.id for u in utterances]

    mapping = _request_mapping(client, utterances, system, tracker)
    error = validate_alignment(mapping, expected_ids)
    if error is None:
        return mapping

    retry = _request_mapping(client, utterances, system, tracker,
                             error_feedback=error)
    if validate_alignment(retry, expected_ids) is None:
        return retry
    mapping = {**retry, **{k: v for k, v in mapping.items() if k not in retry}}
    mapping = {k: v for k, v in mapping.items() if k in set(expected_ids)}

    # Granular fallback: only ids that are still missing, one by one (§9).
    by_id = {u.id: u for u in utterances}
    for missing_id in sorted(set(expected_ids) - set(mapping)):
        single = _request_mapping(client, [by_id[missing_id]], system, tracker)
        if missing_id not in single:
            raise TranslateError(
                f"Gagal menerjemahkan id={missing_id} "
                f"({by_id[missing_id].ja[:40]}...) setelah fallback per-baris.")
        mapping[missing_id] = single[missing_id]
    return mapping


def translate_all(client, utterances: list[Utterance], system: str, tracker,
                  on_batch=None, load_checkpoint=None,
                  save_checkpoint=None) -> dict[int, str]:
    """Translate in BATCH_SIZE groups with optional per-batch checkpoints.

    load_checkpoint(batch_index) -> dict | None
    save_checkpoint(batch_index, mapping) -> None
    on_batch(done_count, total_count) -> None   (progress callback)
    """
    batches = [utterances[i:i + BATCH_SIZE]
               for i in range(0, len(utterances), BATCH_SIZE)]
    result: dict[int, str] = {}
    for idx, batch in enumerate(batches, start=1):
        mapping = load_checkpoint(idx) if load_checkpoint else None
        if mapping is None:
            mapping = translate_batch(client, batch, system, tracker)
            if save_checkpoint:
                save_checkpoint(idx, mapping)
        result.update(mapping)
        if on_batch:
            on_batch(idx, len(batches))
    return result
