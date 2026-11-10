"""JP -> ID translation in numbered JSON batches (§6b)."""
from __future__ import annotations

import inspect
import json
import re

from app.providers import call_with_retries
from app.transcribe import Utterance

BATCH_SIZE = 100
MAX_GLOSSARY_TERMS = 80


class TranslateError(Exception):
    """Translation failed after retry + per-line fallback."""


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def _clean_terms(raw: dict | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, str] = {}
    for k, v in raw.items():
        if k is None or isinstance(k, (dict, list, bool)):
            continue
        if v is None or isinstance(v, (dict, list, bool)):
            continue
        jp = str(k).strip()
        id_val = str(v).strip()
        if jp and id_val:
            cleaned[jp] = id_val
    return cleaned


def merge_glossary(base: dict[str, str], additions: dict[str, str],
                   limit: int = MAX_GLOSSARY_TERMS) -> dict[str, str]:
    """Merge new JP->ID terms into base, keeping at most `limit` entries."""
    merged = dict(base)
    for jp, id_val in _clean_terms(additions).items():
        if jp in merged:
            del merged[jp]
        merged[jp] = id_val
    cap = max(0, limit)
    while len(merged) > cap:
        oldest = next(iter(merged))
        del merged[oldest]
    return merged


def _merge_into_glossary(target: dict[str, str] | None,
                         additions: dict[str, str]) -> None:
    if target is None or not additions:
        return
    updated = merge_glossary(target, additions)
    target.clear()
    target.update(updated)


def parse_translation_response(text: str) -> tuple[dict[int, str], dict[str, str]]:
    """Parse model output -> ({id: id_text}, {jp_term: id_term}).

    Accepts both {"translations": [...], "new_terms": {...}} and plain JSON arrays.
    """
    if text is None:
        raise ValueError("output kosong")
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    raw_terms: dict[str, str] = {}
    items = None

    if isinstance(data, dict):
        raw_terms = _clean_terms(data.get("new_terms"))
        if isinstance(data.get("translations"), list):
            items = data["translations"]
        else:
            for value in data.values():
                if isinstance(value, list):
                    items = value
                    break
    elif isinstance(data, list):
        items = data

    if not isinstance(items, list):
        raise ValueError("output bukan array JSON")

    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            raw_id = item["id"]
            raw_text = item["id_text"]
            if raw_id is None or isinstance(raw_id, (bool, float, dict, list)):
                continue
            if raw_text is None or isinstance(raw_text, (dict, list, bool)):
                continue
            item_id = int(raw_id)
            id_text = str(raw_text).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if id_text:
            result[item_id] = id_text
    return result, raw_terms


def parse_translation(text: str) -> dict[int, str]:
    """Backward-compatible wrapper returning only {id: id_text}."""
    mapping, _ = parse_translation_response(text)
    return mapping


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


def build_translate_system(context_md: str, members_md: str,
                           additional_context: str | None = None) -> str:
    result = (
        "Kamu adalah penerjemah subtitle JP->ID untuk variety show "
        "Soko Magattara, Sakurazaka?. Patuhi style guide, glossary, dan "
        "ejaan nama persis seperti referensi berikut.\n\n"
        "=== KONTEKS & STYLE GUIDE ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )
    if additional_context:
        result += "\n\n=== KONTEKS VIDEO ===\n" + additional_context
    return result


def _build_user_prompt(utterances: list[Utterance],
                       error_feedback: str | None = None,
                       glossary: dict[str, str] | None = None,
                       include_terms: bool = True) -> str:
    payload = json.dumps([{"id": u.id, "ja": u.ja} for u in utterances],
                         ensure_ascii=False)
    glossary_block = ""
    if glossary:
        lines = "\n".join(f"- {jp} -> {id_val}" for jp, id_val in glossary.items())
        glossary_block = (
            "=== GLOSARIUM EPISODE (WAJIB KONSISTEN) ===\n"
            + lines + "\n\n"
        )
    if include_terms:
        schema_instr = (
            "Balas HANYA JSON dengan format:\n"
            '{"translations": [{"id": <id sama>, "id_text": "<terjemahan>"}], '
            '"new_terms": {"<istilah/segmen JP berulang baru>": "<padanan ID konsisten>"}}\n'
            "Catatan:\n"
            "- Jumlah item di 'translations' HARUS SAMA PERSIS dengan id input.\n"
            "- Jika baris 'ja' memiliki format 2 pembicara diawali '- ', pertahankan format dua baris '- ' di id_text.\n"
            "- Isi 'new_terms' hanya untuk nama segmen, permainan, julukan, atau istilah berulang baru (boleh {}).\n"
        )
    else:
        schema_instr = (
            "Balas HANYA array JSON dengan id yang SAMA PERSIS dan jumlah item "
            "yang SAMA PERSIS, format per item: "
            '{"id": <id sama>, "id_text": "<terjemahan>"}.\n'
            "Jika baris 'ja' memiliki format 2 pembicara diawali '- ', "
            "pertahankan format dua baris '- ' di id_text.\n"
        )
    prompt = (
        glossary_block
        + "Terjemahkan tiap item berikut ke Bahasa Indonesia sesuai style guide.\n"
        + schema_instr
        + "Jangan menggabung, memecah, menambah, atau membuang item.\n\n" + payload
    )
    if error_feedback:
        prompt = (f"Output sebelumnya SALAH ({error_feedback}). Perbaiki dan "
                  f"ikuti instruksi dengan tepat.\n\n" + prompt)
    return prompt


def _request_mapping(client, utterances, system, tracker,
                     error_feedback=None, glossary=None,
                     text_runner=None) -> tuple[dict[int, str], dict[str, str]]:
    user_prompt = _build_user_prompt(
        utterances, error_feedback=error_feedback,
        glossary=glossary, include_terms=len(utterances) > 1,
    )
    if text_runner is not None:
        response, model_used = text_runner(system, user_prompt)
    else:
        response = call_with_retries(
            lambda: client.generate(system=system, user=user_prompt))
        model_used = client.model
    if tracker is not None:
        tracker.add(model_used, response.input_tokens, response.output_tokens)
    try:
        return parse_translation_response(response.text)
    except (json.JSONDecodeError, ValueError):
        return {}, {}


def translate_batch(client, utterances: list[Utterance], system: str,
                    tracker, glossary: dict[str, str] | None = None,
                    text_runner=None) -> dict[int, str]:
    if not utterances:
        return {}
    expected_ids = [u.id for u in utterances]

    def _merge_new_only(terms: dict[str, str]) -> None:
        if glossary is None or not terms:
            return
        new_only = {k: v for k, v in terms.items() if k not in glossary}
        _merge_into_glossary(glossary, new_only)

    mapping, new_terms = _request_mapping(
        client, utterances, system, tracker,
        glossary=glossary, text_runner=text_runner,
    )
    error = validate_alignment(mapping, expected_ids)
    if error is None:
        _merge_new_only(new_terms)
        return mapping

    retry, retry_terms = _request_mapping(
        client, utterances, system, tracker,
        error_feedback=error, glossary=glossary, text_runner=text_runner,
    )
    if validate_alignment(retry, expected_ids) is None:
        _merge_new_only({**new_terms, **retry_terms})
        return retry

    mapping = {**retry, **{k: v for k, v in mapping.items() if k not in retry}}
    mapping = {k: v for k, v in mapping.items() if k in set(expected_ids)}

    by_id = {u.id: u for u in utterances}
    for missing_id in sorted(set(expected_ids) - set(mapping)):
        single, _ = _request_mapping(
            client, [by_id[missing_id]], system, tracker,
            glossary=glossary, text_runner=text_runner,
        )
        if missing_id not in single:
            raise TranslateError(
                f"Gagal menerjemahkan id={missing_id} "
                f"({by_id[missing_id].ja[:40]}...) setelah fallback per-baris.")
        mapping[missing_id] = single[missing_id]

    _merge_new_only({**new_terms, **retry_terms})
    return mapping


def _unpack_checkpoint(raw) -> tuple[dict[int, str], dict[str, str]] | None:
    if not isinstance(raw, dict):
        return None
    if "translations" in raw:
        raw_trans = raw["translations"]
        if not isinstance(raw_trans, dict):
            return None
        raw_gloss = raw.get("glossary")
    else:
        raw_trans = raw
        raw_gloss = None

    trans: dict[int, str] = {}
    for k, v in raw_trans.items():
        if isinstance(k, bool) or not isinstance(k, (int, str)):
            return None
        try:
            item_id = int(k)
        except ValueError:
            return None
        if not isinstance(v, str) or not v.strip():
            return None
        trans[item_id] = v.strip()

    return trans, _clean_terms(raw_gloss)


def translate_all(client, utterances: list[Utterance], system: str, tracker,
                  on_batch=None, load_checkpoint=None,
                  save_checkpoint=None, glossary: dict[str, str] | None = None,
                  text_runner=None) -> dict[int, str]:
    """Translate in BATCH_SIZE groups with optional per-batch checkpoints & glossary."""
    batches = [utterances[i:i + BATCH_SIZE]
               for i in range(0, len(utterances), BATCH_SIZE)]
    result: dict[int, str] = {}
    active_glossary: dict[str, str] = glossary if glossary is not None else {}

    batch_params = inspect.signature(translate_batch).parameters
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in batch_params.values()
    )
    supports_glossary = "glossary" in batch_params or has_var_kw
    supports_runner = "text_runner" in batch_params or has_var_kw

    for idx, batch in enumerate(batches, start=1):
        ckpt_raw = load_checkpoint(idx) if load_checkpoint else None
        unpacked = _unpack_checkpoint(ckpt_raw) if ckpt_raw is not None else None
        if unpacked is not None:
            mapping, ckpt_glossary = unpacked
            _merge_into_glossary(active_glossary, ckpt_glossary)
        else:
            if supports_glossary:
                kwargs: dict = {"glossary": active_glossary}
                if supports_runner:
                    kwargs["text_runner"] = text_runner
                mapping = translate_batch(client, batch, system, tracker, **kwargs)
            else:
                mapping = translate_batch(client, batch, system, tracker)
            if save_checkpoint:
                save_checkpoint(idx, {
                    "translations": mapping,
                    "glossary": dict(active_glossary),
                })
        result.update(mapping)
        if on_batch:
            on_batch(idx, len(batches))
    return result
