import json

import pytest

from app.providers import LLMResponse
from app.transcribe import Utterance
from app.translate import (
    MAX_GLOSSARY_TERMS,
    TranslateError,
    _build_user_prompt,
    _unpack_checkpoint,
    build_translate_system,
    merge_glossary,
    parse_translation,
    parse_translation_response,
    translate_all,
    translate_batch,
    validate_alignment,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.model = "fake-flash"

    def generate(self, system, user, audio=None):
        self.prompts.append(user)
        return self.responses.pop(0)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, i, o):
        self.calls.append((model, i, o))


def U(id, ja):
    return Utterance(id=id, start=float(id), end=float(id) + 1, type="dialogue",
                     ja=ja)


def R(items):
    return LLMResponse(text=json.dumps(items, ensure_ascii=False),
                       input_tokens=10, output_tokens=10)


UTTS = [U(1, "こんにちは"), U(2, "ありがとう")]
GOOD_ITEMS = [{"id": 1, "id_text": "Halo"}, {"id": 2, "id_text": "Makasih"}]


def test_parse_translation_tolerates_fences_and_wrapper():
    text = '```json\n{"items": [{"id": 1, "id_text": "Halo"}]}\n```'
    assert parse_translation(text) == {1: "Halo"}


def test_validate_alignment_passes_on_exact_match():
    assert validate_alignment({1: "a", 2: "b"}, [1, 2]) is None


def test_validate_alignment_reports_missing_and_extra():
    error = validate_alignment({1: "a", 3: "c"}, [1, 2])
    assert "2" in error and "3" in error


def test_translate_batch_happy_path():
    client = FakeClient([R(GOOD_ITEMS)])
    tracker = FakeTracker()
    result = translate_batch(client, UTTS, "SYSTEM", tracker)
    assert result == {1: "Halo", 2: "Makasih"}
    assert tracker.calls == [("fake-flash", 10, 10)]
    assert "JSON" in client.prompts[0]
    assert "こんにちは" in client.prompts[0]


def test_translate_batch_retries_with_error_feedback():
    bad = R([{"id": 1, "id_text": "Halo"}])  # missing id 2
    client = FakeClient([bad, R(GOOD_ITEMS)])
    result = translate_batch(client, UTTS, "SYSTEM", FakeTracker())
    assert result == {1: "Halo", 2: "Makasih"}
    assert len(client.prompts) == 2
    assert "id" in client.prompts[1]  # retry prompt carries the error


def test_translate_batch_falls_back_per_line_for_missing_ids():
    bad = R([{"id": 1, "id_text": "Halo"}])
    single = R([{"id": 2, "id_text": "Makasih"}])
    client = FakeClient([bad, bad, single])  # batch, retry, then per-line id 2
    result = translate_batch(client, UTTS, "SYSTEM", FakeTracker())
    assert result == {1: "Halo", 2: "Makasih"}
    assert len(client.prompts) == 3
    assert "ありがとう" in client.prompts[2]
    assert "こんにちは" not in client.prompts[2]  # only the missing line is resent


def test_translate_batch_raises_when_fallback_fails():
    garbage = LLMResponse(text="not json", input_tokens=1, output_tokens=1)
    client = FakeClient([garbage] * 6)
    with pytest.raises(TranslateError):
        translate_batch(client, UTTS, "SYSTEM", FakeTracker())


def test_build_translate_system_injects_context():
    system = build_translate_system("CTX", "MEMBERS")
    assert "CTX" in system and "MEMBERS" in system


def test_build_translate_system_includes_additional_context():
    system = build_translate_system("CTX", "MEMBERS",
                                    additional_context="Fishing vlog in Kanagawa")
    assert "Fishing vlog in Kanagawa" in system


def test_parse_translation_response_accepts_object_with_new_terms():
    raw = json.dumps({
        "translations": [{"id": 1, "id_text": "Pertandingan Ratu Serigala"}],
        "new_terms": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"},
    }, ensure_ascii=False)
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Pertandingan Ratu Serigala"}
    assert terms == {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"}


def test_parse_translation_response_tolerates_legacy_plain_array():
    raw = json.dumps([{"id": 1, "id_text": "Halo"}])
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Halo"}
    assert terms == {}


def test_parse_translation_response_skips_none_values():
    raw = json.dumps({
        "translations": [
            {"id": 1, "id_text": "Halo"},
            {"id": 2, "id_text": None},
        ],
        "new_terms": {
            "k": None,
            "kosong": "   ",
            "人狼": "Manusia Serigala",
        },
    }, ensure_ascii=False)
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Halo"}
    assert terms == {"人狼": "Manusia Serigala"}


def test_merge_glossary_caps_at_max_terms():
    g = {f"jp{i}": f"id{i}" for i in range(MAX_GLOSSARY_TERMS)}
    merged = merge_glossary(g, {"baru": "terbaru"})
    assert len(merged) == MAX_GLOSSARY_TERMS
    assert merged["baru"] == "terbaru"
    assert "jp0" not in merged


def test_merge_glossary_refreshes_reinforced_term_and_skips_none():
    g = {f"jp{i}": f"id{i}" for i in range(MAX_GLOSSARY_TERMS)}
    merged = merge_glossary(g, {"jp0": "id0_baru", "skip": None, "baru": "terbaru"})
    assert len(merged) == MAX_GLOSSARY_TERMS
    assert merged["jp0"] == "id0_baru"
    assert merged["baru"] == "terbaru"
    assert "jp1" not in merged
    assert "skip" not in merged


def test_translate_all_propagates_glossary_and_checkpoints():
    batch1_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": i, "id_text": f"ID-{i}"} for i in range(1, 101)],
            "new_terms": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    batch2_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 101, "id_text": "Pertandingan Ratu Serigala!"}],
            "new_terms": {},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    client = FakeClient([batch1_resp, batch2_resp])
    utts = [U(i, "台詞") for i in range(1, 102)]
    glossary = {}
    saved = {}

    def save_ckpt(idx, data):
        saved[idx] = data

    res = translate_all(
        client, utts, "SYS", FakeTracker(),
        save_checkpoint=save_ckpt,
        glossary=glossary,
    )
    assert res[101] == "Pertandingan Ratu Serigala!"
    assert "人狼クイーン決定戦" in client.prompts[1]
    assert glossary["人狼クイーン決定戦"] == "Pertandingan Ratu Serigala"
    assert saved[1]["glossary"]["人狼クイーン決定戦"] == "Pertandingan Ratu Serigala"


def test_translate_all_loads_legacy_flat_checkpoint():
    utts = [U(1, "こんにちは")]
    legacy_ckpt = {1: "Halo"}
    glossary = {}
    res = translate_all(
        FakeClient([]), utts, "SYS", FakeTracker(),
        load_checkpoint=lambda idx: legacy_ckpt,
        glossary=glossary,
    )
    assert res == {1: "Halo"}
    assert glossary == {}


def test_translate_all_loads_v2_checkpoint_and_propagates_glossary():
    batch1_ckpt = {
        "translations": {str(i): f"ID-{i}" for i in range(1, 101)},
        "glossary": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala", "ignored": None},
    }
    batch2_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 101, "id_text": "Pertandingan Ratu Serigala!"}],
            "new_terms": {},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    client = FakeClient([batch2_resp])
    utts = [U(i, "台詞") for i in range(1, 102)]
    glossary = {}

    def load_ckpt(idx):
        return batch1_ckpt if idx == 1 else None

    res = translate_all(
        client, utts, "SYS", FakeTracker(),
        load_checkpoint=load_ckpt,
        glossary=glossary,
    )
    assert res[1] == "ID-1"
    assert res[101] == "Pertandingan Ratu Serigala!"
    assert len(client.prompts) == 1
    assert "人狼クイーン決定戦" in client.prompts[0]
    assert glossary == {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"}


def test_translate_batch_uses_text_runner_fallback():
    tracker = FakeTracker()
    runner_calls = []

    def fake_runner(system, user):
        runner_calls.append(user)
        return R(GOOD_ITEMS), "fallback-model"

    res = translate_batch(None, UTTS, "SYS", tracker, text_runner=fake_runner)
    assert res == {1: "Halo", 2: "Makasih"}
    assert len(runner_calls) == 1
    assert tracker.calls == [("fallback-model", 10, 10)]


def test_translate_all_with_glossary_none_propagates_terms():
    batch1_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": i, "id_text": f"ID-{i}"} for i in range(1, 101)],
            "new_terms": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    batch2_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 101, "id_text": "Pertandingan Ratu Serigala!"}],
            "new_terms": {},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    client = FakeClient([batch1_resp, batch2_resp])
    utts = [U(i, "台詞") for i in range(1, 102)]

    res = translate_all(client, utts, "SYS", FakeTracker())
    assert res[101] == "Pertandingan Ratu Serigala!"
    assert "人狼クイーン決定戦" in client.prompts[1]
    assert "Pertandingan Ratu Serigala" in client.prompts[1]


def test_translate_all_supports_four_arg_monkeypatch(monkeypatch):
    monkeypatch.setattr(
        "app.translate.translate_batch",
        lambda c, u, s, t: {x.id: f"MOCK-{x.id}" for x in u},
    )
    res = translate_all(FakeClient([]), UTTS, "SYS", FakeTracker())
    assert res == {1: "MOCK-1", 2: "MOCK-2"}


def test_build_user_prompt_includes_two_speaker_instruction():
    utts = [U(1, "- おはよう\n- おはよう")]
    with_terms = _build_user_prompt(utts, include_terms=True)
    without_terms = _build_user_prompt(utts, include_terms=False)

    assert "2 pembicara diawali '- '" in with_terms
    assert "new_terms" in with_terms
    assert "2 pembicara diawali '- '" in without_terms
    assert "new_terms" not in without_terms


def test_translate_all_preserves_two_speaker_prefix():
    utts = [U(1, "- おはよう\n- おはようございます"), U(2, "はい")]
    resp = LLMResponse(
        text=json.dumps({
            "translations": [
                {"id": 1, "id_text": "- Pagi\n- Selamat pagi"},
                {"id": 2, "id_text": "Ya"},
            ],
            "new_terms": {},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    res = translate_all(FakeClient([resp]), utts, "SYS", FakeTracker())
    assert res == {1: "- Pagi\n- Selamat pagi", 2: "Ya"}


def test_translate_batch_text_runner_error_propagates_without_tracking():
    tracker = FakeTracker()

    def failing_runner(system, user):
        raise RuntimeError("runner failed")

    with pytest.raises(RuntimeError, match="runner failed"):
        translate_batch(None, UTTS, "SYS", tracker, text_runner=failing_runner)
    assert tracker.calls == []


def test_translate_batch_text_runner_used_across_retry_and_fallback():
    tracker = FakeTracker()
    prompts = []
    partial = LLMResponse(
        text=json.dumps({"translations": [{"id": 1, "id_text": "Halo"}], "new_terms": {}}),
        input_tokens=10, output_tokens=5,
    )
    single = R([{"id": 2, "id_text": "Makasih"}])
    responses = [partial, partial, single]

    def fake_runner(system, user):
        prompts.append(user)
        return responses.pop(0), "fallback-model"

    res = translate_batch(None, UTTS, "SYS", tracker, text_runner=fake_runner)
    assert res == {1: "Halo", 2: "Makasih"}
    assert len(prompts) == 3
    assert "new_terms" in prompts[0]
    assert "Output sebelumnya SALAH" in prompts[1]
    assert "new_terms" not in prompts[2]
    assert "array JSON" in prompts[2]
    assert len(tracker.calls) == 3


def test_translate_batch_preserves_canonical_term_on_drift():
    glossary = {"人狼": "Manusia Serigala"}
    resp = LLMResponse(
        text=json.dumps({
            "translations": GOOD_ITEMS,
            "new_terms": {"人狼": "Serigala Jadi-jadian", "鬼": "Oni"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    res = translate_batch(FakeClient([resp]), UTTS, "SYS", FakeTracker(), glossary=glossary)
    assert res == {1: "Halo", 2: "Makasih"}
    assert glossary == {"人狼": "Manusia Serigala", "鬼": "Oni"}


def test_translate_batch_merges_initial_and_retry_terms_on_recovery():
    glossary = {"既存": "Tetap"}
    attempt1 = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 1, "id_text": "Halo"}],
            "new_terms": {"既存": "Drift1", "A": "AwalA", "B": "AwalB"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=5,
    )
    attempt2 = LLMResponse(
        text=json.dumps({
            "translations": GOOD_ITEMS,
            "new_terms": {"既存": "Drift2", "A": "RevisiA", "C": "BaruC"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    res = translate_batch(
        FakeClient([attempt1, attempt2]), UTTS, "SYS", FakeTracker(), glossary=glossary
    )
    assert res == {1: "Halo", 2: "Makasih"}
    assert glossary == {
        "既存": "Tetap",
        "A": "RevisiA",
        "B": "AwalB",
        "C": "BaruC",
    }


def test_translate_batch_leaves_glossary_untouched_on_translate_error():
    glossary = {"既存": "Tetap"}
    bad = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 1, "id_text": "Halo"}],
            "new_terms": {"A": "BaruA"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=5,
    )
    garbage = LLMResponse(text="not json", input_tokens=1, output_tokens=1)
    client = FakeClient([bad, bad, garbage])
    with pytest.raises(TranslateError):
        translate_batch(client, UTTS, "SYS", FakeTracker(), glossary=glossary)
    assert glossary == {"既存": "Tetap"}


def test_translate_batch_empty_utterances_returns_empty():
    client = FakeClient([])
    tracker = FakeTracker()
    assert translate_batch(client, [], "SYS", tracker) == {}
    assert tracker.calls == []


def test_parse_translation_response_skips_invalid_id_and_text_types():
    raw = json.dumps({
        "translations": [
            {"id": 1, "id_text": "Valid"},
            {"id": True, "id_text": "BoolId"},
            {"id": 2.5, "id_text": "FloatId"},
            {"id": [3], "id_text": "ListId"},
            {"id": {"x": 4}, "id_text": "DictId"},
            {"id": 5, "id_text": ["ListText"]},
            {"id": 6, "id_text": True},
            {"id": 7, "id_text": {"nested": "DictText"}},
        ]
    })
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Valid"}
    assert terms == {}


def test_parse_translation_response_raises_on_none():
    with pytest.raises(ValueError):
        parse_translation_response(None)


@pytest.mark.parametrize("corrupt", [
    None,
    ["not", "a", "dict"],
    {"translations": None},
    {"translations": []},
    {"translations": {"abc": "Halo"}},
    {"translations": {"1.5": "Halo"}},
    {"translations": {True: "Halo"}},
    {"translations": {1.5: "Halo"}},
    {"translations": {"1": None}},
    {"translations": {"1": ["list"]}},
    {"translations": {"1": {"nested": 1}}},
    {"translations": {"1": True}},
    {"translations": {"1": 123}},
    {"translations": {"1": "   "}},
    {"bad_key": "Halo"},
    {"1": None},
])
def test_unpack_checkpoint_returns_none_on_corrupt_data(corrupt):
    assert _unpack_checkpoint(corrupt) is None


def test_translate_all_reruns_corrupt_checkpoint():
    client = FakeClient([R(GOOD_ITEMS)])
    saved = {}
    res = translate_all(
        client, UTTS, "SYS", FakeTracker(),
        load_checkpoint=lambda idx: {"translations": None},
        save_checkpoint=lambda idx, data: saved.__setitem__(idx, data),
    )
    assert res == {1: "Halo", 2: "Makasih"}
    assert len(client.prompts) == 1
    assert saved[1]["translations"] == {1: "Halo", 2: "Makasih"}


def test_merge_glossary_limit_zero_returns_empty():
    assert merge_glossary({"a": "b"}, {"c": "d"}, limit=0) == {}
