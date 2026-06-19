import json

import pytest

from app.providers import LLMResponse
from app.transcribe import Utterance
from app.translate import (
    TranslateError, build_translate_system, parse_translation,
    translate_batch, validate_alignment,
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
