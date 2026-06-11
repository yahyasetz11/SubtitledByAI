import pytest

from app.transcribe import (
    TranscriptParseError, Utterance, build_transcribe_prompts,
    parse_timestamp, parse_transcript,
)


def test_parse_timestamp_mm_ss_mmm():
    assert parse_timestamp("01:23.450") == pytest.approx(83.45)


def test_parse_timestamp_h_mm_ss():
    assert parse_timestamp("1:02:03.5") == pytest.approx(3723.5)


def test_parse_timestamp_plain_seconds():
    assert parse_timestamp("45.2") == pytest.approx(45.2)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("abc")


VALID = """[
  {"id": 1, "start": "00:01.000", "end": "00:02.500",
   "type": "dialogue", "ja": "こんにちは"},
  {"id": 2, "start": "00:03.000", "end": "00:04.000",
   "type": "narration", "ja": "今日の企画は"}
]"""


def test_parse_transcript_valid():
    utts = parse_transcript(VALID)
    assert utts == [
        Utterance(id=1, start=1.0, end=2.5, type="dialogue", ja="こんにちは"),
        Utterance(id=2, start=3.0, end=4.0, type="narration", ja="今日の企画は"),
    ]


def test_parse_transcript_strips_code_fences_and_wrapper_dict():
    fenced = '```json\n{"utterances": ' + VALID + '}\n```'
    assert len(parse_transcript(fenced)) == 2


def test_parse_transcript_skips_bad_items_and_normalizes():
    text = """[
      {"id": 1, "start": "00:01.0", "end": "00:00.5", "type": "vo", "ja": "あ"},
      {"id": 2, "start": "bad", "end": "00:04.0", "type": "dialogue", "ja": "い"},
      {"id": 3, "start": "00:05.0", "end": "00:06.0", "type": "dialogue", "ja": ""}
    ]"""
    utts = parse_transcript(text)
    assert len(utts) == 1                    # items 2 (bad time) & 3 (empty) skipped
    assert utts[0].type == "dialogue"        # unknown type normalized
    assert utts[0].end == pytest.approx(1.5) # end<=start clamped to start+0.5


def test_parse_transcript_raises_on_unparseable_json():
    with pytest.raises(TranscriptParseError):
        parse_transcript('[{"id": 1, "start": "00:01.0", "end": "00:0')  # truncated


def test_parse_transcript_raises_when_empty():
    with pytest.raises(TranscriptParseError):
        parse_transcript("[]")


def test_build_transcribe_prompts_injects_context():
    system, user = build_transcribe_prompts("CONTEXT-BODY", "MEMBERS-BODY")
    assert "CONTEXT-BODY" in system
    assert "MEMBERS-BODY" in system
    assert "MM:SS.mmm" in user
    assert "JSON" in user
    assert "narration" in user
