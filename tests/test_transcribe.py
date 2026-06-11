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


from app.audio import PlannedChunk
from app.transcribe import merge_transcripts


def U(id, start, end, ja, type="dialogue"):
    return Utterance(id=id, start=start, end=end, type=type, ja=ja)


def test_merge_shifts_by_offset_and_renumbers():
    chunks = [PlannedChunk(1, 0.0, 700.0, False),
              PlannedChunk(2, 700.0, 1400.0, False)]
    per_chunk = [
        [U(1, 1.0, 2.0, "あ"), U(2, 5.0, 6.0, "い")],
        [U(1, 0.5, 1.5, "う")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert [u.id for u in merged] == [1, 2, 3]
    assert merged[2].start == pytest.approx(700.5)
    assert merged[2].ja == "う"


def test_merge_dedupes_overlap_zone():
    # chunk 2 re-covers the last 4s of chunk 1 (hard cut at 840)
    chunks = [PlannedChunk(1, 0.0, 840.0, False),
              PlannedChunk(2, 836.0, 1400.0, True)]
    per_chunk = [
        [U(1, 830.0, 833.0, "今日は楽しかったですね")],
        [U(1, 0.2, 2.8, "今日は楽しかったですね。"),  # dupe (similar text)
         U(2, 10.0, 12.0, "次のコーナーです")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert [u.ja for u in merged] == ["今日は楽しかったですね", "次のコーナーです"]


def test_merge_keeps_different_text_in_overlap_zone():
    chunks = [PlannedChunk(1, 0.0, 840.0, False),
              PlannedChunk(2, 836.0, 1400.0, True)]
    per_chunk = [
        [U(1, 830.0, 833.0, "全然違う話")],
        [U(1, 0.2, 2.8, "新しいセリフです")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert len(merged) == 2


from app.providers import LLMResponse
from app.transcribe import transcribe_chunk


class FakeGemini:
    """Returns queued LLMResponses; records uploads."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.uploads = []
        self.model = "gemini-2.5-pro"

    def upload_audio(self, path):
        self.uploads.append(path)
        return f"file-ref:{path}"

    def generate(self, system, user, audio=None):
        return self.responses.pop(0)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, input_tokens, output_tokens):
        self.calls.append((model, input_tokens, output_tokens))


GOOD = LLMResponse(text=VALID, input_tokens=100, output_tokens=50)
HALF = LLMResponse(
    text='[{"id":1,"start":"00:01.0","end":"00:02.0","type":"dialogue","ja":"あ"}]',
    input_tokens=10, output_tokens=5)


def test_transcribe_chunk_happy_path(tmp_path):
    gemini = FakeGemini([GOOD])
    tracker = FakeTracker()
    utts = transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                            "sys", "user", tracker)
    assert len(utts) == 2
    assert tracker.calls == [("gemini-2.5-pro", 100, 50)]
    assert gemini.uploads == [tmp_path / "chunk_001.mp3"]


def test_transcribe_chunk_splits_on_truncation(tmp_path, monkeypatch):
    truncated = LLMResponse(text=VALID, input_tokens=1, output_tokens=1,
                            truncated=True)
    gemini = FakeGemini([truncated, HALF, HALF])

    def fake_split(path):
        return (path.with_name(path.stem + "_a.mp3"),
                path.with_name(path.stem + "_b.mp3"), 420.0)

    monkeypatch.setattr("app.transcribe.audio.split_audio", fake_split)
    utts = transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                            "sys", "user", FakeTracker())
    assert len(utts) == 2
    assert utts[1].start == pytest.approx(421.0)  # second half shifted by 420
    assert len(gemini.uploads) == 3


def test_transcribe_chunk_gives_up_after_max_depth(tmp_path, monkeypatch):
    bad = LLMResponse(text="not json", input_tokens=1, output_tokens=1)
    gemini = FakeGemini([bad] * 7)
    monkeypatch.setattr(
        "app.transcribe.audio.split_audio",
        lambda p: (p.with_name(p.stem + "_a.mp3"),
                   p.with_name(p.stem + "_b.mp3"), 60.0))
    with pytest.raises(TranscriptParseError):
        transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                         "sys", "user", FakeTracker())
