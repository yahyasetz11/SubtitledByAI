import json

import pytest

from app import providers
from app.audio import PlannedChunk
from app.providers import LLMResponse
from app.timing import correct_chunk_timing, parse_timing_response, reanchor_all
from app.transcribe import Utterance


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, inp, out):
        self.calls.append((model, inp, out))


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeGemini:
    def __init__(self, responses=None, upload_error=None, model="gemini-2.5-flash"):
        self.responses = list(responses or [])
        self.upload_error = upload_error
        self.model = model
        self.uploads = []
        self.prompts = []

    def upload_audio(self, path):
        self.uploads.append(path)
        if self.upload_error is not None:
            raise self.upload_error
        return f"audio-ref:{path.name}"

    def generate(self, system, user, audio=None):
        self.prompts.append((system, user, audio))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_correct_chunk_timing_propagates_api_error_when_runner_fails(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]

    def failing_runner(path, system, user):
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        correct_chunk_timing(
            None, chunk_path, utts, 0.0, FakeTracker(),
            audio_runner=failing_runner,
        )


def test_correct_chunk_timing_returns_fallback_on_parse_mismatch(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]
    tracker = FakeTracker()

    def bad_json_runner(path, system, user):
        return LLMResponse(text="not json", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    corrected, success = correct_chunk_timing(
        None, chunk_path, utts, 0.0, tracker,
        audio_runner=bad_json_runner,
    )
    assert success is False
    assert corrected == utts
    assert tracker.calls == [("gemini-3.6-flash", 5, 2)]


def test_correct_chunk_timing_default_gemini_happy_path(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 11.0, 12.0, "dialogue", "あ")]
    resp = LLMResponse(
        text='[{"id": 1, "start": 1.25, "end": 2.5}]',
        input_tokens=20, output_tokens=10,
    )
    gemini = FakeGemini([resp], model="gemini-2.5-flash")
    tracker = FakeTracker()

    corrected, success = correct_chunk_timing(
        gemini, chunk_path, utts, 10.0, tracker,
    )
    assert success is True
    assert corrected == [Utterance(1, 11.25, 12.5, "dialogue", "あ")]
    assert gemini.uploads == [chunk_path]
    assert len(gemini.prompts) == 1
    assert '"start": 1.0' in gemini.prompts[0][1]
    assert tracker.calls == [("gemini-2.5-flash", 20, 10)]


def test_correct_chunk_timing_default_gemini_propagates_upload_error(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]
    gemini = FakeGemini(upload_error=FakeHTTPError(503))
    tracker = FakeTracker()

    with pytest.raises(FakeHTTPError) as exc_info:
        correct_chunk_timing(gemini, chunk_path, utts, 0.0, tracker)
    assert exc_info.value.status_code == 503
    assert tracker.calls == []


def test_correct_chunk_timing_default_gemini_propagates_generate_error(tmp_path, monkeypatch):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]
    gemini = FakeGemini([FakeHTTPError(503)] * 4)
    tracker = FakeTracker()

    monkeypatch.setattr(
        "app.timing.call_with_retries",
        lambda fn: providers.call_with_retries(fn, sleep=lambda _: None),
    )

    with pytest.raises(FakeHTTPError) as exc_info:
        correct_chunk_timing(gemini, chunk_path, utts, 0.0, tracker)
    assert exc_info.value.status_code == 503
    assert len(gemini.prompts) == 4
    assert tracker.calls == []


def test_correct_chunk_timing_default_gemini_propagates_non_retryable_error(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]
    gemini = FakeGemini([FakeHTTPError(400)])
    tracker = FakeTracker()

    with pytest.raises(FakeHTTPError) as exc_info:
        correct_chunk_timing(gemini, chunk_path, utts, 0.0, tracker)
    assert exc_info.value.status_code == 400
    assert len(gemini.prompts) == 1
    assert tracker.calls == []


def test_parse_timing_response_clamps_end_le_start():
    payload = json.dumps([
        {"id": 1, "start": 2.0, "end": 2.0},
        {"id": 2, "start": 5.0, "end": 3.5},
        {"id": 3, "start": 6.0, "end": 7.2},
    ])
    parsed = parse_timing_response(payload, [1, 2, 3])
    assert parsed == {
        1: (2.0, 2.5),
        2: (5.0, 5.5),
        3: (6.0, 7.2),
    }


@pytest.mark.parametrize("payload,expected_ids", [
    ('[{"id": 1, "start": 0.0, "end": 1.0}]', [1, 2]),          # missing ID
    ('[{"id": 1, "start": 0.0, "end": 1.0}, {"id": 99, "start": 1.0, "end": 2.0}]', [1]),  # extra ID
    ('[{"id": 2, "start": 0.0, "end": 1.0}]', [1]),             # wrong ID
])
def test_parse_timing_response_returns_none_on_id_mismatch(payload, expected_ids):
    assert parse_timing_response(payload, expected_ids) is None


@pytest.mark.parametrize("payload", [
    "not json",
    '[{"id": 1, "start": 0.0',
    "{}",
    '"just a string"',
    "123",
    '[{"id": 1, "start": "bad", "end": 1.0}]',
])
def test_parse_timing_response_returns_none_on_malformed_json(payload):
    assert parse_timing_response(payload, [1]) is None


def test_parse_timing_response_accepts_fences_and_wrapper_dict():
    text = '```json\n{"items": [{"id": 1, "start": 0.5, "end": 1.5}]}\n```'
    assert parse_timing_response(text, [1]) == {1: (0.5, 1.5)}


def test_reanchor_all_uses_per_chunk_checkpoints(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_001.mp3").write_bytes(b"a")
    (chunks_dir / "chunk_002.mp3").write_bytes(b"b")
    chunks = [
        PlannedChunk(1, 0.0, 10.0, False),
        PlannedChunk(2, 10.0, 20.0, False),
    ]
    utts = [
        Utterance(1, 1.0, 2.0, "dialogue", "一"),
        Utterance(2, 11.0, 12.0, "dialogue", "二"),
    ]
    saved = {}
    runner_calls = []

    def load_ckpt(idx):
        if idx == 1:
            return {
                "utterances": [{"id": 1, "start": 1.2, "end": 2.2, "type": "dialogue", "ja": "一"}],
                "report_entry": {"chunk": 1, "status": "corrected", "count": 1},
            }
        return None

    def save_ckpt(idx, payload):
        saved[idx] = payload

    def fake_runner(path, system, user):
        runner_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 2, "start": 1.5, "end": 2.5}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner, load_checkpoint=load_ckpt, save_checkpoint=save_ckpt,
    )
    assert runner_calls == ["chunk_002.mp3"]  # chunk 1 skipped via checkpoint
    assert result[0].start == pytest.approx(1.2)
    assert result[1].start == pytest.approx(11.5)
    assert 2 in saved and saved[2]["report_entry"]["status"] == "corrected"
    assert [r["chunk"] for r in report] == [1, 2]


def test_reanchor_all_propagates_error_after_saving_prior_checkpoint(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_001.mp3").write_bytes(b"a")
    (chunks_dir / "chunk_002.mp3").write_bytes(b"b")
    chunks = [
        PlannedChunk(1, 0.0, 10.0, False),
        PlannedChunk(2, 10.0, 20.0, False),
    ]
    utts = [
        Utterance(1, 1.0, 2.0, "dialogue", "一"),
        Utterance(2, 11.0, 12.0, "dialogue", "二"),
    ]
    saved = {}
    progress_calls = []

    def save_ckpt(idx, payload):
        saved[idx] = payload

    def flaky_runner(path, system, user):
        if path.name == "chunk_001.mp3":
            return LLMResponse(
                text='[{"id": 1, "start": 1.3, "end": 2.3}]',
                input_tokens=10, output_tokens=5,
            ), "gemini-3.5-flash"
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        reanchor_all(
            utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
            on_progress=lambda idx, n: progress_calls.append((idx, n)),
            audio_runner=flaky_runner,
            load_checkpoint=saved.get,
            save_checkpoint=save_ckpt,
        )

    assert 1 in saved
    assert saved[1]["report_entry"]["status"] == "corrected"
    assert saved[1]["utterances"][0]["start"] == pytest.approx(1.3)
    assert 2 not in saved
    assert progress_calls == [(1, 2)]

    # Resume using the saved checkpoint: chunk 1 is not re-sent, chunk 2 runs and completes.
    resumed_calls = []

    def recovering_runner(path, system, user):
        resumed_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 2, "start": 1.4, "end": 2.4}]',
            input_tokens=8, output_tokens=4,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=recovering_runner,
        load_checkpoint=saved.get,
        save_checkpoint=save_ckpt,
    )
    assert resumed_calls == ["chunk_002.mp3"]
    assert [u.start for u in result] == [pytest.approx(1.3), pytest.approx(11.4)]
    assert [r["chunk"] for r in report] == [1, 2]


def test_reanchor_all_checkpoints_skipped_and_missing_chunk_files(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks = [
        PlannedChunk(1, 0.0, 10.0, False),   # no utterances -> skipped
        PlannedChunk(2, 10.0, 20.0, False),  # missing mp3 -> fallback
    ]
    utts = [Utterance(1, 11.0, 12.0, "dialogue", "二")]
    saved = {}

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )

    assert len(result) == 1
    assert result[0].start == pytest.approx(11.0)
    assert report == [
        {"chunk": 1, "status": "skipped", "reason": "no_utterances"},
        {"chunk": 2, "status": "fallback", "reason": "chunk_file_missing"},
    ]
    assert saved[1] == {
        "utterances": [],
        "report_entry": {"chunk": 1, "status": "skipped", "reason": "no_utterances"},
    }
    assert saved[2]["report_entry"]["reason"] == "chunk_file_missing"
    assert len(saved[2]["utterances"]) == 1
