import json
from pathlib import Path

import pytest

from app import providers
from app.audio import PlannedChunk
from app.providers import LLMResponse
from app.timing import (
    build_timing_system,
    build_timing_user_prompt,
    correct_chunk_timing,
    load_timing,
    parse_timing_response,
    reanchor_all,
    save_timing,
)
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


def _utt(
    uid: int = 1,
    start: float = 1.0,
    end: float = 2.0,
    ja: str = "あ",
    utype: str = "dialogue",
) -> Utterance:
    return Utterance(uid, start, end, utype, ja)


def _make_chunk_file(directory: Path, index: int = 1, content: bytes = b"mp3") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"chunk_{index:03d}.mp3"
    path.write_bytes(content)
    return path


def _make_chunks(*spans: tuple[int, float, float]) -> list[PlannedChunk]:
    return [PlannedChunk(idx, start, end, False) for idx, start, end in spans]


def test_correct_chunk_timing_propagates_api_error_when_runner_fails(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt()]

    def failing_runner(path, system, user):
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        correct_chunk_timing(
            None, chunk_path, utts, 0.0, FakeTracker(),
            audio_runner=failing_runner,
        )


def test_correct_chunk_timing_returns_fallback_on_parse_mismatch(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt()]
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
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt(1, 11.0, 12.0)]
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
    assert corrected == [_utt(1, 11.25, 12.5)]
    assert gemini.uploads == [chunk_path]
    assert len(gemini.prompts) == 1
    system, user, audio = gemini.prompts[0]
    assert system == ""
    assert '"start": 1.0' in user
    assert audio == "audio-ref:chunk_001.mp3"
    assert tracker.calls == [("gemini-2.5-flash", 20, 10)]


def test_correct_chunk_timing_audio_runner_happy_path_and_none_tracker(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt(1, 101.0, 102.5)]
    runner_calls = []

    def runner(path, system, user):
        runner_calls.append((path, system, user))
        return LLMResponse(
            text='[{"id": 1, "start": 1.4, "end": 2.9}]',
            input_tokens=12, output_tokens=6,
        ), "gemini-3.5-flash"

    corrected, success = correct_chunk_timing(
        None, chunk_path, utts, 100.0, tracker=None,
        audio_runner=runner, max_duration=30.0,
    )
    assert success is True
    assert corrected == [_utt(1, 101.4, 102.9)]
    assert len(runner_calls) == 1
    assert runner_calls[0][0] == chunk_path
    assert runner_calls[0][1] == ""
    assert '"start": 1.0' in runner_calls[0][2]


def test_correct_chunk_timing_empty_utterances_makes_zero_calls(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    gemini = FakeGemini()
    tracker = FakeTracker()

    def unexpected_runner(path, system, user):
        raise AssertionError("audio_runner should not be called for empty utterances")

    corrected_gemini, ok_gemini = correct_chunk_timing(
        gemini, chunk_path, [], 0.0, tracker,
    )
    corrected_runner, ok_runner = correct_chunk_timing(
        None, chunk_path, [], 0.0, tracker, audio_runner=unexpected_runner,
    )
    assert (corrected_gemini, ok_gemini) == ([], True)
    assert (corrected_runner, ok_runner) == ([], True)
    assert gemini.uploads == []
    assert gemini.prompts == []
    assert tracker.calls == []


def test_correct_chunk_timing_rejects_out_of_bounds_max_duration(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt(1, 101.0, 102.0)]

    def leaked_abs_runner(path, system, user):
        return LLMResponse(
            text='[{"id": 1, "start": 101.5, "end": 102.5}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    corrected, success = correct_chunk_timing(
        None, chunk_path, utts, 100.0, FakeTracker(),
        audio_runner=leaked_abs_runner, max_duration=10.0,
    )
    assert success is False
    assert corrected == utts


def test_correct_chunk_timing_default_gemini_propagates_upload_error(tmp_path):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt()]
    gemini = FakeGemini(upload_error=FakeHTTPError(503))
    tracker = FakeTracker()

    with pytest.raises(FakeHTTPError) as exc_info:
        correct_chunk_timing(gemini, chunk_path, utts, 0.0, tracker)
    assert exc_info.value.status_code == 503
    assert tracker.calls == []


def test_correct_chunk_timing_default_gemini_propagates_generate_error(tmp_path, monkeypatch):
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt()]
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
    chunk_path = _make_chunk_file(tmp_path)
    utts = [_utt()]
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


@pytest.mark.parametrize("payload", [
    None,
    '[{"id": 1, "start": -0.1, "end": 1.0}]',
    '[{"id": 1, "start": "NaN", "end": 1.0}]',
    '[{"id": 1, "start": 0.0, "end": "inf"}]',
    '[{"id": 1, "start": "-inf", "end": 1.0}]',
    '[{"id": 1, "start": NaN, "end": 1.0}]',
    '[{"id": 1, "start": 0.0, "end": Infinity}]',
    '[{"id": true, "start": 0.0, "end": 1.0}]',
    '[{"id": 1, "start": false, "end": 1.0}]',
    '[{"id": 1, "start": 0.0, "end": true}]',
    '[{"id": 1.5, "start": 0.0, "end": 1.0}]',
    '[{"id": 1, "start": 0.0, "end": 1.0}, {"id": 1, "start": 2.0, "end": 3.0}]',
])
def test_parse_timing_response_rejects_invalid_values(payload):
    assert parse_timing_response(payload, [1]) is None


def test_parse_timing_response_max_duration_bounds():
    # Absolute episode timestamp leaked into a 10s chunk response
    leaked = '[{"id": 1, "start": 310.0, "end": 311.5}]'
    assert parse_timing_response(leaked, [1], max_duration=10.0) is None

    assert parse_timing_response(
        '[{"id": 1, "start": 10.6, "end": 11.0}]', [1], max_duration=10.0
    ) is None
    assert parse_timing_response(
        '[{"id": 1, "start": 9.5, "end": 12.1}]', [1], max_duration=10.0
    ) is None
    assert parse_timing_response(
        '[{"id": 1, "start": 10.5, "end": 12.0}]', [1], max_duration=10.0
    ) == {1: (10.5, 12.0)}


def test_parse_timing_response_accepts_fences_and_wrapper_dict():
    text = '```json\n{"items": [{"id": 1, "start": 0.5, "end": 1.5}]}\n```'
    assert parse_timing_response(text, [1]) == {1: (0.5, 1.5)}


def test_reanchor_all_uses_per_chunk_checkpoints(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1, b"a")
    _make_chunk_file(chunks_dir, 2, b"b")
    chunks = _make_chunks((1, 0.0, 10.0), (2, 10.0, 20.0))
    utts = [_utt(1, 1.0, 2.0, "一"), _utt(2, 11.0, 12.0, "二")]
    saved = {}
    runner_calls = []

    def load_ckpt(idx):
        if idx == 1:
            return {
                "utterances": [_utt(1, 1.2, 2.2, "一").to_dict()],
                "report_entry": {"chunk": 1, "status": "corrected", "count": 1},
            }
        return None

    def fake_runner(path, system, user):
        runner_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 2, "start": 1.5, "end": 2.5}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner, load_checkpoint=load_ckpt,
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert runner_calls == ["chunk_002.mp3"]  # chunk 1 skipped via checkpoint
    assert result[0].start == pytest.approx(1.2)
    assert result[1].start == pytest.approx(11.5)
    assert 2 in saved and saved[2]["report_entry"]["status"] == "corrected"
    assert [r["chunk"] for r in report] == [1, 2]


@pytest.mark.parametrize("bad_ckpt", [
    {},
    [],
    {"utterances": None, "report_entry": {"chunk": 1, "status": "corrected"}},
    {"utterances": [], "report_entry": {"chunk": 1, "status": "corrected", "count": 0}},
    {"utterances": [_utt(1, 1.0, 2.0, "一").to_dict()], "report_entry": "bad"},
    {"utterances": [_utt(1, 1.0, 2.0, "一").to_dict()], "report_entry": {"chunk": 1}},
    {
        "utterances": [{"id": 1, "start": float("nan"), "end": 2.0, "type": "dialogue", "ja": "一"}],
        "report_entry": {"chunk": 1, "status": "corrected"},
    },
])
def test_reanchor_all_recomputes_on_corrupt_checkpoint(tmp_path, bad_ckpt):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    chunks = _make_chunks((1, 0.0, 10.0))
    utts = [_utt(1, 1.0, 2.0, "一")]
    saved = {}
    runner_calls = []

    def fake_runner(path, system, user):
        runner_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 1, "start": 1.6, "end": 2.6}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner,
        load_checkpoint=lambda idx: bad_ckpt,
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert runner_calls == ["chunk_001.mp3"]
    assert result == [_utt(1, 1.6, 2.6, "一")]
    assert report == [{"chunk": 1, "status": "corrected", "count": 1}]
    assert saved[1]["report_entry"]["status"] == "corrected"
    assert saved[1]["utterances"][0]["start"] == pytest.approx(1.6)


@pytest.mark.parametrize("stale_ckpt", [
    {"utterances": [], "report_entry": {"chunk": 1, "status": "skipped", "reason": "no_utterances"}},
    {"utterances": [_utt(1, 1.0, 2.0, "一").to_dict()], "report_entry": {"chunk": 1, "status": "skipped"}},
])
def test_reanchor_all_recomputes_skipped_checkpoint_when_chunk_has_utterances(tmp_path, stale_ckpt):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    chunks = _make_chunks((1, 0.0, 10.0))
    utts = [_utt(1, 1.0, 2.0, "一")]
    saved = {}
    runner_calls = []

    def fake_runner(path, system, user):
        runner_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 1, "start": 1.6, "end": 2.6}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner,
        load_checkpoint=lambda idx: stale_ckpt,
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert runner_calls == ["chunk_001.mp3"]
    assert result == [_utt(1, 1.6, 2.6, "一")]
    assert report == [{"chunk": 1, "status": "corrected", "count": 1}]
    assert saved[1]["report_entry"]["status"] == "corrected"


@pytest.mark.parametrize("stale_ckpt", [
    {"utterances": [_utt(1, 1.2, 2.2, "幻").to_dict()], "report_entry": {"chunk": 1, "status": "corrected", "count": 1}},
    {"utterances": [_utt(1, 1.2, 2.2, "幻").to_dict()], "report_entry": {"chunk": 1, "status": "skipped", "reason": "no_utterances"}},
])
def test_reanchor_all_returns_skipped_when_empty_chunk_has_non_empty_checkpoint(tmp_path, stale_ckpt):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    chunks = _make_chunks((1, 0.0, 10.0))
    saved = {}

    def unexpected_runner(path, system, user):
        raise AssertionError("audio_runner should not be called for empty chunk")

    result, report = reanchor_all(
        [], chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=unexpected_runner,
        load_checkpoint=lambda idx: stale_ckpt,
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert result == []
    assert report == [{"chunk": 1, "status": "skipped", "reason": "no_utterances"}]
    assert saved[1] == {
        "utterances": [],
        "report_entry": {"chunk": 1, "status": "skipped", "reason": "no_utterances"},
    }


def test_reanchor_all_propagates_error_after_saving_prior_checkpoint(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1, b"a")
    _make_chunk_file(chunks_dir, 2, b"b")
    chunks = _make_chunks((1, 0.0, 10.0), (2, 10.0, 20.0))
    utts = [_utt(1, 1.0, 2.0, "一"), _utt(2, 11.0, 12.0, "二")]
    saved = {}
    progress_calls = []

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
            save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
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
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert resumed_calls == ["chunk_002.mp3"]
    assert [u.start for u in result] == [pytest.approx(1.3), pytest.approx(11.4)]
    assert [r["chunk"] for r in report] == [1, 2]


def test_reanchor_all_checkpoints_skipped_and_missing_chunk_files(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunks = _make_chunks(
        (1, 0.0, 10.0),   # no utterances -> skipped
        (2, 10.0, 20.0),  # missing mp3 -> fallback
    )
    utts = [_utt(1, 11.0, 12.0, "二")]
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


def test_reanchor_all_empty_chunks_preserves_sorts_and_renumbers(tmp_path):
    utts = [_utt(9, 8.0, 9.0, "後"), _utt(4, 1.0, 2.0, "先")]
    result, report = reanchor_all(
        utts, [], tmp_path / "chunks", gemini=None, tracker=FakeTracker(),
    )
    assert report == []
    assert result == [_utt(1, 1.0, 2.0, "先"), _utt(2, 8.0, 9.0, "後")]


def test_reanchor_all_routes_pre_first_chunk_and_boundary_timestamps(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    _make_chunk_file(chunks_dir, 2)
    _make_chunk_file(chunks_dir, 3)
    chunks = _make_chunks((1, 10.0, 20.0), (2, 20.0, 30.0), (3, 30.0, 40.0))
    utts = [
        _utt(1, 9.5, 10.5, "前"),    # < chunks[0].start -> must route to chunk 1, not chunk 3
        _utt(2, 20.0, 21.0, "境"),   # == chunk 1 end -> routes to chunk 2
        _utt(3, 40.0, 41.0, "末"),   # == last chunk end -> routes to chunk 3
    ]
    seen_by_chunk = {}

    def fake_runner(path, system, user):
        idx = int(path.stem.split("_")[1])
        seen_by_chunk[idx] = user
        if idx == 1:
            text = '[{"id": 1, "start": 0.0, "end": 1.0}]'
        elif idx == 2:
            text = '[{"id": 2, "start": 0.2, "end": 1.2}]'
        else:
            text = '[{"id": 3, "start": 9.8, "end": 10.8}]'
        return LLMResponse(text=text, input_tokens=5, output_tokens=3), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner,
    )
    assert set(seen_by_chunk) == {1, 2, 3}
    assert "前" in seen_by_chunk[1]
    assert "境" in seen_by_chunk[2]
    assert "末" in seen_by_chunk[3]
    assert [r["status"] for r in report] == ["corrected", "corrected", "corrected"]
    assert [u.start for u in result] == [
        pytest.approx(10.0), pytest.approx(20.2), pytest.approx(39.8),
    ]


def test_reanchor_all_checkpoints_gemini_parse_error_and_respects_max_duration(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    _make_chunk_file(chunks_dir, 2)
    chunks = _make_chunks((1, 0.0, 10.0), (2, 10.0, 20.0))
    utts = [_utt(1, 1.0, 2.0, "一"), _utt(2, 11.0, 12.0, "二")]
    saved = {}
    calls = []

    def runner(path, system, user):
        calls.append(path.name)
        if path.name == "chunk_001.mp3":
            return LLMResponse(text="not json", input_tokens=4, output_tokens=2), "gemini-3.5-flash"
        # Absolute timestamp 15.0 exceeds chunk 2 max_duration=10.0 (+0.5/+2.0)
        return LLMResponse(
            text='[{"id": 2, "start": 15.0, "end": 16.0}]',
            input_tokens=4, output_tokens=2,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=runner,
        save_checkpoint=lambda idx, payload: saved.__setitem__(idx, payload),
    )
    assert result == utts
    assert report == [
        {"chunk": 1, "status": "fallback", "reason": "gemini_parse_error"},
        {"chunk": 2, "status": "fallback", "reason": "gemini_parse_error"},
    ]
    assert saved[1]["report_entry"] == {"chunk": 1, "status": "fallback", "reason": "gemini_parse_error"}
    assert saved[2]["report_entry"] == {"chunk": 2, "status": "fallback", "reason": "gemini_parse_error"}

    # Resuming with saved fallback checkpoints does not invoke runner again.
    calls.clear()
    resumed, resumed_report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=runner, load_checkpoint=saved.get,
    )
    assert calls == []
    assert resumed == utts
    assert resumed_report == report


def test_reanchor_all_sorts_and_renumbers_sequential_ids(tmp_path):
    chunks_dir = tmp_path / "chunks"
    _make_chunk_file(chunks_dir, 1)
    _make_chunk_file(chunks_dir, 2)
    chunks = _make_chunks((1, 0.0, 10.0), (2, 10.0, 20.0))
    utts = [_utt(10, 9.0, 9.5, "一"), _utt(25, 10.5, 11.5, "二")]

    def runner(path, system, user):
        if path.name == "chunk_001.mp3":
            # Shifts within +0.5s tolerance past chunk 2's corrected start (10.3 > 10.1)
            return LLMResponse(
                text='[{"id": 10, "start": 10.3, "end": 10.8}]',
                input_tokens=5, output_tokens=3,
            ), "gemini-3.5-flash"
        return LLMResponse(
            text='[{"id": 25, "start": 0.1, "end": 0.6}]',
            input_tokens=5, output_tokens=3,
        ), "gemini-3.5-flash"

    result, _ = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=runner,
    )
    assert result == [
        _utt(1, 10.1, 10.6, "二"),
        _utt(2, 10.3, 10.8, "一"),
    ]


def test_save_and_load_timing_roundtrip(tmp_path):
    utts = [_utt(1, 1.25, 2.5, "こんにちは"), _utt(2, 3.0, 4.5, "ナレーション", "narration")]
    report = [{"chunk": 1, "status": "corrected", "count": 2}]

    save_timing(utts, report, tmp_path)

    loaded = load_timing(tmp_path)
    assert loaded == [u.to_dict() for u in utts]
    saved_report = json.loads((tmp_path / "timing" / "report.json").read_text(encoding="utf-8"))
    assert saved_report == report


def test_build_timing_prompts():
    system = build_timing_system("acara info", "roster member")
    assert "acara info" in system
    assert "roster member" in system
    assert "JANGAN ubah teks ja" in system

    prompt = build_timing_user_prompt([_utt(7, 1.23456, 2.34567, "テスト")])
    assert '"id": 7' in prompt
    assert '"start": 1.235' in prompt
    assert '"end": 2.346' in prompt
    assert '"ja": "テスト"' in prompt
