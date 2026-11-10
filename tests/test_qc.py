import pytest

from app.audio import AudioError, PlannedChunk, Silence
from app.qc import ACOUSTIC_FLAG_REASONS, audit_local


def _row(id, start, end, ja="こんにちは", id_text="Halo"):
    return {"id": id, "start": start, "end": end, "type": "dialogue", "ja": ja, "id_text": id_text}


def test_audit_local_detects_rule_violations():
    rows = [
        _row(1, 0.0, 2.0, ja="あ" * 40, id_text="Halo saya mau makan"),         # rate_outlier (>18 cps JP) + pronoun_violation
        _row(2, 1.8, 3.0, ja="あいうえ", id_text="Ini masih テスト ya"),          # overlap with row 1 + untranslated_jp
        _row(3, 5.0, 6.0, ja="あ", id_text="- A\n- B\n- C"),                     # speaker_overflow (>=3 turns)
        _row(4, 10.0, 11.0, ja="あいう", id_text="x" * 30),                      # high_cps (>25)
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert "rate_outlier" in flags[1]
    assert "pronoun_violation" in flags[1]
    assert "overlap" in flags[2]
    assert "untranslated_jp" in flags[2]
    assert "speaker_overflow" in flags[3]
    assert "high_cps" in flags[4]


def test_audit_local_flags_timing_fallback_chunks_and_silence(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [Silence(10.0, 15.0)])
    chunks_offsets = [
        {"index": 1, "start": 0.0, "end": 8.0, "overlap_prev": False},
        {"index": 2, "start": 8.0, "end": 20.0, "overlap_prev": False},
    ]
    timing_report = [
        {"chunk": 1, "status": "corrected"},
        {"chunk": 2, "status": "fallback", "reason": "gemini_parse_error"},
    ]
    rows = [
        _row(1, 1.0, 2.0, ja="おはよう", id_text="Pagi"),
        _row(2, 11.0, 12.5, ja="どうした", id_text="Kenapa?"),  # in chunk 2 fallback AND inside silence 10..15
    ]
    flags = audit_local(
        rows, audio_path=audio_path, timing_report=timing_report,
        chunk_offsets=chunks_offsets, cps_threshold=25.0,
    )
    assert 1 not in flags
    assert "timing_fallback" in flags[2]
    assert "in_silence" in flags[2]


def test_audit_local_detects_sparse_rate_outlier():
    rows = [
        _row(1, 0.0, 5.0, ja="あ い う", id_text="Eh?"),  # 3 JP chars / 5.0s = 0.6 < 1.2
        _row(2, 6.0, 11.0, ja="あいうえおか", id_text="Lumayan"),  # 6 JP chars / 5.0s = 1.2 (boundary ok)
        _row(3, 12.0, 15.0, ja="あ", id_text="Ya"),  # dur == 3.0 (not > 3.0)
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert flags == {1: ["rate_outlier"]}


def test_audit_local_detects_ja_turns_speaker_overflow_and_ignores_two_turns():
    rows = [
        _row(1, 0.0, 2.0, ja="- ええ\n- そう\n- マジ", id_text="- Eh\n- Iya sih"),  # ja_turns == 3
        _row(2, 3.0, 5.0, ja="- はい\n- いいえ", id_text="- Iya\n- Nggak"),          # 2 turns ok
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert "speaker_overflow" in flags[1]
    assert 2 not in flags


def test_audit_local_overlap_tracks_max_prev_end_across_multi_cue_and_out_of_order():
    rows = [
        _row(1, 0.0, 8.0, ja="あ" * 16, id_text="Panjang sekali"),
        _row(2, 2.0, 3.0, ja="あいう", id_text="Di tengah"),         # overlaps row 1
        _row(3, 5.0, 6.0, ja="あいう", id_text="Masih di dalam"),    # still < row 1 end (8.0), even though > row 2 end (3.0)
        _row(4, 1.0, 2.0, ja="あいう", id_text="Out of order"),      # out-of-order < max prev_end (8.0)
        _row(5, 8.0, 9.0, ja="あいう", id_text="Setelahnya"),        # clean boundary
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert 1 not in flags
    assert "overlap" in flags[2]
    assert "overlap" in flags[3]
    assert "overlap" in flags[4]
    assert 5 not in flags


def test_audit_local_ignores_short_cues_inside_silence(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [Silence(10.0, 15.0)])
    rows = [
        _row(1, 11.0, 11.6, ja="えっ", id_text="Hah?"),      # dur 0.6s < 0.7s -> not flagged
        _row(2, 12.0, 12.75, ja="えっと", id_text="Anu..."),  # dur 0.75s >= 0.7s -> flagged
    ]
    flags = audit_local(rows, audio_path=audio_path, timing_report=None, cps_threshold=25.0)
    assert 1 not in flags
    assert "in_silence" in flags[2]


def test_audit_local_degrades_gracefully_when_detect_silences_fails(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")

    def boom(p, db):
        raise AudioError("ffmpeg gagal")

    monkeypatch.setattr("app.qc.audio.detect_silences", boom)
    rows = [
        _row(1, 1.0, 2.5, ja="おはよう", id_text="Halo saya mau makan"),
    ]
    flags = audit_local(rows, audio_path=audio_path, timing_report=None, cps_threshold=25.0)
    assert flags == {1: ["pronoun_violation"]}


def test_audit_local_pronoun_boundaries_and_acoustic_constants():
    rows = [
        _row(1, 0.0, 2.0, ja="おはよう", id_text="Halo sayang, canda lagu melodi"),
        _row(2, 3.0, 5.0, ja="おはよう", id_text="GUE dan LO serta Anda"),
        _row(3, 6.0, 8.0, ja="おはよう", id_text="Masih ada ひらがな"),
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert 1 not in flags
    assert "pronoun_violation" in flags[2]
    assert "untranslated_jp" in flags[3]
    assert ACOUSTIC_FLAG_REASONS == {
        "timing_fallback",
        "in_silence",
        "overlap",
        "rate_outlier",
        "speaker_overflow",
    }


def test_audit_local_timing_fallback_boundaries_and_empty_rows():
    assert audit_local([]) == {}

    chunks = [
        PlannedChunk(1, 10.0, 20.0, False),
        PlannedChunk(2, 20.0, 30.0, False),
    ]
    rows = [
        _row(1, 9.5, 10.5, ja="あいう", id_text="Pre-first"),   # < chunk 1 start -> chunk 1
        _row(2, 20.0, 21.0, ja="あいう", id_text="Boundary"),   # == chunk 1 end -> chunk 2
        _row(3, 31.0, 32.0, ja="あいう", id_text="After-last"), # >= chunk 2 end -> chunk 2
    ]

    flags_c1 = audit_local(
        rows,
        timing_report=[{"chunk": 1, "status": "fallback"}, {"chunk": 2, "status": "corrected"}],
        chunk_offsets=chunks,
    )
    assert flags_c1 == {1: ["timing_fallback"]}

    flags_c2 = audit_local(
        rows,
        timing_report=[{"chunk": 1, "status": "corrected"}, {"chunk": 2, "status": "fallback"}],
        chunk_offsets=chunks,
    )
    assert flags_c2 == {2: ["timing_fallback"], 3: ["timing_fallback"]}


def test_audit_local_sorts_working_copy_without_mutating_caller_rows():
    disjoint = [
        _row(1, 5.0, 6.0, ja="あいう", id_text="Kedua"),
        _row(2, 1.0, 2.0, ja="あいう", id_text="Pertama"),
    ]
    flags_disjoint = audit_local(disjoint)
    assert flags_disjoint == {}
    assert [r["id"] for r in disjoint] == [1, 2]

    reversed_overlap = [
        _row(1, 5.0, 6.0, ja="あいう", id_text="Kedua"),
        _row(2, 4.5, 5.5, ja="あいう", id_text="Pertama"),
    ]
    flags_overlap = audit_local(reversed_overlap)
    assert flags_overlap == {1: ["overlap"]}
    assert [r["id"] for r in reversed_overlap] == [1, 2]


def test_audit_local_handles_null_ja_and_id_text():
    rows = [
        _row(1, 0.0, 0.1, ja=None, id_text=None),   # short 0.1s: must NOT trigger high_cps
        _row(2, 1.0, 4.2, ja=None, id_text=None),   # long 3.2s: empty ja triggers rate_outlier
    ]
    flags = audit_local(rows, cps_threshold=25.0)
    assert 1 not in flags
    assert flags[2] == ["rate_outlier"]


def test_audit_local_empty_none_and_unsorted_chunk_offsets():
    rows = [
        _row(1, 1.0, 2.0, ja="あいう", id_text="Satu"),
        _row(2, 12.0, 13.0, ja="あいう", id_text="Dua"),
    ]
    timing_report = [
        {"chunk": 1, "status": "fallback"},
        {"chunk": 2, "status": "corrected"},
    ]
    assert audit_local(rows, timing_report=timing_report, chunk_offsets=[]) == {}
    assert audit_local(rows, timing_report=timing_report, chunk_offsets=None) == {}

    unsorted_chunks = [
        PlannedChunk(2, 10.0, 20.0, False),
        PlannedChunk(1, 0.0, 10.0, False),
    ]
    flags = audit_local(rows, timing_report=timing_report, chunk_offsets=unsorted_chunks)
    assert flags == {1: ["timing_fallback"]}


def test_audit_local_empty_rows_skips_detect_silences(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    called = False

    def spy_detect(p, db):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr("app.qc.audio.detect_silences", spy_detect)
    assert audit_local(
        [],
        audio_path=audio_path,
        timing_report=[{"chunk": 1, "status": "fallback"}],
        chunk_offsets=[{"index": 1, "start": 0.0, "end": 10.0}],
    ) == {}
    assert called is False


def test_audit_local_flags_zero_and_inverted_duration_as_rate_outlier():
    rows = [
        _row(1, 2.0, 2.0, ja="", id_text=""),          # end == start
        _row(2, 5.0, 4.0, ja="あ", id_text="Halo"),    # end < start
    ]
    flags = audit_local(rows, cps_threshold=25.0)
    assert "rate_outlier" in flags[1]
    assert "rate_outlier" in flags[2]
    assert "rate_outlier" in ACOUSTIC_FLAG_REASONS
    assert set(flags[1]) & ACOUSTIC_FLAG_REASONS == {"rate_outlier"}
    assert set(flags[2]) & ACOUSTIC_FLAG_REASONS == {"rate_outlier"}

