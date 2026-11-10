import json

import pytest

from app.audio import AudioError, PlannedChunk, Silence
from app.providers import LLMResponse
from app.qc import (
    ACOUSTIC_FLAG_REASONS,
    apply_text_qc_results,
    audit_local,
    run_text_qc,
)


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


def test_apply_text_qc_results_updates_translations_and_collects_audio_flags():
    rows = [
        _row(1, 0.0, 2.0, ja="人狼クイーン決定戦", id_text="Pertandingan Wolf Queen"),
        _row(2, 2.5, 4.5, ja="変な日本語", id_text="Kalimat aneh"),
    ]
    qc_items = [
        {
            "id": 1,
            "fixed_id_text": "Pertandingan Ratu Serigala",
            "needs_audio_check": False,
            "reason": "Konsistensi glosarium",
        },
        {
            "id": 2,
            "fixed_id_text": "",
            "needs_audio_check": True,
            "reason": "ja mencurigakan",
        },
    ]
    updated, issues, audio_ids = apply_text_qc_results(rows, qc_items)
    assert updated[0]["id_text"] == "Pertandingan Ratu Serigala"
    assert updated[1]["id_text"] == "Kalimat aneh"
    assert audio_ids == {2}
    assert issues[0]["type"] == "translation"
    assert issues[0]["pass"] == "text_qc"


def test_apply_text_qc_results_ignores_invalid_and_noop_fixes():
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Teks satu"),
        _row(2, 2.0, 4.0, ja="い", id_text="Teks dua"),
        _row(3, 4.0, 6.0, ja="う", id_text="Teks tiga"),
        _row(4, 6.0, 8.0, ja="え", id_text="Teks empat"),
        _row(5, 8.0, 10.0, ja="お", id_text="Teks lima"),
        _row(6, 10.0, 12.0, ja="か", id_text=None),
    ]
    qc_items = [
        "not-a-dict",
        {"id": True, "fixed_id_text": "Salah id bool", "needs_audio_check": True},
        {"id": 1.5, "fixed_id_text": "Salah id float"},
        {"id": "bad", "fixed_id_text": "Salah id str"},
        {"id": 1, "fixed_id_text": True, "needs_audio_check": "true"},
        {"id": 2, "fixed_id_text": 12345, "needs_audio_check": 1},
        {"id": 3, "fixed_id_text": ["list"], "needs_audio_check": False},
        {"id": 4, "fixed_id_text": "  Teks empat  ", "needs_audio_check": False},
        {"id": 5, "fixed_id_text": "   ", "needs_audio_check": False},
        {"id": 6, "fixed_id_text": None, "needs_audio_check": True, "reason": ""},
    ]
    updated, issues, audio_ids = apply_text_qc_results(rows, qc_items)
    assert [r["id_text"] for r in updated[:5]] == [
        "Teks satu",
        "Teks dua",
        "Teks tiga",
        "Teks empat",
        "Teks lima",
    ]
    assert updated[5]["id_text"] is None
    assert audio_ids == {6}
    assert issues == [{
        "id": 6,
        "pass": "text_qc",
        "type": "audio_flag",
        "reason": "ja/timing dicurigai perlu dicek audio",
    }]
    # Input rows must remain untouched
    assert rows[0]["id_text"] == "Teks satu"


def test_run_text_qc_single_pass_and_windowed():
    rows = [_row(i, float(i), float(i) + 0.9, ja="あ", id_text=f"Teks {i}") for i in range(1, 5)]
    calls = []

    def fake_text_runner(system, user):
        calls.append(user)
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Teks Konsisten", "needs_audio_check": False, "reason": "Glosarium"},
                {"id": 3, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio"},
            ]),
            input_tokens=20, output_tokens=10,
        ), "gemini-3.6-flash"

    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    fixed, issues, audio_ids = run_text_qc(
        rows, system="SYS", glossary={"A": "B"}, audit_flags={3: ["rate_outlier"]},
        tracker=T(), text_runner=fake_text_runner,
    )
    assert len(calls) == 1
    assert "A -> B" in calls[0]
    assert fixed[0]["id_text"] == "Teks Konsisten"
    assert 3 in audio_ids
    assert tracker_calls == [("gemini-3.6-flash", 20, 10)]


def _extract_prompt_items(user_prompt: str) -> list[dict]:
    marker = "=== TRANSKRIP & TERJEMAHAN ===\n"
    start = user_prompt.index(marker) + len(marker)
    end = user_prompt.index("\n\nTugasmu", start)
    return json.loads(user_prompt[start:end])


def test_run_text_qc_windowed_episode_overlap_context_and_isolation():
    rows = [
        _row(i, float(i), float(i) + 0.9, ja="あいう", id_text=f"Teks {i}")
        for i in range(1, 261)
    ]
    calls: list[str] = []
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def fake_text_runner(system, user):
        calls.append(user)
        if len(calls) == 1:
            return LLMResponse(
                text=json.dumps([
                    {"id": 180, "fixed_id_text": "Teks 180 diperbaiki", "needs_audio_check": False, "reason": "Glosarium"},
                    {"id": 185, "fixed_id_text": "", "needs_audio_check": True, "reason": "Overlap W1 cek"},
                    {"id": 250, "fixed_id_text": "Bocor dari W1", "needs_audio_check": True, "reason": "Di luar W1"},
                ]),
                input_tokens=100, output_tokens=40,
            ), "gemini-3.6-flash"
        return LLMResponse(
            text=json.dumps([
                {"id": 180, "fixed_id_text": "Seharusnya diabaikan", "needs_audio_check": True, "reason": "Overlap context_only"},
                {"id": 201, "fixed_id_text": "Teks 201 konsisten", "needs_audio_check": False, "reason": "Style guide"},
                {"id": 250, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio W2"},
            ]),
            input_tokens=80, output_tokens=30,
        ), "gpt-4.1"

    fixed, issues, audio_ids = run_text_qc(
        rows,
        system="SYS",
        glossary={"テスト": "Tes"},
        audit_flags={201: ["pronoun_violation"]},
        tracker=T(),
        text_runner=fake_text_runner,
    )

    assert len(calls) == 2
    w1_items = _extract_prompt_items(calls[0])
    w2_items = _extract_prompt_items(calls[1])

    assert [it["id"] for it in w1_items] == list(range(1, 201))
    assert all("context_only" not in it for it in w1_items)

    assert [it["id"] for it in w2_items] == list(range(171, 261))
    assert all(it.get("context_only") is True for it in w2_items[:30])
    assert all("context_only" not in it for it in w2_items[30:])

    # Window 1 fix on overlap row 180 is visible in Window 2 prompt
    item_180_w2 = next(it for it in w2_items if it["id"] == 180)
    assert item_180_w2["id_text"] == "Teks 180 diperbaiki"
    item_201_w2 = next(it for it in w2_items if it["id"] == 201)
    assert item_201_w2["audit_flags"] == ["pronoun_violation"]

    # Window 2 overlap edit on 180 was ignored; Window 1 fix on 180 preserved
    assert fixed[179]["id_text"] == "Teks 180 diperbaiki"
    assert 180 not in audio_ids
    # Out-of-window hallucination (250 from W1) was ignored; W2 target fixes applied
    assert fixed[200]["id_text"] == "Teks 201 konsisten"
    assert fixed[249]["id_text"] == "Teks 250"
    assert audio_ids == {185, 250}
    assert len(issues) == 4
    assert tracker_calls == [("gemini-3.6-flash", 100, 40), ("gpt-4.1", 80, 30)]
    assert rows[179]["id_text"] == "Teks 180"


def test_run_text_qc_checkpoint_save_resume_and_corrupt_recompute(tmp_path):
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Satu"),
        _row(2, 2.0, 4.0, ja="い", id_text="Dua"),
    ]
    ckpt = tmp_path / "qc" / "text_qc.json"
    calls = 0
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def runner(system, user):
        nonlocal calls
        calls += 1
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Satu Baru", "needs_audio_check": False, "reason": "Glosarium"},
                {"id": 2, "fixed_id_text": "", "needs_audio_check": True, "reason": "Audio aneh"},
            ]),
            input_tokens=15, output_tokens=5,
        ), "gemini-3.6-flash"

    fixed1, issues1, audio1 = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=T(), text_runner=runner,
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    assert calls == 1
    assert audio1 == {2}
    assert tracker_calls == [("gemini-3.6-flash", 15, 5)]

    # Resume from valid checkpoint: must not invoke runner or tracker
    resume_tracker_calls = []

    class FailTracker:
        def add(self, m, i, o):
            resume_tracker_calls.append((m, i, o))

    def fail_runner(system, user):
        raise AssertionError("text_runner must not be called when valid checkpoint exists")

    fixed2, issues2, audio2 = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=FailTracker(),
        text_runner=fail_runner, checkpoint_path=ckpt,
    )
    assert fixed2 == fixed1
    assert issues2 == issues1
    assert audio2 == audio1
    assert resume_tracker_calls == []

    # Corrupt checkpoints must trigger clean recomputation
    corrupt_payloads = [
        "{not-json",
        json.dumps(["not", "a", "dict"]),
        json.dumps({"rows": "bad", "issues": [], "audio_ids": []}),
        json.dumps({"rows": [rows[0]], "issues": [], "audio_ids": []}),
        json.dumps({"rows": ["bad", "bad"], "issues": [], "audio_ids": []}),
        json.dumps({"rows": rows, "issues": ["bad"], "audio_ids": []}),
        json.dumps({"rows": rows, "issues": [], "audio_ids": [True]}),
    ]
    for bad in corrupt_payloads:
        ckpt.write_text(bad, encoding="utf-8")
        before = calls
        fixed3, issues3, audio3 = run_text_qc(
            rows, system="SYS", glossary={}, audit_flags={}, tracker=T(),
            text_runner=runner, checkpoint_path=ckpt,
        )
        assert calls == before + 1
        assert fixed3[0]["id_text"] == "Satu Baru"
        assert audio3 == {2}


def test_run_text_qc_propagates_runner_error_without_checkpoint_or_charge(tmp_path):
    rows = [_row(1, 0.0, 2.0, ja="あ", id_text="Satu")]
    ckpt = tmp_path / "qc" / "text_qc.json"
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def boom(system, user):
        raise RuntimeError("503 all fallback slots failed")

    with pytest.raises(RuntimeError, match="503 all fallback slots failed"):
        run_text_qc(
            rows, system="SYS", glossary={}, audit_flags={}, tracker=T(),
            text_runner=boom, checkpoint_path=ckpt,
        )
    assert tracker_calls == []
    assert not ckpt.exists()

    # Multi-window: error on window 2 propagates and does NOT write checkpoint
    rows_260 = [_row(i, float(i), float(i) + 0.9, ja="あ", id_text=f"Teks {i}") for i in range(1, 261)]
    w_calls = 0

    def fail_on_w2(system, user):
        nonlocal w_calls
        w_calls += 1
        if w_calls == 2:
            raise RuntimeError("429 window 2 failed")
        return LLMResponse(text="[]", input_tokens=10, output_tokens=5), "gemini-3.6-flash"

    with pytest.raises(RuntimeError, match="429 window 2 failed"):
        run_text_qc(
            rows_260, system="SYS", glossary={}, audit_flags={}, tracker=T(),
            text_runner=fail_on_w2, checkpoint_path=ckpt,
        )
    assert tracker_calls == [("gemini-3.6-flash", 10, 5)]
    assert not ckpt.exists()