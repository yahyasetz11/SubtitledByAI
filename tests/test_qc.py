import json

import pytest

from app.audio import AudioError, PlannedChunk, Silence
from app.providers import LLMResponse
from app.qc import (
    ACOUSTIC_FLAG_REASONS,
    apply_audio_qc_results,
    apply_text_qc_results,
    audit_local,
    build_audio_qc_user_prompt,
    build_audio_qc_windows,
    build_text_qc_user_prompt,
    load_qc,
    parse_qc_response,
    run_qc_all,
    run_text_qc,
    save_qc,
    slice_audio_for_batch,
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


def test_run_text_qc_single_pass():
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


def test_parse_qc_response_and_prompt_formatting():
    with pytest.raises(ValueError, match="output kosong"):
        parse_qc_response(None)
    with pytest.raises(ValueError, match="output bukan array JSON"):
        parse_qc_response("{}")
    assert parse_qc_response('{"items": []}') == []
    assert parse_qc_response('{"items": [{"id": 1}]}') == [{"id": 1}]
    assert parse_qc_response('```json\n{"items": [{"id": 1}]}\n```') == [{"id": 1}]
    assert parse_qc_response('[{"id": 1}]') == [{"id": 1}]
    assert parse_qc_response('[]') == []
    assert parse_qc_response('{"id": 1, "fixed_id_text": "Baru"}') == [{"id": 1, "fixed_id_text": "Baru"}]
    assert parse_qc_response('{"notes": "ok", "items": [{"id": 2}]}') == [{"id": 2}]

    rows = [
        _row(1, 0.0, 1.5, ja="あ", id_text="Satu"),
        _row(2, 1.5, 3.0, ja="い", id_text="Dua"),
    ]
    prompt = build_text_qc_user_prompt(
        rows,
        glossary={"あ": "A"},
        audit_flags={1: [], 2: ["high_cps"]},
        check_ids={2},
    )
    assert '"items"' in prompt
    assert '{"items": []}' in prompt
    assert '"context_only": true' in prompt
    items = _extract_prompt_items(prompt)
    assert items[0]["context_only"] is True
    assert "audit_flags" not in items[0]
    assert "context_only" not in items[1]
    assert items[1]["audit_flags"] == ["high_cps"]


def test_run_text_qc_accepts_openai_json_object_responses():
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Teks satu"),
        _row(2, 2.0, 4.0, ja="い", id_text="Teks dua"),
    ]
    clean_calls = 0

    def openai_clean_runner(system, user):
        nonlocal clean_calls
        clean_calls += 1
        assert '"items"' in user
        return LLMResponse(text='{"items": []}', input_tokens=10, output_tokens=3), "gpt-4.1"

    fixed_clean, issues_clean, audio_clean = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
        text_runner=openai_clean_runner,
    )
    assert clean_calls == 1
    assert fixed_clean == rows
    assert issues_clean == []
    assert audio_clean == set()

    fix_calls = 0

    def openai_fix_runner(system, user):
        nonlocal fix_calls
        fix_calls += 1
        return LLMResponse(
            text=json.dumps({
                "items": [
                    {"id": 1, "fixed_id_text": "Teks satu diperbaiki", "needs_audio_check": False, "reason": "Glosarium"},
                ]
            }),
            input_tokens=12, output_tokens=8,
        ), "gpt-4.1"

    fixed_up, issues_up, audio_up = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
        text_runner=openai_fix_runner,
    )
    assert fix_calls == 1
    assert fixed_up[0]["id_text"] == "Teks satu diperbaiki"
    assert fixed_up[1]["id_text"] == "Teks dua"
    assert len(issues_up) == 1
    assert audio_up == set()


@pytest.mark.parametrize("bad_text", ["not-json", "{}", None])
def test_run_text_qc_retries_corrupt_json_and_recovers(bad_text):
    rows = [_row(1, 0.0, 2.0, ja="あ", id_text="Lama")]
    prompts: list[str] = []
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def flaky_runner(system, user):
        prompts.append(user)
        if len(prompts) == 1:
            return LLMResponse(text=bad_text, input_tokens=10, output_tokens=5), "gemini-3.6-flash"
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Baru", "needs_audio_check": False, "reason": "Glosarium"},
            ]),
            input_tokens=12, output_tokens=6,
        ), "gpt-4.1"

    fixed, issues, audio_ids = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=T(), text_runner=flaky_runner,
    )
    assert len(prompts) == 2
    assert prompts[1].startswith("Output sebelumnya SALAH (")
    assert 'Balas HANYA objek JSON {"items": [...]} (atau array JSON) sesuai skema.' in prompts[1]
    assert fixed[0]["id_text"] == "Baru"
    assert audio_ids == set()
    assert [i["type"] for i in issues] == ["translation"]
    assert tracker_calls == [("gemini-3.6-flash", 10, 5), ("gpt-4.1", 12, 6)]


def test_run_text_qc_second_parse_failure_warns_without_crashing_or_losing_prior_windows(tmp_path):
    rows = [_row(1, 0.0, 2.0, ja="あ", id_text="Tetap")]
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def always_corrupt(system, user):
        return LLMResponse(text="{}" if "SALAH" in user else "not-json", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    fixed, issues, audio_ids = run_text_qc(
        rows, system="SYS", glossary={}, audit_flags={}, tracker=T(), text_runner=always_corrupt,
    )
    assert fixed[0]["id_text"] == "Tetap"
    assert audio_ids == set()
    assert len(issues) == 1
    assert issues[0]["pass"] == "text_qc"
    assert issues[0]["type"] == "warning"
    assert "Respons Pass 1B tidak valid" in issues[0]["reason"]
    assert tracker_calls == [("gemini-3.6-flash", 5, 2), ("gemini-3.6-flash", 5, 2)]

    # Multi-window: W1 succeeds, W2 fails twice -> W1 updates preserved + W2 warning added
    rows_260 = [_row(i, float(i), float(i) + 0.9, ja="あ", id_text=f"Teks {i}") for i in range(1, 261)]
    ckpt = tmp_path / "qc" / "text_qc.json"
    w_calls = 0
    w_tracker = []

    class WT:
        def add(self, m, i, o):
            w_tracker.append((m, i, o))

    def w1_ok_w2_bad(system, user):
        nonlocal w_calls
        w_calls += 1
        if w_calls == 1:
            return LLMResponse(
                text=json.dumps([
                    {"id": 1, "fixed_id_text": "Teks 1 Baru", "needs_audio_check": False, "reason": "Glosarium"},
                    {"id": 2, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio"},
                ]),
                input_tokens=50, output_tokens=20,
            ), "gemini-3.6-flash"
        return LLMResponse(text=None if w_calls == 2 else "not-json", input_tokens=40, output_tokens=10), "gpt-4.1"

    fixed_w, issues_w, audio_w = run_text_qc(
        rows_260, system="SYS", glossary={}, audit_flags={}, tracker=WT(),
        text_runner=w1_ok_w2_bad, checkpoint_path=ckpt,
    )
    assert w_calls == 3
    assert fixed_w[0]["id_text"] == "Teks 1 Baru"
    assert fixed_w[200]["id_text"] == "Teks 201"
    assert audio_w == {2}
    assert [i["type"] for i in issues_w] == ["translation", "audio_flag", "warning"]
    assert "Respons Pass 1B tidak valid" in issues_w[2]["reason"]
    assert w_tracker == [
        ("gemini-3.6-flash", 50, 20),
        ("gpt-4.1", 40, 10),
        ("gpt-4.1", 40, 10),
    ]
    assert ckpt.exists()


def test_run_text_qc_checkpoint_edge_cases_and_empty_rows(tmp_path):
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Satu"),
        _row(2, 2.0, 4.0, ja="い", id_text="Dua"),
    ]
    ckpt = tmp_path / "qc" / "text_qc.json"
    calls = 0

    def runner(system, user):
        nonlocal calls
        calls += 1
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Satu Recomputed", "needs_audio_check": False, "reason": "Ok"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    # Empty rows must ignore stale checkpoint and skip runner
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps({
        "rows": [_row(1, 0.0, 1.0, id_text="Stale")],
        "issues": [{"pass": "text_qc", "type": "translation"}],
        "audio_ids": [1],
    }), encoding="utf-8")
    assert run_text_qc(
        [], system="SYS", glossary={}, audit_flags={}, tracker=None,
        text_runner=lambda s, u: (_ for _ in ()).throw(AssertionError("runner called on empty rows")),
        checkpoint_path=ckpt,
    ) == ([], [], set())

    row_missing_start = dict(rows[0])
    del row_missing_start["start"]
    row_bool_start = dict(rows[0], start=True)
    row_str_end = dict(rows[0], end="2.0")
    row_none_end = dict(rows[0], end=None)

    bad_checkpoints = [
        # Matching count (2) but mismatched IDs
        json.dumps({"rows": [_row(1, 0.0, 2.0), _row(3, 2.0, 4.0)], "issues": [], "audio_ids": []}),
        # Matching count & set, wrong sequence
        json.dumps({"rows": [rows[1], rows[0]], "issues": [], "audio_ids": []}),
        # Missing / invalid start / end
        json.dumps({"rows": [row_missing_start, rows[1]], "issues": [], "audio_ids": []}),
        json.dumps({"rows": [row_bool_start, rows[1]], "issues": [], "audio_ids": []}),
        json.dumps({"rows": [row_str_end, rows[1]], "issues": [], "audio_ids": []}),
        json.dumps({"rows": [row_none_end, rows[1]], "issues": [], "audio_ids": []}),
        '{"rows": [{"id": 1, "start": NaN, "end": 2.0}, {"id": 2, "start": 2.0, "end": 4.0}], "issues": [], "audio_ids": []}',
        # audio_ids outside expected row IDs
        json.dumps({"rows": rows, "issues": [], "audio_ids": [99]}),
    ]
    for payload in bad_checkpoints:
        ckpt.write_text(payload, encoding="utf-8")
        before = calls
        fixed, issues, audio_ids = run_text_qc(
            rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
            text_runner=runner, checkpoint_path=ckpt,
        )
        assert calls == before + 1
        assert fixed[0]["id_text"] == "Satu Recomputed"
        assert audio_ids == set()


def test_apply_text_qc_results_merges_duplicate_items():
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Lama 1"),
        _row(2, 2.0, 4.0, ja="い", id_text="Lama 2"),
    ]
    qc_items = [
        {"id": 1, "fixed_id_text": "Baru 1", "needs_audio_check": False, "reason": "Glosarium"},
        {"id": " 1 ", "fixed_id_text": "", "needs_audio_check": True, "reason": "Audio aneh"},
        {"id": 2, "fixed_id_text": "Draft 2", "needs_audio_check": True, "reason": "Cek audio"},
        {"id": 2, "fixed_id_text": "Final 2", "needs_audio_check": False, "reason": "Style guide"},
    ]
    updated, issues, audio_ids = apply_text_qc_results(rows, qc_items)
    assert updated[0]["id_text"] == "Baru 1"
    assert updated[1]["id_text"] == "Final 2"
    assert audio_ids == {1, 2}
    assert len(issues) == 4
    trans_1 = next(i for i in issues if i["id"] == 1 and i["type"] == "translation")
    audio_1 = next(i for i in issues if i["id"] == 1 and i["type"] == "audio_flag")
    assert trans_1["fixed"] == "Baru 1"
    assert "Glosarium" in trans_1["reason"] and "Audio aneh" in trans_1["reason"]
    assert "Audio aneh" in audio_1["reason"]


def test_run_text_qc_duplicate_row_ids_window_boundary_and_nonsequential_ids(tmp_path):
    dup_rows = [_row(1, 0.0, 1.0), _row(1, 1.0, 2.0)]
    with pytest.raises(ValueError, match="duplikat"):
        run_text_qc(
            dup_rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
            text_runner=lambda s, u: (_ for _ in ()).throw(AssertionError("should not run")),
        )

    # 250 rows -> 1 window; 251 rows -> 2 windows
    for count, expected_calls in [(250, 1), (251, 2)]:
        batch_rows = [_row(i, float(i), float(i) + 0.9, ja="あ", id_text=f"Teks {i}") for i in range(1, count + 1)]
        prompts: list[str] = []

        def count_runner(system, user):
            prompts.append(user)
            return LLMResponse(text="[]", input_tokens=1, output_tokens=1), "gemini-3.6-flash"

        run_text_qc(batch_rows, system="SYS", glossary={}, audit_flags={}, tracker=None, text_runner=count_runner)
        assert len(prompts) == expected_calls
        if count == 251:
            w2_items = _extract_prompt_items(prompts[1])
            assert [it["id"] for it in w2_items] == list(range(171, 252))
            assert all(it.get("context_only") is True for it in w2_items[:30])
            assert all("context_only" not in it for it in w2_items[30:])

    # Non-sequential unique IDs [10, 25, 42]
    sparse_rows = [
        _row(10, 0.0, 1.5, ja="あ", id_text="Sepuluh"),
        _row(25, 2.0, 3.5, ja="い", id_text="Dua Lima"),
        _row(42, 4.0, 5.5, ja="う", id_text="Empat Dua"),
    ]
    ckpt = tmp_path / "qc" / "sparse_ckpt.json"
    sparse_calls = 0

    def sparse_runner(system, user):
        nonlocal sparse_calls
        sparse_calls += 1
        return LLMResponse(
            text=json.dumps([
                {"id": 10, "fixed_id_text": "Sepuluh Baru", "needs_audio_check": False, "reason": "Fix"},
                {"id": 42, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    fixed_s, issues_s, audio_s = run_text_qc(
        sparse_rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
        text_runner=sparse_runner, checkpoint_path=ckpt,
    )
    assert sparse_calls == 1
    assert [r["id_text"] for r in fixed_s] == ["Sepuluh Baru", "Dua Lima", "Empat Dua"]
    assert audio_s == {42}
    assert len(issues_s) == 2

    # Resume from sparse checkpoint
    resumed = run_text_qc(
        sparse_rows, system="SYS", glossary={}, audit_flags={}, tracker=None,
        text_runner=lambda s, u: (_ for _ in ()).throw(AssertionError("should resume")),
        checkpoint_path=ckpt,
    )
    assert resumed == (fixed_s, issues_s, audio_s)


def test_build_audio_qc_windows_groups_nearby_flagged_rows_and_skips_clean():
    rows = [
        _row(1, 1.0, 2.0),
        _row(2, 3.0, 4.0),    # flagged
        _row(3, 5.0, 6.0),    # close to row 2 (gap 1s) -> same slice
        _row(4, 120.0, 122.0),
        _row(5, 130.0, 132.0), # flagged, far from 2/3 -> separate slice
    ]
    assert build_audio_qc_windows(rows, flagged_ids=set(), deep_audio_qc=False) == []
    windows = build_audio_qc_windows(rows, flagged_ids={2, 3, 5}, deep_audio_qc=False)
    assert len(windows) == 2
    assert [r["id"] for r in windows[0]["batch"]] == [2, 3]
    assert [r["id"] for r in windows[1]["batch"]] == [5]


def test_build_audio_qc_windows_span_limit_deep_clamping_and_context():
    assert build_audio_qc_windows([], flagged_ids={1}, deep_audio_qc=True) == []
    assert build_audio_qc_windows([], flagged_ids={1}, deep_audio_qc=False) == []

    # Gap <= 15s between consecutive flagged rows, but total span > 90s -> splits slice
    rows = [
        _row(1, 0.0, 5.0),
        _row(2, 18.0, 20.0),   # gap 13s, span 20s <= 90 -> same slice
        _row(3, 33.0, 45.0),   # gap 13s, span 45s <= 90 -> same slice
        _row(4, 58.0, 75.0),   # gap 13s, span 75s <= 90 -> same slice
        _row(5, 88.0, 95.0),   # gap 13s, span 95s > 90 -> new slice!
    ]
    windows = build_audio_qc_windows(rows, flagged_ids={1, 2, 3, 4, 5}, deep_audio_qc=False)
    assert len(windows) == 2
    assert [r["id"] for r in windows[0]["batch"]] == [1, 2, 3, 4]
    assert [r["id"] for r in windows[1]["batch"]] == [5]
    # Hybrid mode uses +/-5 context window around batch
    assert [r["id"] for r in windows[1]["context_window"]] == [1, 2, 3, 4, 5]

    # Deep Audio QC clamps batch_size <= 0 to 1 and window_size < 0 to 0
    many_rows = [_row(i, float(i), float(i) + 0.5) for i in range(1, 5)]
    deep_clamped = build_audio_qc_windows(
        many_rows, flagged_ids=set(), deep_audio_qc=True, batch_size=0, window_size=-3,
    )
    assert len(deep_clamped) == 4
    assert [r["id"] for r in deep_clamped[1]["batch"]] == [2]
    assert [r["id"] for r in deep_clamped[1]["context_window"]] == [2]

    deep_normal = build_audio_qc_windows(
        many_rows, flagged_ids=set(), deep_audio_qc=True, batch_size=2, window_size=1,
    )
    assert len(deep_normal) == 2
    assert [r["id"] for r in deep_normal[0]["batch"]] == [1, 2]
    assert [r["id"] for r in deep_normal[0]["context_window"]] == [1, 2, 3]
    assert [r["id"] for r in deep_normal[1]["batch"]] == [3, 4]
    assert [r["id"] for r in deep_normal[1]["context_window"]] == [2, 3, 4]


def test_build_audio_qc_user_prompt_includes_glossary_context_and_relative_times():
    batch = [_row(2, 10.5, 12.25, ja=None, id_text="Halo")]
    ctx = [
        _row(1, 8.0, 9.5, ja="前", id_text="Sebelumnya"),
        batch[0],
        _row(3, 13.0, 14.0, ja="次", id_text="Berikutnya"),
    ]
    prompt = build_audio_qc_user_prompt(
        batch, ctx, slice_start=9.0, glossary={"人狼": "Manusia Serigala"},
    )
    assert "人狼 -> Manusia Serigala" in prompt
    assert "[1] 8.00-9.50s | 前 | Sebelumnya" in prompt
    assert "detik 9.00 video asli" in prompt
    assert '"start": 1.5' in prompt
    assert '"end": 3.25' in prompt


def test_apply_audio_qc_results_applies_ja_timing_and_split():
    batch = [
        _row(1, 10.0, 12.0, ja="間違えた", id_text="Salah"),
        _row(2, 13.0, 17.0, ja="A B C D", id_text="Empat orang"),
    ]
    qc_items = [
        {
            "id": 1,
            "fixed_ja": "正しい日本語",
            "fixed_id_text": "Jepang yang benar",
            "fixed_start": 2.0,   # relative to slice_start=8.5 -> 10.5
            "fixed_end": 4.0,     # -> 12.5
            "reason": "Perbaiki ja dan geser timing",
        },
        {
            "id": 2,
            "split": [
                {"start": 4.5, "end": 6.0, "ja": "- A\n- B", "id_text": "- A\n- B"},
                {"start": 6.2, "end": 8.2, "ja": "- C\n- D", "id_text": "- C\n- D"},
            ],
            "reason": "Pecah 4 pembicara",
        },
    ]
    fixed, issues = apply_audio_qc_results(batch, qc_items, slice_start=8.5, slice_duration=10.0)
    assert len(fixed) == 3
    assert fixed[0]["ja"] == "正しい日本語"
    assert fixed[0]["id_text"] == "Jepang yang benar"
    assert fixed[0]["start"] == pytest.approx(10.5)
    assert fixed[0]["end"] == pytest.approx(12.5)
    assert fixed[1]["ja"] == "- A\n- B"
    assert fixed[1]["start"] == pytest.approx(13.0)
    assert fixed[2]["start"] == pytest.approx(14.7)
    types = {i["type"] for i in issues}
    assert {"ja", "translation", "timing", "split"} <= types


def test_apply_audio_qc_results_rejects_invalid_timestamps_splits_and_types():
    nar_row = {"id": 1, "start": 10.0, "end": 12.0, "type": "narration", "ja": "元の文", "id_text": "Awal"}
    bad_type_row = {"id": 2, "start": 12.5, "end": 14.5, "type": "WEIRD", "ja": "二人", "id_text": "Dua"}
    rows = [nar_row, bad_type_row, _row(3, 15.0, 17.0, ja="三人目", id_text="Tiga")]

    qc_items = [
        # Row 1: out-of-range timing + invalid split (only 1 valid part survives) + non-str fixed_ja
        {
            "id": 1,
            "fixed_ja": True,
            "fixed_id_text": 999,
            "fixed_start": 1.0,
            "fixed_end": 25.0,   # > slice_duration (10.0) + 0.5
            "split": [
                {"start": 1.0, "end": 2.0, "ja": "A", "id_text": "A"},
                {"start": -1.0, "end": 3.0, "ja": "B", "id_text": "B"},  # < 0.0 invalid
            ],
            "reason": "Test invalid",
        },
        # Row 2: valid split with weird source type -> normalized to "dialogue"; 5th item ignored
        {
            "id": 2,
            "split": [
                {"start": 4.0, "end": 4.5, "ja": "一", "id_text": "Satu"},
                {"start": 4.6, "end": 5.0, "ja": "二", "id_text": "Dua"},
                {"start": 5.1, "end": 5.5, "ja": "三", "id_text": "Tiga"},
                {"start": 5.6, "end": 6.0, "ja": "四", "id_text": "Empat"},
                {"start": 6.1, "end": 6.5, "ja": "五", "id_text": "Lima"},
            ],
            "reason": "Pecah 4",
        },
        # Row 3: bool timestamp + empty split list
        {
            "id": 3,
            "fixed_start": True,
            "fixed_end": 8.0,
            "split": [],
            "reason": "Bool timing",
        },
        # Invalid entries that must be ignored safely
        "bad-entry",
        {"id": True, "fixed_ja": "X"},
    ]
    fixed, issues = apply_audio_qc_results(rows, qc_items, slice_start=8.5, slice_duration=10.0)
    # Row 1 kept untouched (1) + Row 2 split into 4 (4) + Row 3 kept untouched (1) = 6
    assert len(fixed) == 6
    assert fixed[0]["ja"] == "元の文"
    assert fixed[0]["id_text"] == "Awal"
    assert fixed[0]["start"] == 10.0
    assert fixed[0]["end"] == 12.0
    assert all(r["type"] == "dialogue" for r in fixed[1:5])
    assert fixed[5]["start"] == 15.0 and fixed[5]["end"] == 17.0
    warnings = [i for i in issues if i["type"] == "warning"]
    # Row 1 invalid split + Row 1 out-of-range timing + Row 3 empty split + Row 3 bool timing
    assert len(warnings) >= 4
    # Caller input untouched
    assert rows[0]["ja"] == "元の文"


def test_run_qc_all_text_only_skips_audio(tmp_path):
    rows = [_row(1, 1.0, 3.0, ja="おはよう", id_text="Selamat pagi")]
    text_calls = []
    audio_calls = []

    def fake_text(system, user):
        text_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    def fake_audio(path, system, user):
        audio_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    class T:
        def add(self, *a):
            pass

    fixed, report = run_qc_all(
        None, rows, audio_path=None, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=T(),
        text_runner=fake_text, audio_runner=fake_audio,
        deep_audio_qc=False,
    )
    assert len(text_calls) == 1
    assert len(audio_calls) == 0
    assert report["mode"] == "text_only"
    assert report["summary"]["audio_slices_checked"] == 0
    assert fixed[0]["id"] == 1


def test_run_qc_all_hybrid_with_audio_skips_slices_when_zero_flags(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    monkeypatch.setattr(
        "app.qc.audio.cut_chunk",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cut_chunk must not be called")),
    )

    rows = [_row(1, 1.0, 3.0, ja="おはよう", id_text="Selamat pagi")]
    text_calls = []

    def fake_text(system, user):
        text_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    def fail_audio(*a, **kw):
        raise AssertionError("audio_runner must not be called when zero flags")

    tmp_dir = tmp_path / "qc_tmp"
    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fail_audio,
        deep_audio_qc=False,
    )
    assert len(text_calls) == 1
    assert report["mode"] == "hybrid"
    assert report["summary"]["audio_slices_checked"] == 0
    assert not tmp_dir.exists()
    assert fixed[0]["id"] == 1


def test_run_qc_all_translation_only_runs_pass1a_and_pass1b_and_counts_unresolved_flags(tmp_path):
    rows = [
        _row(1, 0.0, 2.0, ja="あ", id_text="Halo saya mau makan"),     # pronoun_violation (non-acoustic)
        _row(2, 2.5, 4.5, ja="変な日本語", id_text="Aneh nih"),
    ]
    audio_calls = []

    def fake_text(system, user):
        assert "pronoun_violation" in user
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Halo aku mau makan", "needs_audio_check": False, "reason": "Pronoun"},
                {"id": 2, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio nanti"},
            ]),
            input_tokens=15, output_tokens=8,
        ), "gpt-4.1"

    def fake_audio(path, system, user):
        audio_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    fixed, report = run_qc_all(
        None, rows, audio_path=None, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fake_audio,
        deep_audio_qc=True,  # still skipped because audio_path is None
    )
    assert len(audio_calls) == 0
    assert report["mode"] == "text_only"
    assert report["summary"]["audio_slices_checked"] == 0
    assert report["summary"]["translation_fixes"] == 1
    assert report["summary"]["unresolved_warnings"] == 1  # audio_flag on id=2 was NOT checked
    assert fixed[0]["id_text"] == "Halo aku mau makan"


def test_run_qc_all_deep_audio_qc_processes_all_batches_and_reindexes_splits(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    cut_calls = []

    def fake_cut(src, dst, start, end):
        cut_calls.append((start, end))
        dst.write_bytes(b"slice")

    monkeypatch.setattr("app.qc.audio.cut_chunk", fake_cut)

    rows = [
        _row(10, 5.0, 7.0, ja="元の文", id_text="Lama 1"),
        _row(20, 8.0, 12.0, ja="A B C", id_text="Tiga orang"),
        _row(30, 15.0, 17.0, ja="大丈夫", id_text="Aman"),
    ]
    audio_calls = []
    progress_calls = []
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def fake_text(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 20, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    def fake_audio(slice_p, system, user):
        assert slice_p.exists()
        audio_calls.append(user)
        if len(audio_calls) == 1:
            # slice_start = 5.0 - 1.5 = 3.5
            return LLMResponse(
                text=json.dumps([
                    {
                        "id": 10,
                        "fixed_ja": "正しい",
                        "fixed_id_text": "Benar",
                        "fixed_start": 1.0,  # -> 4.5
                        "fixed_end": 3.0,    # -> 6.5
                        "reason": "Perbaikan audio",
                    },
                    {
                        "id": 20,
                        "split": [
                            {"start": 4.5, "end": 6.0, "ja": "- A\n- B", "id_text": "- A\n- B"},
                            {"start": 6.2, "end": 8.0, "ja": "C", "id_text": "C"},
                        ],
                        "reason": "Split speaker",
                    },
                ]),
                input_tokens=30, output_tokens=15,
            ), "gemini-3.5-flash"
        return LLMResponse(text="[]", input_tokens=20, output_tokens=5), "gemini-3.5-flash"

    tmp_dir = tmp_path / "qc_tmp"
    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=T(),
        batch_size=2, window_size=1,
        on_batch=lambda d, tot: progress_calls.append((d, tot)),
        deep_audio_qc=True,
        text_runner=fake_text, audio_runner=fake_audio,
    )
    assert len(cut_calls) == 2
    assert len(audio_calls) == 2
    assert progress_calls == [(1, 2), (2, 2)]
    assert [r["id"] for r in fixed] == [1, 2, 3, 4]
    assert fixed[0]["ja"] == "正しい"
    assert fixed[0]["start"] == pytest.approx(4.5)
    assert fixed[1]["ja"] == "- A\n- B"
    assert fixed[2]["ja"] == "C"
    assert fixed[3]["id_text"] == "Aman"
    assert report["mode"] == "deep"
    assert report["deep_audio_qc"] is True
    assert report["summary"]["total_rows"] == 4
    assert report["summary"]["audio_slices_checked"] == 2
    assert report["summary"]["ja_fixes"] == 1
    assert report["summary"]["translation_fixes"] == 1
    assert report["summary"]["timing_fixes"] == 1
    assert report["summary"]["splits"] == 1
    # Row 20's Pass 1B audio_flag WAS checked in Pass 2 -> 0 unresolved!
    assert report["summary"]["unresolved_warnings"] == 0
    # Temp slices cleaned up
    assert list(tmp_dir.glob("qc_slice_*.mp3")) == []
    assert len(tracker_calls) == 3


def test_run_qc_all_pass2_checkpoints_json_retry_fallback_and_api_error(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    monkeypatch.setattr("app.qc.audio.cut_chunk", lambda src, dst, s, e: dst.write_bytes(b"slice"))

    rows = [
        _row(1, 2.0, 4.0, ja="元の", id_text="Satu"),
        _row(2, 60.0, 62.0, ja="二番", id_text="Dua"),
    ]
    ckpt_dir = tmp_path / "chunks"
    tmp_dir = tmp_path / "qc_tmp"
    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    def fake_text(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 1"},
                {"id": 2, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 2"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    # Case A: Slice 0 retries bad JSON and recovers; Slice 1 fails JSON twice -> warning fallback
    slice_calls = {0: 0, 1: 0}

    def flaky_audio(slice_p, system, user):
        idx = 0 if "qc_slice_000" in slice_p.name else 1
        slice_calls[idx] += 1
        if idx == 0:
            if slice_calls[0] == 1:
                return LLMResponse(text="not-json", input_tokens=10, output_tokens=2), "gemini-3.5-flash"
            assert user.startswith("Output sebelumnya SALAH (")
            return LLMResponse(
                text=json.dumps([{"id": 1, "fixed_ja": "修正済", "reason": "Recovered"}]),
                input_tokens=12, output_tokens=4,
            ), "gemini-3.5-flash"
        return LLMResponse(text="{}" if "SALAH" in user else "bad", input_tokens=8, output_tokens=2), "gemini-3.5-flash"

    fixed1, report1 = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=T(),
        text_runner=fake_text, audio_runner=flaky_audio,
        checkpoint_dir=ckpt_dir,
    )
    assert slice_calls == {0: 2, 1: 2}
    assert fixed1[0]["ja"] == "修正済"
    assert fixed1[1]["ja"] == "二番"
    assert report1["summary"]["ja_fixes"] == 1
    assert report1["summary"]["unresolved_warnings"] == 1  # warning from slice 1 parse failure
    assert (ckpt_dir / "qc_pass1.json").exists()
    assert (ckpt_dir / "qc_audio_000.json").exists()
    assert (ckpt_dir / "qc_audio_001.json").exists()
    assert list(tmp_dir.glob("qc_slice_*.mp3")) == []

    # Case B: Resume skips both Pass 1B and Pass 2 checkpoints without calling runners
    def fail_runner(*a, **kw):
        raise AssertionError("runners must not be called when checkpoints are valid")

    fixed2, report2 = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=T(),
        text_runner=fail_runner, audio_runner=fail_runner,
        checkpoint_dir=ckpt_dir,
    )
    assert fixed2 == fixed1
    assert report2 == report1

    # Case C: Corrupt slice 1 checkpoint triggers recompute only for slice 1;
    # API error on slice 1 propagates without writing checkpoint or leaving temp slice
    corrupt_payloads = [
        "{bad-json",
        json.dumps(["not", "dict"]),
        json.dumps({"batch_ids": [True], "fixed": [rows[1]], "issues": []}),
        json.dumps({"batch_ids": [99], "fixed": [rows[1]], "issues": []}),
        json.dumps({"batch_ids": [2], "fixed": [], "issues": []}),
        json.dumps({"batch_ids": [2], "fixed": [{**rows[1], "start": True}], "issues": []}),
        json.dumps({"batch_ids": [2], "fixed": [rows[1]], "issues": "bad"}),
    ]
    for bad in corrupt_payloads:
        (ckpt_dir / "qc_audio_001.json").write_text(bad, encoding="utf-8")

        def api_boom(slice_p, system, user):
            assert slice_p.exists()
            raise RuntimeError("503 audio backend busy")

        with pytest.raises(RuntimeError, match="503 audio backend busy"):
            run_qc_all(
                None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
                system="SYS", tracker=T(),
                text_runner=fail_runner, audio_runner=api_boom,
                checkpoint_dir=ckpt_dir,
            )
        assert list(tmp_dir.glob("qc_slice_*.mp3")) == []
        assert (ckpt_dir / "qc_audio_001.json").read_text(encoding="utf-8") == bad


def test_save_and_load_qc_structured_report(tmp_path):
    rows = [_row(1, 0.0, 1.0)]
    report = {"mode": "hybrid", "deep_audio_qc": False, "summary": {"total_rows": 1}, "issues": []}
    save_qc(rows, report, tmp_path)
    assert load_qc(tmp_path) == rows
    disk_report = json.loads((tmp_path / "qc" / "report.json").read_text(encoding="utf-8"))
    assert disk_report["mode"] == "hybrid"

    # Legacy list[dict] issues compatibility
    legacy_dir = tmp_path / "legacy"
    save_qc(rows, [{"id": 1, "type": "translation", "original": "A", "fixed": "B"}], legacy_dir)
    assert load_qc(legacy_dir) == rows
    legacy_report = json.loads((legacy_dir / "qc" / "report.json").read_text(encoding="utf-8"))
    assert legacy_report["summary"]["translation_fixes"] == 1
    assert legacy_report["issues"][0]["fixed"] == "B"


def test_run_qc_all_hybrid_routes_only_acoustic_flags_and_ignores_context_hallucinations(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [Silence(10.0, 12.8)])
    monkeypatch.setattr("app.qc.audio.cut_chunk", lambda src, dst, s, e: dst.write_bytes(b"slice"))

    rows = [
        _row(1, 1.0, 2.5, ja="おはよう", id_text="Halo saya makan"),      # non-acoustic only: pronoun_violation
        _row(2, 11.0, 12.5, ja="どうした", id_text="Kenapa?"),            # acoustic: in_silence
        _row(3, 13.0, 14.5, ja="大丈夫", id_text="Aman saja"),             # clean (in context_window for row 2!)
    ]
    audio_prompts = []

    def fake_text(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Halo aku makan", "needs_audio_check": False, "reason": "Pronoun"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    def fake_audio(slice_p, system, user):
        audio_prompts.append(user)
        # Hallucinates an edit for context-only id=3 and id=1, plus real fix for id=2
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_ja": "HALLUCINATED-1"},
                {"id": 2, "fixed_ja": "どうしたの？", "fixed_id_text": "Ada apa?", "reason": "Lebih jelas"},
                {"id": 3, "fixed_id_text": "HALLUCINATED-3"},
            ]),
            input_tokens=20, output_tokens=8,
        ), "gemini-3.5-flash"

    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fake_audio,
        deep_audio_qc=False,
    )
    assert len(audio_prompts) == 1
    assert report["mode"] == "hybrid"
    assert report["summary"]["local_flags"] == 2
    assert report["summary"]["audio_slices_checked"] == 1
    assert fixed[0]["ja"] == "おはよう"
    assert fixed[0]["id_text"] == "Halo aku makan"
    assert fixed[1]["ja"] == "どうしたの？"
    assert fixed[1]["id_text"] == "Ada apa?"
    assert fixed[2]["id_text"] == "Aman saja"


def test_run_qc_all_partial_resume_empty_rows_and_default_gemini_client(tmp_path, monkeypatch):
    # Empty rows -> zero calls, structured empty report
    assert run_qc_all(
        None, [], audio_path=None, tmp_dir=tmp_path / "empty",
        system="SYS", tracker=None,
    ) == ([], {
        "mode": "text_only",
        "deep_audio_qc": False,
        "summary": {
            "total_rows": 0,
            "local_flags": 0,
            "audio_slices_checked": 0,
            "translation_fixes": 0,
            "ja_fixes": 0,
            "timing_fixes": 0,
            "splits": 0,
            "unresolved_warnings": 0,
        },
        "issues": [],
    })

    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    monkeypatch.setattr("app.qc.audio.cut_chunk", lambda src, dst, s, e: dst.write_bytes(b"slice"))

    # Unsorted input + equal-start tie-breaking + partial resume
    rows = [
        _row(20, 60.0, 62.0, ja="二", id_text="Dua"),
        _row(10, 5.0, 7.0, ja="一", id_text="Satu"),
    ]
    ckpt_dir = tmp_path / "chunks"
    tmp_dir = tmp_path / "qc_tmp"
    calls = []

    def text_ok(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 10, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 10"},
                {"id": 20, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 20"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    def fail_second_slice(slice_p, system, user):
        calls.append(slice_p.name)
        if len(calls) == 1:
            return LLMResponse(
                text=json.dumps([{"id": 10, "fixed_ja": "一修正", "reason": "Ok"}]),
                input_tokens=10, output_tokens=5,
            ), "gemini-3.5-flash"
        raise RuntimeError("503 slice 2")

    with pytest.raises(RuntimeError, match="503 slice 2"):
        run_qc_all(
            None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
            system="SYS", tracker=None,
            text_runner=text_ok, audio_runner=fail_second_slice,
            checkpoint_dir=ckpt_dir,
        )
    assert (ckpt_dir / "qc_pass1.json").exists()
    assert (ckpt_dir / "qc_audio_000.json").exists()
    assert not (ckpt_dir / "qc_audio_001.json").exists()

    # Retry resumes slice 0 and only runs slice 1
    retry_calls = []

    def finish_second_slice(slice_p, system, user):
        retry_calls.append(slice_p.name)
        return LLMResponse(
            text=json.dumps([{"id": 20, "fixed_ja": "二修正", "reason": "Ok 2"}]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=lambda s, u: (_ for _ in ()).throw(AssertionError("Pass 1B must resume")),
        audio_runner=finish_second_slice,
        checkpoint_dir=ckpt_dir,
    )
    assert retry_calls == ["qc_slice_001.mp3"]
    assert [r["id"] for r in fixed] == [1, 2]
    assert [r["ja"] for r in fixed] == ["一修正", "二修正"]
    assert report["summary"]["ja_fixes"] == 2
    assert report["summary"]["unresolved_warnings"] == 0

    # Default gemini client path (no custom runners)
    class FakeGemini:
        model = "gemini-3.5-flash"

        def __init__(self):
            self.uploads = 0
            self.gen_calls = 0

        def upload_audio(self, path):
            self.uploads += 1
            return f"ref:{path.name}"

        def generate(self, system, user, audio=None):
            self.gen_calls += 1
            if audio is None:
                return LLMResponse(
                    text=json.dumps([{"id": 1, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek"}]),
                    input_tokens=5, output_tokens=2,
                )
            return LLMResponse(
                text=json.dumps([{"id": 1, "fixed_ja": "直った", "reason": "Audio fix"}]),
                input_tokens=15, output_tokens=6,
            )

    gem = FakeGemini()
    fixed_gem, report_gem = run_qc_all(
        gem, [_row(1, 1.0, 2.0, ja="前", id_text="Awal")],
        audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
    )
    assert gem.uploads == 1
    assert gem.gen_calls == 2
    assert fixed_gem[0]["ja"] == "直った"
    assert report_gem["summary"]["ja_fixes"] == 1


def test_inverted_timestamp_windows_slice_and_pass2_repair(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    cut_calls = []

    def fake_cut(src, dst, start, end):
        cut_calls.append((start, end))
        dst.write_bytes(b"slice")

    monkeypatch.setattr("app.qc.audio.cut_chunk", fake_cut)

    # Direct slice_audio_for_batch on inverted row 10.0..5.0 -> slice_start=3.5, slice_end=11.5
    inv_row = _row(1, 10.0, 5.0, ja="あいう", id_text="Terbalik")
    out_p, s_start = slice_audio_for_batch(audio_path, [inv_row], tmp_path / "direct.mp3")
    assert out_p == tmp_path / "direct.mp3"
    assert s_start == pytest.approx(3.5)
    assert cut_calls[-1] == pytest.approx((3.5, 11.5))

    # build_audio_qc_windows keeps grouping intact across inverted timestamps
    win_rows = [
        _row(1, 0.0, 5.0),
        _row(2, 18.0, 2.0),   # inverted: max=18.0 -> group_end becomes 18.0
        _row(3, 30.0, 32.0),  # gap = 30.0 - 18.0 = 12.0 <= 15 -> same slice!
        _row(4, 92.0, 10.0),  # inverted: max=92.0 -> span 92.0 > 90 -> new slice!
    ]
    windows = build_audio_qc_windows(win_rows, flagged_ids={1, 2, 3, 4}, deep_audio_qc=False)
    assert len(windows) == 2
    assert [r["id"] for r in windows[0]["batch"]] == [1, 2, 3]
    assert [r["id"] for r in windows[1]["batch"]] == [4]

    # Full run_qc_all: Pass 1A flags inverted row 10.0..5.0, Pass 2 slices 3.5..11.5 and repairs to 5.0..10.0
    cut_calls.clear()
    rows = [
        _row(1, 10.0, 5.0, ja="あいうえお", id_text="Terbalik"),
        _row(2, 7.0, 8.0, ja="かきく", id_text="Tengah"),
    ]

    def fake_text(system, user):
        assert "rate_outlier" in user
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    def fake_audio(slice_p, system, user):
        assert "detik 3.50 video asli" in user
        # slice_start=3.5 -> relative 1.5..6.5 maps to 5.0..10.0 (now spans across row 2)
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_start": 1.5, "fixed_end": 6.5, "reason": "Perbaiki terbalik"},
            ]),
            input_tokens=15, output_tokens=5,
        ), "gemini-3.5-flash"

    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fake_audio,
    )
    assert cut_calls == [pytest.approx((3.5, 11.5))]
    assert report["summary"]["timing_fixes"] == 1
    assert fixed[0]["start"] == pytest.approx(5.0)
    assert fixed[0]["end"] == pytest.approx(10.0)
    assert [r["id"] for r in fixed] == [1, 2]


def test_non_adjacent_flagged_rows_share_slice_and_ignore_clean_row_edits(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    cut_calls = []

    def fake_cut(src, dst, start, end):
        cut_calls.append((start, end))
        dst.write_bytes(b"slice")

    monkeypatch.setattr("app.qc.audio.cut_chunk", fake_cut)

    rows = [
        _row(1, 1.0, 2.0, ja="一", id_text="Satu"),
        _row(2, 4.0, 5.0, ja="二", id_text="Dua"),     # flagged
        _row(3, 6.0, 7.0, ja="三", id_text="Tiga"),    # clean middle row
        _row(4, 8.0, 9.0, ja="四", id_text="Empat"),   # flagged
        _row(5, 11.0, 12.0, ja="五", id_text="Lima"),
    ]

    windows = build_audio_qc_windows(rows, flagged_ids={2, 4}, deep_audio_qc=False)
    assert len(windows) == 1
    assert [r["id"] for r in windows[0]["batch"]] == [2, 4]
    assert [r["id"] for r in windows[0]["context_window"]] == [1, 2, 3, 4, 5]

    audio_prompts = []

    def fake_text(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 2, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 2"},
                {"id": 4, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek 4"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    def fake_audio(slice_p, system, user):
        audio_prompts.append(user)
        return LLMResponse(
            text=json.dumps([
                {"id": 2, "fixed_ja": "二修正", "reason": "Fix 2"},
                {"id": 3, "fixed_ja": "HALLUCINATED-3", "fixed_id_text": "HALLUCINATED-3"},
                {"id": 4, "fixed_ja": "四修正", "reason": "Fix 4"},
            ]),
            input_tokens=20, output_tokens=8,
        ), "gemini-3.5-flash"

    fixed, report = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fake_audio,
    )
    assert len(cut_calls) == 1
    assert len(audio_prompts) == 1
    # Row 3 appears in context section, NOT in batch payload
    ctx_part, batch_part = audio_prompts[0].split("=== ITEM UNTUK DIAUDIT DENGAN AUDIO ===")
    assert "[3] 6.00-7.00s | 三 | Tiga" in ctx_part
    assert '"id": 3' not in batch_part
    assert fixed[1]["ja"] == "二修正"
    assert fixed[2]["ja"] == "三"
    assert fixed[2]["id_text"] == "Tiga"
    assert fixed[3]["ja"] == "四修正"
    assert report["summary"]["audio_slices_checked"] == 1
    assert report["summary"]["ja_fixes"] == 2


def test_apply_audio_qc_results_duplicate_merge_partial_timing_and_nan_inf():
    rows = [
        _row(1, 10.0, 12.0, ja="一", id_text="Satu"),
        _row(2, 13.0, 16.0, ja="二", id_text="Dua"),
        _row(3, 17.0, 19.0, ja="三", id_text="Tiga"),
        _row(4, 20.0, 22.0, ja="四", id_text="Empat"),
        _row(5, 23.0, 25.0, ja="五", id_text="Lima"),
    ]
    qc_items = [
        # ID 1: valid timing first (10.5..12.5), followed by incomplete duplicates -> preserves 10.5..12.5
        {"id": 1, "fixed_start": 2.0, "fixed_end": 4.0, "reason": "Geser 1"},
        {"id": 1, "fixed_start": 2.5, "fixed_ja": "一改", "reason": "Update ja"},
        {"id": 1, "fixed_start": None, "fixed_end": 4.5, "reason": "Hanya end"},
        # ID 2: valid split first, followed by empty [] and "bad" -> preserves valid split
        {
            "id": 2,
            "split": [
                {"start": 4.5, "end": 5.5, "ja": "二A", "id_text": "Dua A"},
                {"start": 5.6, "end": 7.0, "ja": "二B", "id_text": "Dua B"},
            ],
            "reason": "Pecah 2",
        },
        {"id": 2, "split": [], "reason": "Dup kosong"},
        {"id": 2, "split": "bad", "reason": "Dup invalid"},
        # ID 3: partial timing only -> warning
        {"id": 3, "fixed_start": 8.0, "reason": "Parsial"},
        # ID 4: NaN and inf timestamps -> warning
        {"id": 4, "fixed_start": float("nan"), "fixed_end": float("inf"), "reason": "Non finite"},
        # ID 5: empty split first, upgraded by valid split + incomplete timing upgraded by complete timing
        {"id": 5, "fixed_start": 14.0, "fixed_end": None, "split": [], "reason": "Awal"},
        {"id": 5, "fixed_start": 14.5, "fixed_end": 16.0, "reason": "Lengkap"},
    ]
    fixed, issues = apply_audio_qc_results(rows, qc_items, slice_start=8.5, slice_duration=20.0)
    # Row 1: timing 10.5..12.5 + ja updated; no warning!
    assert fixed[0]["start"] == pytest.approx(10.5)
    assert fixed[0]["end"] == pytest.approx(12.5)
    assert fixed[0]["ja"] == "一改"
    assert any(i["id"] == 1 and i["type"] == "timing" for i in issues)
    assert not any(i["id"] == 1 and i["type"] == "warning" for i in issues)

    # Row 2: split into 2 rows preserved!
    assert fixed[1]["ja"] == "二A"
    assert fixed[2]["ja"] == "二B"
    assert any(i["id"] == 2 and i["type"] == "split" for i in issues)
    assert not any(i["id"] == 2 and i["type"] == "warning" for i in issues)

    # Row 3: partial timing -> warning, untouched
    assert fixed[3]["start"] == 17.0 and fixed[3]["end"] == 19.0
    assert any(i["id"] == 3 and i["type"] == "warning" for i in issues)

    # Row 4: NaN/inf -> warning, untouched
    assert fixed[4]["start"] == 20.0 and fixed[4]["end"] == 22.0
    assert any(i["id"] == 4 and i["type"] == "warning" for i in issues)

    # Row 5: invalid empty split warned, upgraded timing 8.5+14.5=23.0 .. 8.5+16.0=24.5 applied
    assert fixed[5]["start"] == pytest.approx(23.0)
    assert fixed[5]["end"] == pytest.approx(24.5)
    assert any(i["id"] == 5 and i["type"] == "timing" for i in issues)


def test_run_qc_all_resumes_split_checkpoint_and_sorts_unsorted_multiwindow(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [])
    monkeypatch.setattr("app.qc.audio.cut_chunk", lambda src, dst, s, e: dst.write_bytes(b"slice"))

    rows = [
        _row(1, 2.0, 5.0, ja="A B C", id_text="Tiga pembicara"),
        _row(2, 6.0, 8.0, ja="大丈夫", id_text="Aman"),
    ]
    ckpt_dir = tmp_path / "chunks"
    tmp_dir = tmp_path / "qc_tmp"

    def fake_text(system, user):
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "", "needs_audio_check": True, "reason": "Split"},
            ]),
            input_tokens=10, output_tokens=5,
        ), "gemini-3.6-flash"

    def fake_audio(slice_p, system, user):
        # slice_start = 2.0 - 1.5 = 0.5
        return LLMResponse(
            text=json.dumps([
                {
                    "id": 1,
                    "split": [
                        {"start": 1.5, "end": 2.8, "ja": "- A\n- B", "id_text": "- A\n- B"},
                        {"start": 3.0, "end": 4.5, "ja": "C", "id_text": "C"},
                    ],
                    "reason": "Pecah pembicara",
                },
            ]),
            input_tokens=20, output_tokens=10,
        ), "gemini-3.5-flash"

    fixed1, report1 = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=fake_text, audio_runner=fake_audio,
        checkpoint_dir=ckpt_dir,
    )
    assert len(fixed1) == 3
    assert report1["summary"]["splits"] == 1

    # Resume split row directly from qc_audio_000.json without calling audio_runner
    def fail_runner(*a, **kw):
        raise AssertionError("runners must not be called when split checkpoint exists")

    fixed2, report2 = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=fail_runner, audio_runner=fail_runner,
        checkpoint_dir=ckpt_dir,
    )
    assert fixed2 == fixed1
    assert report2 == report1
    assert [r["id"] for r in fixed2] == [1, 2, 3]

    # Corrupt ja field in qc_audio_000.json must reject checkpoint and recompute
    saved_ckpt = json.loads((ckpt_dir / "qc_audio_000.json").read_text(encoding="utf-8"))
    saved_ckpt["fixed"][0]["ja"] = 123
    (ckpt_dir / "qc_audio_000.json").write_text(json.dumps(saved_ckpt), encoding="utf-8")
    recomputed = []

    def recompute_audio(slice_p, system, user):
        recomputed.append(1)
        return fake_audio(slice_p, system, user)

    fixed3, _ = run_qc_all(
        None, rows, audio_path=audio_path, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=fail_runner, audio_runner=recompute_audio,
        checkpoint_dir=ckpt_dir,
    )
    assert recomputed == [1]
    assert fixed3 == fixed1

    # Unsorted >250 rows passed to run_qc_all -> Pass 1B multi-window receives chronological order
    unsorted_260 = [
        _row(i, float(i), float(i) + 0.5, ja="あ", id_text=f"Teks {i}")
        for i in range(260, 0, -1)
    ]
    pass1_prompts: list[str] = []

    def capture_text(system, user):
        pass1_prompts.append(user)
        return LLMResponse(text="[]", input_tokens=1, output_tokens=1), "gemini-3.6-flash"

    fixed_sorted, _ = run_qc_all(
        None, unsorted_260, audio_path=None, tmp_dir=tmp_dir,
        system="SYS", tracker=None,
        text_runner=capture_text,
    )
    assert len(pass1_prompts) == 2
    w1_ids = [it["id"] for it in _extract_prompt_items(pass1_prompts[0])]
    w2_ids = [it["id"] for it in _extract_prompt_items(pass1_prompts[1])]
    assert w1_ids == list(range(1, 201))
    assert w2_ids == list(range(171, 261))
    assert [r["id"] for r in fixed_sorted] == list(range(1, 261))


