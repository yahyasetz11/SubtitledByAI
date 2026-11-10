import pytest

from app.subtitle import SubEvent, postprocess


def E(start, end, text, type="dialogue", turns=0):
    return SubEvent(start=start, end=end, text=text, type=type, turns=turns)


def run(events, **kw):
    defaults = dict(min_duration=0.7, max_duration=7.5, merge_gap=0.5,
                    cps_threshold=25.0)
    defaults.update(kw)
    return postprocess(events, **defaults)


def test_merges_short_reaction_with_continuation():
    # spec §7 example: "Aku senang sekali. Impianku jadi kenyataan."
    events, _ = run([E(0.0, 0.5, "Aku senang sekali."),
                     E(0.8, 2.5, "Impianku jadi kenyataan.")])
    assert len(events) == 1
    assert events[0].text == "Aku senang sekali. Impianku jadi kenyataan."
    assert events[0].start == 0.0 and events[0].end == 2.5


def test_does_not_merge_across_types():
    events, _ = run([E(0.0, 0.5, "Eh!?"),
                     E(0.8, 2.5, "Episode hari ini...", type="narration")])
    assert len(events) == 2


def test_does_not_merge_when_gap_too_large():
    events, _ = run([E(0.0, 0.5, "Eh!?"), E(1.2, 3.0, "Lanjut.")])
    assert len(events) == 2


def test_extends_short_event_to_min_duration():
    events, _ = run([E(0.0, 0.4, "Oke."), E(5.0, 7.0, "Lanjut.")])
    assert events[0].end == pytest.approx(0.7)


def test_extension_clamped_by_next_event():
    events, _ = run([E(0.0, 0.4, "Oke.", type="narration"),
                     E(0.6, 3.0, "Lanjut.")])  # different type: no merge
    assert events[0].end == pytest.approx(0.6)


def test_splits_long_event_at_sentence_pause():
    text = "Halo semua, selamat datang. Hari ini kita main game seru banget loh."
    events, _ = run([E(0.0, 10.0, text)])
    assert len(events) == 2
    assert events[0].text == "Halo semua, selamat datang."
    assert events[1].text == "Hari ini kita main game seru banget loh."
    assert events[0].end == pytest.approx(events[1].start)
    assert events[0].end == pytest.approx(10.0 * len("Halo semua, selamat datang.")
                                          / len(text.replace(" ", " ")), abs=0.6)
    assert events[1].end == pytest.approx(10.0)


def test_split_falls_back_to_space():
    text = "kata " * 40  # no punctuation, 200 chars
    events, _ = run([E(0.0, 9.0, text.strip())])
    assert len(events) >= 2
    assert all(e.duration <= 7.5 for e in events)


def test_unsplittable_long_event_left_alone():
    events, _ = run([E(0.0, 8.0, "a" * 50)])  # no punct, no space
    assert len(events) == 1


def test_overlap_shifts_next_block():
    events, _ = run([E(0.0, 5.0, "Pertama panjang sekali ya."),
                     E(4.0, 6.0, "Kedua juga lumayan.")])
    assert events[0].end == pytest.approx(5.0)
    assert events[1].start == pytest.approx(5.0)
    assert events[1].end == pytest.approx(6.0)


def test_cps_flags_fast_lines_without_cutting():
    fast = "x" * 100
    events, flags = run([E(0.0, 2.0, fast)])  # 50 cps
    assert events[0].text == fast              # never auto-cut (editorial call)
    assert len(flags) == 1
    assert flags[0]["cps"] == pytest.approx(50.0)
    assert flags[0]["index"] == 1


def test_events_sorted_by_start():
    events, _ = run([E(5.0, 6.0, "Kedua."), E(0.0, 1.0, "Pertama.")])
    assert [e.text for e in events] == ["Pertama.", "Kedua."]

from pathlib import Path

from app.subtitle import format_ass_time, format_srt_time, render_ass, render_srt

TEMPLATE = Path("context/template.ass").read_text(encoding="utf-8")


def test_format_ass_time():
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(3661.456) == "1:01:01.46"
    assert format_ass_time(59.999) == "0:01:00.00"  # centisecond carry


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(83.4567) == "00:01:23,457"
    assert format_srt_time(3600) == "01:00:00,000"


def test_render_ass_copies_template_header_and_appends_dialogue():
    events = [E(1.0, 2.5, "Halo semua!"),
              E(3.0, 4.0, "Episode kali ini...", type="narration")]
    out = render_ass(events, TEMPLATE)
    assert "[Script Info]" in out
    assert "PlayResX: 1920" in out
    assert "Style: Default,Comic Sans MS" in out
    lines = out.splitlines()
    fmt_i = next(i for i, l in enumerate(lines) if l.startswith("Format: Layer"))
    assert lines[fmt_i + 1] == \
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Halo semua!"
    assert lines[fmt_i + 2] == \
        "Dialogue: 0,0:00:03.00,0:00:04.00,Narrator,,0,0,0,,Episode kali ini..."


def test_render_ass_escapes_newlines():
    out = render_ass([E(0.0, 1.0, "baris satu\nbaris dua")], TEMPLATE)
    assert "baris satu\\Nbaris dua" in out


def test_render_srt_format():
    events = [E(1.0, 2.5, "Halo semua!"), E(3.0, 4.0, "Lanjut.")]
    assert render_srt(events) == (
        "1\n00:00:01,000 --> 00:00:02,500\nHalo semua!\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLanjut.\n"
    )


# ---------------------------------------------------------------------------
# parse_ass_utterances tests
# ---------------------------------------------------------------------------

from app.subtitle import parse_ass_utterances  # noqa: E402

_SAMPLE_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,今日は楽しいですね
Dialogue: 0,0:00:04.00,0:00:06.00,Narrator,,0,0,0,,ナレーション
Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,また来週
"""


def test_parse_ass_utterances_count():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert len(utts) == 3


def test_parse_ass_utterances_timing():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[0].start == pytest.approx(1.0)
    assert utts[0].end == pytest.approx(3.5)


def test_parse_ass_utterances_text():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[0].ja == '今日は楽しいですね'


def test_parse_ass_utterances_narration_type():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[1].type == 'narration'
    assert utts[0].type == 'dialogue'


def test_parse_ass_utterances_ids_sequential():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert [u.id for u in utts] == [1, 2, 3]


def test_parse_ass_utterances_newline_normalization():
    ass = (
        "[Events]\nFormat: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,一行目\\N二行目\n"
    )
    utts = parse_ass_utterances(ass)
    assert utts[0].ja == "一行目\n二行目"


def test_merge_short_never_chains_past_two_blocks():
    # Four rapid <0.7s blocks should NOT all collapse into one giant 4-speaker line.
    events, _ = run([
        E(0.0, 0.4, "Eh!?"),
        E(0.5, 0.9, "Serius?!"),
        E(1.0, 1.4, "Masa sih!?"),
        E(1.5, 1.9, "Keren banget!"),
    ])
    assert len(events) == 2
    assert events[0].text == "Eh!? Serius?!"
    assert events[1].text == "Masa sih!? Keren banget!"


def test_merge_short_does_not_merge_two_speaker_dashed_block():
    two_speaker = "- Eh!?\n- Serius?!"
    events, _ = run([
        E(0.0, 0.5, two_speaker),
        E(0.6, 1.0, "Gila sih!"),
    ])
    assert len(events) == 2
    assert events[0].text == two_speaker
    assert events[1].text == "Gila sih!"


def test_render_ass_preserves_two_speaker_dashed_lines():
    out = render_ass([E(0.0, 1.5, "- Line 1\n- Line 2")], TEMPLATE)
    assert "Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,- Line 1\\N- Line 2" in out


def test_postprocess_stripping_preserves_turns():
    events, _ = run([
        E(0.0, 0.4, "  Halo!  ", turns=2),
        E(0.5, 0.9, "  Dunia!  "),
    ])
    assert len(events) == 2
    assert events[0].text == "Halo!"
    assert events[0].turns == 2
    assert events[1].text == "Dunia!"
    assert events[1].turns == 1


def test_subevent_equality_ignores_turns():
    assert E(0.0, 1.0, "Halo!", turns=1) == E(0.0, 1.0, "Halo!", turns=2)
    assert E(0.0, 1.0, "Halo!", turns=2).turns == 2


def test_merge_short_does_not_merge_single_line_dashed_cues():
    events_first_dashed, _ = run([
        E(0.0, 0.5, "- Eh!?"),
        E(0.6, 1.0, "Serius?!"),
    ])
    assert len(events_first_dashed) == 2
    assert events_first_dashed[0].text == "- Eh!?"
    assert events_first_dashed[1].text == "Serius?!"

    events_both_dashed, _ = run([
        E(0.0, 0.5, "- Eh!?"),
        E(0.6, 1.0, "- Serius?!"),
    ])
    assert len(events_both_dashed) == 2
    assert events_both_dashed[0].text == "- Eh!?"
    assert events_both_dashed[1].text == "- Serius?!"

    events_second_dashed, _ = run([
        E(0.0, 0.5, "Eh!?"),
        E(0.6, 1.0, "- Serius?!"),
    ])
    assert len(events_second_dashed) == 2
    assert events_second_dashed[0].text == "Eh!?"
    assert events_second_dashed[1].text == "- Serius?!"


def test_merge_short_merges_multiline_with_midline_hyphen():
    events, _ = run([
        E(0.0, 0.5, "Kata - pertama\nlanjutan"),
        E(0.6, 1.0, "Oke"),
    ])
    assert len(events) == 1
    assert events[0].text == "Kata - pertama\nlanjutan Oke"


def test_merge_short_does_not_merge_two_speaker_dashed_block_second():
    two_speaker = "- Eh!?\n- Serius?!"
    events, _ = run([
        E(0.0, 0.5, "Gila sih!"),
        E(0.6, 1.2, two_speaker),
    ])
    assert len(events) == 2
    assert events[0].text == "Gila sih!"
    assert events[1].text == two_speaker
