import pytest

from app.subtitle import SubEvent, postprocess


def E(start, end, text, type="dialogue"):
    return SubEvent(start=start, end=end, text=text, type=type)


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
