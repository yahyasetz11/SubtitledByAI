import pytest

from app.audio import (
    DEFAULT_CHUNK_MAX_SECONDS, DEFAULT_CHUNK_MIN_SECONDS,
    OVERLAP_SECONDS, PlannedChunk, Silence, parse_silences, plan_chunks,
)

SAMPLE_STDERR = """\
[mp3float @ 000001] Header missing
[silencedetect @ 000002] silence_start: 612.501
[silencedetect @ 000002] silence_end: 613.4 | silence_duration: 0.899
size=N/A time=00:14:00.00 bitrate=N/A speed= 980x
[silencedetect @ 000002] silence_start: 1305.2
[silencedetect @ 000002] silence_end: 1306.0 | silence_duration: 0.8
"""


def test_parse_silences_pairs_start_end():
    silences = parse_silences(SAMPLE_STDERR)
    assert silences == [Silence(612.501, 613.4), Silence(1305.2, 1306.0)]
    assert silences[0].duration == pytest.approx(0.899)
    assert silences[0].midpoint == pytest.approx(612.9505)


def test_parse_silences_ignores_unmatched_trailing_start():
    silences = parse_silences("[x] silence_start: 5.0\n")
    assert silences == []


def make_lookup(by_threshold):
    """dict {-30: [Silence...]} -> get_silences callable; missing -> []"""
    return lambda threshold: by_threshold.get(threshold, [])


def test_plan_single_chunk_for_short_audio():
    # A 5-minute audio fits in one chunk with default 4–6 min settings
    chunks = plan_chunks(300.0, make_lookup({}))
    assert chunks == [PlannedChunk(index=1, start=0.0, end=300.0,
                                   overlap_prev=False)]


def test_plan_cuts_at_longest_silence_in_window():
    # Use old 10–14 min window explicitly to preserve original test intent
    MIN, MAX = 600.0, 840.0
    silences = [
        Silence(300.0, 301.0),    # outside window (before min 10)
        Silence(650.0, 650.6),    # in window, 0.6s
        Silence(700.0, 701.2),    # in window, 1.2s  <- best
        Silence(900.0, 901.0),    # outside window (after min 14)
    ]
    chunks = plan_chunks(1400.0, make_lookup({-30: silences}),
                         min_seconds=MIN, max_seconds=MAX)
    assert chunks[0].end == pytest.approx(700.6)   # midpoint of best silence
    assert chunks[1] == PlannedChunk(2, pytest.approx(700.6), 1400.0, False)


def test_plan_cascades_to_quieter_thresholds():
    MIN, MAX = 600.0, 840.0
    chunks = plan_chunks(1400.0, make_lookup({
        -30: [],
        -25: [],
        -20: [Silence(720.0, 720.8)],
    }), min_seconds=MIN, max_seconds=MAX)
    assert chunks[0].end == pytest.approx(720.4)
    assert chunks[1].overlap_prev is False


def test_plan_hard_cut_with_overlap_when_no_silence():
    MIN, MAX = 600.0, 840.0
    chunks = plan_chunks(1400.0, make_lookup({}), min_seconds=MIN, max_seconds=MAX)
    assert chunks[0] == PlannedChunk(1, 0.0, MAX, False)
    # next chunk starts OVERLAP_SECONDS before the hard cut and is flagged
    assert chunks[1].start == pytest.approx(MAX - OVERLAP_SECONDS)
    assert chunks[1].overlap_prev is True
    assert chunks[1].end == 1400.0


def test_plan_ignores_too_short_silences():
    MIN, MAX = 600.0, 840.0
    chunks = plan_chunks(1400.0, make_lookup({-30: [Silence(700.0, 700.3)]}),
                         min_seconds=MIN, max_seconds=MAX)
    assert chunks[0].end == MAX  # 0.3s < 0.5s minimum -> hard cut


def test_plan_long_audio_produces_sequential_chunks():
    MIN, MAX = 600.0, 840.0
    silences = [Silence(660.0, 661.0), Silence(1380.0, 1381.0)]
    chunks = plan_chunks(2000.0, make_lookup({-30: silences}),
                         min_seconds=MIN, max_seconds=MAX)
    assert [c.index for c in chunks] == [1, 2, 3]
    assert chunks[0].end == chunks[1].start
    assert chunks[1].end == chunks[2].start
    assert chunks[2].end == 2000.0


def test_plan_chunks_respects_default_constants():
    """Defaults should match DEFAULT_CHUNK_MIN/MAX_SECONDS (4–6 min)."""
    # 7 min audio: fits in one default-sized chunk (max 6 min)? No — should split.
    # 5 min audio: fits entirely in one chunk.
    chunks_short = plan_chunks(300.0, make_lookup({}))
    assert len(chunks_short) == 1
    assert chunks_short[0].end == 300.0

    # 8 min audio: exceeds 6 min max, must produce at least 2 chunks (hard cut).
    chunks_long = plan_chunks(480.0, make_lookup({}))
    assert len(chunks_long) >= 2
    assert chunks_long[0].end == pytest.approx(DEFAULT_CHUNK_MAX_SECONDS)


def test_plan_chunks_custom_size_4_6_minutes():
    """4–6 min window: a 10-min audio with silence at 4.5 min cuts cleanly."""
    MIN, MAX = 240.0, 360.0  # 4–6 min
    silence_at_4_5_min = [Silence(268.0, 269.0)]  # midpoint 268.5 s (~4.5 min)
    chunks = plan_chunks(600.0, make_lookup({-30: silence_at_4_5_min}),
                         min_seconds=MIN, max_seconds=MAX)
    assert len(chunks) == 2
    assert chunks[0].end == pytest.approx(268.5)
    assert chunks[1].start == pytest.approx(268.5)
    assert chunks[1].end == 600.0
    assert chunks[1].overlap_prev is False


from app.audio import (
    build_cut_args, build_normalize_args, build_silence_args, build_ytdlp_args,
)


def test_build_normalize_args_mono_16k_64kbps():
    args = build_normalize_args("in.mp4", "out.mp3")
    assert args == ["ffmpeg", "-y", "-i", "in.mp4", "-vn", "-ac", "1",
                    "-ar", "16000", "-b:a", "64k", "out.mp3"]


def test_build_silence_args_uses_threshold():
    args = build_silence_args("audio.mp3", -25)
    assert "-af" in args
    assert "silencedetect=noise=-25dB:d=0.5" in args
    assert args[-2:] == ["-f", "null"] or args[-3:] == ["-f", "null", "-"]


def test_build_cut_args_seeks_before_input_and_copies():
    args = build_cut_args("audio.mp3", "chunks/chunk_001.mp3", 10.0, 614.25)
    i_input = args.index("-i")
    assert args.index("-ss") < i_input          # fast seek
    assert "604.250" in args                    # -t duration, not -to
    assert "copy" in args


def test_build_ytdlp_args_audio_only_default():
    args = build_ytdlp_args("https://youtu.be/x", "out/source.%(ext)s",
                            save_mp4=False, cookies_file=None)
    assert "bestaudio/best" in args
    assert "--cookies" not in args


def test_build_ytdlp_args_mp4_and_cookies():
    args = build_ytdlp_args("https://youtu.be/x", "out/source.%(ext)s",
                            save_mp4=True, cookies_file="c.txt")
    assert "bestvideo*+bestaudio/best" in args
    assert "mp4" in args
    assert "--cookies" in args and "c.txt" in args


def test_build_ytdlp_args_enables_remote_ejs():
    args = build_ytdlp_args("https://youtu.be/x", "out/source.%(ext)s",
                            save_mp4=False, cookies_file=None)
    assert "--remote-components" in args
    assert "ejs:github" in args
