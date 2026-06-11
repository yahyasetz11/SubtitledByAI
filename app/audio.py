"""Audio: extraction, normalization, silencedetect, chunking (ffmpeg/yt-dlp)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# §5: target chunk 10-14 minutes, cut mid-silence; else hard cut + overlap.
TARGET_MIN_SECONDS = 600.0
TARGET_MAX_SECONDS = 840.0
OVERLAP_SECONDS = 4.0
MIN_SILENCE_DURATION = 0.5
THRESHOLD_CASCADE_DB = (-30, -25, -20)


class AudioError(Exception):
    """ffmpeg/yt-dlp failure; message is surfaced to the UI."""


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass
class PlannedChunk:
    index: int
    start: float
    end: float
    overlap_prev: bool  # True when this chunk re-covers the tail of the previous


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def parse_silences(stderr: str) -> list[Silence]:
    silences: list[Silence] = []
    pending_start: float | None = None
    for line in stderr.splitlines():
        if m := _SILENCE_START_RE.search(line):
            pending_start = float(m.group(1))
        elif (m := _SILENCE_END_RE.search(line)) and pending_start is not None:
            silences.append(Silence(pending_start, float(m.group(1))))
            pending_start = None
    return silences


def _find_cut(start: float, get_silences) -> float | None:
    window_lo = start + TARGET_MIN_SECONDS
    window_hi = start + TARGET_MAX_SECONDS
    for threshold in THRESHOLD_CASCADE_DB:
        candidates = [
            s for s in get_silences(threshold)
            if s.duration >= MIN_SILENCE_DURATION
            and window_lo <= s.midpoint <= window_hi
        ]
        if candidates:
            # Longest silence = cleanest cut; ties resolved by lateness.
            best = max(candidates, key=lambda s: (s.duration, s.midpoint))
            return best.midpoint
    return None


def plan_chunks(duration: float, get_silences) -> list[PlannedChunk]:
    """get_silences(threshold_db) -> list[Silence]; called lazily per threshold."""
    chunks: list[PlannedChunk] = []
    start = 0.0
    index = 1
    overlap_prev = False
    while True:
        if duration - start <= TARGET_MAX_SECONDS:
            chunks.append(PlannedChunk(index, start, duration, overlap_prev))
            return chunks
        cut = _find_cut(start, get_silences)
        if cut is not None:
            chunks.append(PlannedChunk(index, start, cut, overlap_prev))
            start, overlap_prev = cut, False
        else:
            hard_cut = start + TARGET_MAX_SECONDS
            chunks.append(PlannedChunk(index, start, hard_cut, overlap_prev))
            start, overlap_prev = hard_cut - OVERLAP_SECONDS, True
        index += 1


def write_offsets(chunks: list[PlannedChunk], path: Path) -> None:
    payload = [{**asdict(c), "file": f"chunk_{c.index:03d}.mp3"} for c in chunks]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_offsets(path: Path) -> list[PlannedChunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [PlannedChunk(r["index"], r["start"], r["end"], r["overlap_prev"])
            for r in raw]


def build_normalize_args(src: str, dst: str) -> list[str]:
    """Any input -> mp3 mono 16kHz 64kbps (§3: ~3-4x smaller uploads)."""
    return ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1",
            "-ar", "16000", "-b:a", "64k", str(dst)]


def build_silence_args(path: str, threshold_db: int) -> list[str]:
    return ["ffmpeg", "-i", str(path), "-af",
            f"silencedetect=noise={threshold_db}dB:d={MIN_SILENCE_DURATION}",
            "-f", "null", "-"]


def build_cut_args(src: str, dst: str, start: float, end: float) -> list[str]:
    # -ss before -i = fast input seek; stream copy keeps it instant.
    return ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(src), "-c", "copy", str(dst)]


def build_ytdlp_args(url: str, output_template: str, *, save_mp4: bool,
                     cookies_file: str | None) -> list[str]:
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist",
            "-o", output_template]
    if save_mp4:
        args += ["-f", "bestvideo*+bestaudio/best", "--merge-output-format", "mp4"]
    else:
        args += ["-f", "bestaudio/best"]
    if cookies_file:
        args += ["--cookies", cookies_file]
    args.append(url)
    return args


def _run(args: list[str], error_hint: str = "") -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise AudioError(f"Program tidak ditemukan: {args[0]}. "
                         "Pastikan ffmpeg ada di PATH.") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise AudioError(f"{args[0]} gagal (exit {proc.returncode}). "
                         f"{error_hint}\n{tail}")
    return proc


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def normalize_audio(src: Path, dst: Path) -> None:
    _run(build_normalize_args(str(src), str(dst)))


def get_duration(path: Path) -> float:
    proc = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(proc.stdout.strip())


def detect_silences(path: Path, threshold_db: int) -> list[Silence]:
    # silencedetect logs to stderr; exit code 0 either way.
    proc = _run(build_silence_args(str(path), threshold_db))
    return parse_silences(proc.stderr)


def cut_chunk(src: Path, dst: Path, start: float, end: float) -> None:
    _run(build_cut_args(str(src), str(dst), start, end))


def split_audio(chunk_path: Path) -> tuple[Path, Path, float]:
    """Halve a chunk (for transcription-truncation recovery). Returns
    (left_path, right_path, right_offset_seconds)."""
    mid = get_duration(chunk_path) / 2
    left = chunk_path.with_name(chunk_path.stem + "_a.mp3")
    right = chunk_path.with_name(chunk_path.stem + "_b.mp3")
    cut_chunk(chunk_path, left, 0.0, mid)
    cut_chunk(chunk_path, right, mid, mid * 2 + 1.0)
    return left, right, mid


def download_youtube(url: str, job_dir: Path, *, save_mp4: bool,
                     cookies_file: str | None) -> Path:
    template = str(job_dir / "source.%(ext)s")
    args = build_ytdlp_args(url, template, save_mp4=save_mp4,
                            cookies_file=cookies_file)
    _run(args, error_hint=(
        "Jika YouTube berubah, perbarui yt-dlp: pip install -U yt-dlp."))
    matches = sorted(job_dir.glob("source.*"))
    if not matches:
        raise AudioError("yt-dlp selesai tetapi file source.* tidak ditemukan.")
    return matches[0]
