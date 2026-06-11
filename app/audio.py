"""Audio: extraction, normalization, silencedetect, chunking (ffmpeg/yt-dlp)."""
from __future__ import annotations

import json
import re
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
