"""Deterministic subtitle post-processing + .ass/.srt rendering (§7, §8)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SubEvent:
    start: float
    end: float
    text: str
    type: str = "dialogue"  # "dialogue" | "narration"

    @property
    def duration(self) -> float:
        return self.end - self.start


def postprocess(events: list[SubEvent], *, min_duration: float,
                max_duration: float, merge_gap: float,
                cps_threshold: float) -> tuple[list[SubEvent], list[dict]]:
    evs = sorted((SubEvent(e.start, e.end, e.text.strip(), e.type)
                  for e in events if e.text.strip()), key=lambda e: e.start)
    evs = _merge_short(evs, min_duration, merge_gap, max_duration)
    evs = _extend_short(evs, min_duration)
    evs = _split_long(evs, max_duration)
    evs = _fix_overlaps(evs)
    return evs, _cps_flags(evs, cps_threshold)


def _merge_short(events, min_duration, merge_gap, max_duration):
    """§7: short reaction + continuation, gap < merge_gap, same type -> one block."""
    result: list[SubEvent] = []
    for ev in events:
        if result:
            prev = result[-1]
            gap = ev.start - prev.end
            short_involved = (prev.duration < min_duration
                              or ev.duration < min_duration)
            if (gap < merge_gap and ev.type == prev.type and short_involved
                    and (ev.end - prev.start) <= max_duration):
                result[-1] = SubEvent(prev.start, ev.end,
                                      f"{prev.text} {ev.text}", prev.type)
                continue
        result.append(ev)
    return result


def _extend_short(events, min_duration):
    for i, ev in enumerate(events):
        if ev.duration < min_duration:
            limit = events[i + 1].start if i + 1 < len(events) else float("inf")
            ev.end = min(ev.start + min_duration, limit)
    return events


_SPLIT_AFTER = "。！？!?.,、"


def _split_point(text: str) -> int | None:
    """Index to cut at (start of the second part), nearest to the middle."""
    target = len(text) / 2
    best: int | None = None
    for i, ch in enumerate(text):
        if ch in _SPLIT_AFTER and 0 < i < len(text) - 1:
            pos = i + 1
            if best is None or abs(pos - target) < abs(best - target):
                best = pos
    if best is None:
        for i, ch in enumerate(text):
            if ch == " " and 0 < i < len(text) - 1:
                if best is None or abs(i - target) < abs(best - target):
                    best = i
    return best


def _split_event(ev: SubEvent, max_duration: float) -> list[SubEvent]:
    if ev.duration <= max_duration:
        return [ev]
    cut = _split_point(ev.text)
    if cut is None:
        return [ev]  # nothing sane to split on; CPS flag will catch it if fast
    first, second = ev.text[:cut].strip(), ev.text[cut:].strip()
    if not first or not second:
        return [ev]
    ratio = len(first) / (len(first) + len(second))
    mid = ev.start + ev.duration * ratio
    return (_split_event(SubEvent(ev.start, mid, first, ev.type), max_duration)
            + _split_event(SubEvent(mid, ev.end, second, ev.type), max_duration))


def _split_long(events, max_duration):
    out: list[SubEvent] = []
    for ev in events:
        out.extend(_split_event(ev, max_duration))
    return out


def _fix_overlaps(events):
    """§7: shift the next block to start after the previous one."""
    for i in range(1, len(events)):
        prev, cur = events[i - 1], events[i]
        if cur.start < prev.end:
            cur.start = prev.end
            if cur.end <= cur.start:
                cur.end = cur.start + 0.5
    return events


def _cps_flags(events, threshold: float) -> list[dict]:
    flags = []
    for index, ev in enumerate(events, start=1):
        cps = len(ev.text) / ev.duration if ev.duration > 0 else float("inf")
        if cps > threshold:
            flags.append({"index": index, "start": round(ev.start, 2),
                          "cps": round(cps, 1), "text": ev.text})
    return flags

def format_ass_time(seconds: float) -> str:
    total_cs = round(seconds * 100)
    cs = total_cs % 100
    s = (total_cs // 100) % 60
    m = (total_cs // 6000) % 60
    h = total_cs // 360000
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def format_srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_FALLBACK_EVENTS_HEADER = [
    "", "[Events]",
    "Format: Layer, Start, End, Style, Actor, "
    "MarginL, MarginR, MarginV, Effect, Text",
]


def render_ass(events: list[SubEvent], template_text: str) -> str:
    """Copy the template verbatim through the [Events] Format line, then
    append Dialogue lines (§8). Narration -> Narrator style."""
    lines = template_text.rstrip().splitlines()
    header = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "[events]":
            for j in range(i + 1, len(lines)):
                if lines[j].strip().lower().startswith("format:"):
                    header = lines[:j + 1]
                    break
            break
    if header is None:
        header = lines + _FALLBACK_EVENTS_HEADER

    out = list(header)
    for ev in events:
        style = "Narrator" if ev.type == "narration" else "Default"
        text = ev.text.replace("\r\n", "\n").replace("\n", "\\N")
        out.append(f"Dialogue: 0,{format_ass_time(ev.start)},"
                   f"{format_ass_time(ev.end)},{style},,0,0,0,,{text}")
    return "\n".join(out) + "\n"


def render_srt(events: list[SubEvent]) -> str:
    blocks = []
    for index, ev in enumerate(events, start=1):
        blocks.append(f"{index}\n{format_srt_time(ev.start)} --> "
                      f"{format_srt_time(ev.end)}\n{ev.text}\n")
    return "\n".join(blocks)
