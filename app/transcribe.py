"""Gemini audio -> structured JP transcript (§6a)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path

from app import audio
from app.providers import call_with_retries


class TranscriptParseError(Exception):
    """Model output is not a usable transcript (bad JSON / empty / truncated)."""


@dataclass(frozen=True)
class Utterance:
    id: int
    start: float
    end: float
    type: str  # "dialogue" | "narration"
    ja: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(id=int(d["id"]), start=float(d["start"]), end=float(d["end"]),
                   type=d.get("type", "dialogue"), ja=d["ja"])


def parse_timestamp(value: str) -> float:
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3 or any(p.strip() == "" for p in parts):
        raise ValueError(f"timestamp tidak valid: {value!r}")
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2]) if len(parts) >= 2 else 0
        hours = int(parts[-3]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"timestamp tidak valid: {value!r}") from exc
    return hours * 3600 + minutes * 60 + seconds


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def _extract_json_array(text: str) -> list:
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise TranscriptParseError("output bukan array JSON")
    return data


def parse_transcript(text: str) -> list[Utterance]:
    try:
        items = _extract_json_array(text)
    except (json.JSONDecodeError, TranscriptParseError) as exc:
        raise TranscriptParseError(f"JSON transkrip tidak valid: {exc}") from exc

    utterances: list[Utterance] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ja = str(item.get("ja", "")).strip()
        if not ja:
            continue
        try:
            start = parse_timestamp(item["start"])
            end = parse_timestamp(item["end"])
        except (KeyError, ValueError):
            continue
        if end <= start:
            end = start + 0.5
        utype = item.get("type", "dialogue")
        if utype not in ("dialogue", "narration"):
            utype = "dialogue"
        utterances.append(Utterance(id=len(utterances) + 1, start=start,
                                    end=end, type=utype, ja=ja))
    if not utterances:
        raise TranscriptParseError("transkrip kosong setelah validasi")
    return utterances


def build_transcribe_prompts(context_md: str, members_md: str) -> tuple[str, str]:
    system = (
        "Kamu adalah transcriber profesional untuk audio variety show Jepang "
        "(Soko Magattara, Sakurazaka?). Gunakan konteks berikut untuk menulis "
        "nama member dengan ejaan yang benar dan memahami istilah acara.\n\n"
        "=== KONTEKS ACARA ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )
    user = (
        "Transkripsikan audio terlampir per ujaran ke bahasa Jepang.\n"
        "Balas HANYA array JSON (tanpa teks lain) dengan skema per item:\n"
        '{"id": 1, "start": "MM:SS.mmm", "end": "MM:SS.mmm", '
        '"type": "dialogue" | "narration", "ja": "..."}\n'
        "Aturan:\n"
        "- id mulai dari 1, berurutan.\n"
        "- start/end relatif terhadap awal audio INI (format MM:SS.mmm).\n"
        '- type "narration" untuk voice-over narator; selain itu "dialogue".\n'
        "- Abaikan backchannel pendek tanpa makna (うん/はい) kecuali jawaban penting.\n"
        "- Teks layar (telop) tidak ditranskripsikan kecuali dibacakan."
    )
    return system, user
