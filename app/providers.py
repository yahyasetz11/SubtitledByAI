"""API-client abstraction: config, retry, usage tracking, LLM clients."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv


class ConfigError(Exception):
    """Startup configuration problem; message is shown to the user."""


@dataclass
class Config:
    gemini_api_key: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    transcribe_model: str
    translate_models: dict[str, str]
    ytdlp_cookies_file: str | None
    sub_min_duration: float
    sub_max_duration: float
    sub_merge_gap: float
    sub_cps_flag: float

    @classmethod
    def load(cls) -> "Config":
        load_dotenv(override=False)
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not gemini_key:
            raise ConfigError(
                "GEMINI_API_KEY belum diisi. Salin .env.example menjadi .env "
                "lalu isi GEMINI_API_KEY (wajib untuk transkripsi & translasi)."
            )
        return cls(
            gemini_api_key=gemini_key,
            openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip() or None,
            transcribe_model=os.getenv("TRANSCRIBE_MODEL", "gemini-2.5-pro"),
            translate_models={
                "gemini": os.getenv("TRANSLATE_MODEL_GEMINI", "gemini-2.5-flash"),
                "openai": os.getenv("TRANSLATE_MODEL_OPENAI", "gpt-4o"),
                "anthropic": os.getenv("TRANSLATE_MODEL_ANTHROPIC", "claude-sonnet-4-6"),
            },
            ytdlp_cookies_file=(os.getenv("YTDLP_COOKIES_FILE") or "").strip() or None,
            sub_min_duration=float(os.getenv("SUB_MIN_DURATION", "0.7")),
            sub_max_duration=float(os.getenv("SUB_MAX_DURATION", "7.5")),
            sub_merge_gap=float(os.getenv("SUB_MERGE_GAP", "0.5")),
            sub_cps_flag=float(os.getenv("SUB_CPS_FLAG", "25")),
        )

    def available_providers(self) -> dict[str, bool]:
        return {
            "gemini": bool(self.gemini_api_key),
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
        }


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
NETWORK_ERRORS = (ConnectionError, TimeoutError, httpx.TransportError)


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after_of(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def call_with_retries(fn, *, attempts: int = 4, base_delay: float = 2.0,
                      sleep=time.sleep):
    """Exponential backoff for transient transport errors (429/5xx/network).

    `attempts` is the total number of tries. Honors Retry-After when present.
    Non-retryable errors (or exhausted attempts) propagate to the caller.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            status = _status_of(exc)
            retryable = status in RETRYABLE_STATUS or isinstance(exc, NETWORK_ERRORS)
            if not retryable or attempt == attempts:
                raise
            delay = _retry_after_of(exc)
            if delay is None:
                delay = base_delay * (2 ** (attempt - 1))
            sleep(delay)


# USD per 1M tokens: (input, output). Update when prices change.
PRICING = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}


class UsageTracker:
    """Accumulates usage_metadata per model; persists to usage.json on every add."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"models": {}}

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        entry = self.data["models"].setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        entry["input_tokens"] += int(input_tokens or 0)
        entry["output_tokens"] += int(output_tokens or 0)
        entry["calls"] += 1
        self._save()

    def summary(self) -> dict:
        total_in = sum(m["input_tokens"] for m in self.data["models"].values())
        total_out = sum(m["output_tokens"] for m in self.data["models"].values())
        cost = 0.0
        unpriced = []
        for model, m in self.data["models"].items():
            if model in PRICING:
                price_in, price_out = PRICING[model]
                cost += m["input_tokens"] / 1e6 * price_in
                cost += m["output_tokens"] / 1e6 * price_out
            else:
                unpriced.append(model)
        line = (f"Tokens: {_fmt_tokens(total_in)} in / {_fmt_tokens(total_out)} out"
                f" (~${cost:.2f})")
        return {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": round(cost, 6),
            "unpriced_models": sorted(unpriced),
            "line": line,
        }

    def _save(self) -> None:
        self.data["summary"] = None  # placeholder so key order is stable
        self.data["summary"] = {
            k: v for k, v in self.summary().items() if k != "line"
        }
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)
