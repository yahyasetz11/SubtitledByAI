"""API-client abstraction: config, retry, usage tracking, LLM clients."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

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
