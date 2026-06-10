"""API-client abstraction: config, retry, usage tracking, LLM clients."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

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
