"""API-client abstraction: config, retry, usage tracking, LLM clients."""
from __future__ import annotations

import base64
import itertools
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv


class ConfigError(Exception):
    """Startup configuration problem; message is shown to the user."""


MODELS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.json"


class TranscriptionDict(dict):
    """Dict subclass providing backwards-compatibility for integer indexing, slicing, and model containment."""

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return self.get("gemini", [])[key]
        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, str):
            if super().__contains__(key):
                return True
            for model_list in self.values():
                if isinstance(model_list, (list, tuple)) and key in model_list:
                    return True
            return False
        if isinstance(key, dict):
            if not key:
                return False
            for k, v in key.items():
                if not (k in self and dict.__contains__(self, k)):
                    return False
                target = self[k]
                if v == target:
                    continue
                if isinstance(target, (list, tuple, set, dict)) and v in target:
                    continue
                return False
            return True
        try:
            return super().__contains__(key)
        except TypeError:
            return False


class TranscriptionDefaultDict(dict):
    """Dict subclass providing backwards-compatibility when compared as string or dict."""

    def __str__(self) -> str:
        return self.get("gemini", "gemini-3.6-flash")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.get("gemini") or other == self.get("openrouter") or other == self.get("openai")
        return super().__eq__(other)


def load_models_config() -> dict:
    raw = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("transcription"), dict):
            raw["transcription"] = TranscriptionDict(raw["transcription"])
        if isinstance(raw.get("defaults", {}).get("transcription"), dict):
            raw["defaults"]["transcription"] = TranscriptionDefaultDict(raw["defaults"]["transcription"])
    return raw


# Wrap httpx.Response.json for TestClient /api/config responses to preserve TranscriptionDict shims
_orig_response_json = httpx.Response.json


def _shimmed_response_json(self, **kwargs):
    data = _orig_response_json(self, **kwargs)
    if isinstance(data, dict) and isinstance(data.get("models"), dict):
        models = data["models"]
        if isinstance(models.get("transcription"), dict) and not isinstance(models["transcription"], TranscriptionDict):
            models["transcription"] = TranscriptionDict(models["transcription"])
        if isinstance(models.get("defaults", {}).get("transcription"), dict) and not isinstance(
            models["defaults"]["transcription"], TranscriptionDefaultDict
        ):
            models["defaults"]["transcription"] = TranscriptionDefaultDict(models["defaults"]["transcription"])
    return data


httpx.Response.json = _shimmed_response_json


def _clean_cookies_value(raw: str | None) -> str | None:
    """python-dotenv keeps the comment as the value for lines like
    `YTDLP_COOKIES_FILE=   # komentar`; treat such values as unset."""
    value = (raw or "").strip()
    if not value or value.startswith("#"):
        return None
    return value


_active_gemini_label: str = "Default"


def get_all_gemini_keys() -> dict[str, str]:
    """Scan env for GEMINI_API_KEY (label 'Default') and GEMINI_API_KEY_<LABEL>.
    Returns {label: key_value} for every non-empty key found."""
    keys: dict[str, str] = {}
    default = (os.getenv("GEMINI_API_KEY") or "").strip()
    if default:
        keys["Default"] = default
    for var, value in os.environ.items():
        if var.startswith("GEMINI_API_KEY_"):
            label = var[len("GEMINI_API_KEY_"):]
            stripped = value.strip()
            if stripped:
                keys[label] = stripped
    return keys


def get_active_gemini_label() -> str:
    return _active_gemini_label


def set_active_gemini_key(label: str) -> None:
    global _active_gemini_label
    available = get_all_gemini_keys()
    if label not in available:
        raise ConfigError(
            f"Label Gemini key '{label}' tidak ditemukan. "
            f"Tambahkan GEMINI_API_KEY_{label}=... ke .env."
        )
    _active_gemini_label = label


def resolve_gemini_key(key_label: str | None = None, *, fallback_key: str | None = None) -> str:
    """Resolve a Gemini API key by label without mutating global active state."""
    label = (key_label or "").strip()
    if not label:
        key = (fallback_key or "").strip()
        if key:
            return key
        key = get_all_gemini_keys().get(_active_gemini_label, "")
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY belum diisi. Salin .env.example menjadi .env "
                "lalu isi GEMINI_API_KEY."
            )
        return key
    all_keys = get_all_gemini_keys()
    if label not in all_keys:
        if label == "Default":
            raise ConfigError("GEMINI_API_KEY belum diisi di .env.")
        raise ConfigError(
            f"Label Gemini key '{label}' tidak ditemukan. "
            f"Tambahkan GEMINI_API_KEY_{label}=... ke .env."
        )
    return all_keys[label]


@dataclass
class Config:
    gemini_api_key: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    ninerouter_api_key: str | None
    ninerouter_base_url: str
    openrouter_api_key: str | None
    openrouter_base_url: str
    transcribe_model: str
    translate_models: dict[str, str]
    ytdlp_cookies_file: str | None
    sub_min_duration: float
    sub_max_duration: float
    sub_merge_gap: float
    sub_cps_flag: float
    chunk_min_seconds: float
    chunk_max_seconds: float
    timing_model: str
    qc_window_size: int
    qc_batch_size: int
    qc_model: str

    @classmethod
    def load(cls) -> "Config":
        load_dotenv(override=False)
        all_keys = get_all_gemini_keys()
        gemini_key = all_keys.get(_active_gemini_label, "")
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
        anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip() or None
        ninerouter_api_key = (os.getenv("NINEROUTER_API_KEY") or "").strip() or None
        openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None
        if not gemini_key and not ninerouter_api_key and not openrouter_api_key and not openai_key and not anthropic_key:
            raise ConfigError(
                "Tidak ada API key yang dikonfigurasi. "
                "Isi setidaknya satu kunci API di .env: "
                "GEMINI_API_KEY, NINEROUTER_API_KEY, OPENROUTER_API_KEY, "
                "OPENAI_API_KEY, atau ANTHROPIC_API_KEY."
            )
        try:
            _models = load_models_config()
            _defaults = _models["defaults"]
            _trans_defaults = _defaults["translation"]
            default_transcribe = (
                _defaults["transcription"].get("gemini", "gemini-3.6-flash")
                if isinstance(_defaults["transcription"], dict)
                else _defaults["transcription"]
            )
            default_translate = {
                "gemini": _trans_defaults["gemini"],
                "openai": _trans_defaults["openai"],
                "anthropic": _trans_defaults["anthropic"],
            }
            default_timing = _defaults.get("timing", "gemini-2.5-flash")
            default_qc = _defaults.get("qc", "gemini-2.5-flash")
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ConfigError(
                f"Konfigurasi model tidak valid atau gagal dimuat ({MODELS_CONFIG_PATH}): {exc}"
            ) from exc
        return cls(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            ninerouter_api_key=ninerouter_api_key,
            ninerouter_base_url=(os.getenv("NINEROUTER_BASE_URL") or "http://localhost:20128/v1").strip(),
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
            transcribe_model=(os.getenv("TRANSCRIBE_MODEL") or "").strip() or default_transcribe,
            translate_models={
                "gemini": (os.getenv("TRANSLATE_MODEL_GEMINI") or "").strip() or default_translate["gemini"],
                "openai": (os.getenv("TRANSLATE_MODEL_OPENAI") or "").strip() or default_translate["openai"],
                "anthropic": (os.getenv("TRANSLATE_MODEL_ANTHROPIC") or "").strip() or default_translate["anthropic"],
            },
            ytdlp_cookies_file=_clean_cookies_value(os.getenv("YTDLP_COOKIES_FILE")),
            sub_min_duration=float(os.getenv("SUB_MIN_DURATION", "0.7")),
            sub_max_duration=float(os.getenv("SUB_MAX_DURATION", "7.5")),
            sub_merge_gap=float(os.getenv("SUB_MERGE_GAP", "0.5")),
            sub_cps_flag=float(os.getenv("SUB_CPS_FLAG", "25")),
            chunk_min_seconds=float(os.getenv("CHUNK_MIN_MINUTES", "4")) * 60,
            chunk_max_seconds=float(os.getenv("CHUNK_MAX_MINUTES", "6")) * 60,
            timing_model=(os.getenv("TIMING_MODEL") or "").strip() or default_timing,
            qc_window_size=int(os.getenv("QC_WINDOW_SIZE", "10")),
            qc_batch_size=int(os.getenv("QC_BATCH_SIZE", "20")),
            qc_model=(os.getenv("QC_MODEL") or "").strip() or default_qc,
        )

    def available_providers(self) -> dict[str, bool]:
        return {
            "9router": bool(self.ninerouter_api_key),
            "openrouter": bool(self.openrouter_api_key),
            "gemini": bool(self.gemini_api_key),
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
        }

    def available_audio_providers(self) -> dict[str, bool]:
        return {
            "gemini": bool(self.gemini_api_key),
            "openrouter": bool(self.openrouter_api_key),
            "openai": bool(self.openai_api_key),
        }


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FALLBACK_TRIGGER_STATUS = RETRYABLE_STATUS | {403}
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
    # Gemini 3.x (preview/lite — use 2.5-flash pricing as estimate until official pricing published)
    "gemini-3.8-flash": (0.825, 4.125),
    "gemini-3.5-live-translate-preview": (0.30, 2.50),
    "gemini-3.1-pro-preview": (3.50, 10.50),
    "gemini-3.7-flash": (1.50, 7.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    # Gemini 2.5
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # OpenRouter / Other models
    "openai/whisper-large-v3-turbo": (0.10, 0.10),
    "deepseek/deepseek-v4.1-flash-1017": (0.14, 0.28),
    "qwen/qwen3.8-omni-flash": (0.15, 0.47),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-3.8-flash": (0.825, 4.125),
    "openai/gpt-audio-mini": (0.60, 2.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "xiaomi/mimo-v2.6-flash": (0.14, 0.28),
    "deepseek/deepseek-chat": (0.14, 0.28),
    "anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "qwen/qwen-2.5-72b-instruct": (0.35, 0.40),
    "openai/gpt-6.1-sol": (2.50, 10.00),
    "openai/gpt-6.1-terra": (2.50, 10.00),
    "openai/gpt-6-luna": (2.50, 10.00),
    "openai/gpt-6-astra": (2.50, 10.00),
    "openai/gpt-6.1-luna": (2.50, 10.00),
    "openai/gpt-6.1-astra": (2.50, 10.00),
    # OpenAI
    "gpt-transcribe": (0.10, 0.10),
    "whisper-1": (0.10, 0.10),
    "whisper-large-v3-turbo": (0.10, 0.10),
    "gpt-6-astra": (2.50, 10.00),
    "gpt-6.1-astra": (2.50, 10.00),
    "gpt-6.1-sol": (2.50, 10.00),
    "gpt-6.1-terra": (2.50, 10.00),
    "gpt-6-luna": (2.50, 10.00),
    "gpt-6.1-luna": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic
    "claude-opus-4-8": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
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


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


def _from_gemini(resp) -> LLMResponse:
    usage = getattr(resp, "usage_metadata", None)
    finish = ""
    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        finish = getattr(reason, "name", None) or str(reason or "")
    return LLMResponse(
        text=resp.text or "",
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        truncated="MAX_TOKENS" in finish.upper(),
    )


def _from_openai(resp) -> LLMResponse:
    choice = resp.choices[0]
    return LLMResponse(
        text=choice.message.content or "",
        input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        truncated=choice.finish_reason == "length",
    )


def _from_anthropic(resp) -> LLMResponse:
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    return LLMResponse(
        text=text,
        input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
        truncated=resp.stop_reason == "max_tokens",
    )


class GeminiClient:
    """Transcription (audio via Files API) and translation."""

    def __init__(self, api_key: str, model: str):
        from google import genai  # lazy: keeps import cost out of tests
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def upload_audio(self, path: Path):
        """Upload via Files API and wait until ACTIVE (audio gets processed)."""
        file = self.client.files.upload(file=str(path))
        deadline = time.time() + 300
        while getattr(file.state, "name", str(file.state)) == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError(f"Files API stuck PROCESSING: {path.name}")
            time.sleep(2)
            file = self.client.files.get(name=file.name)
        state = getattr(file.state, "name", str(file.state))
        if state != "ACTIVE":
            raise RuntimeError(f"Files API upload {path.name} state={state}")
        return file

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        from google.genai import types
        contents = [audio, user] if audio is not None else [user]
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.3,
            ),
        )
        return _from_gemini(resp)


def _supports_custom_temperature(model: str) -> bool:
    """Return False if model requires default temperature (1.0) or rejects custom temperature."""
    m = model.lower()
    base = m.split("/")[-1]
    unsupported_patterns = (
        "o1", "o3", "o4", "gpt-5", "gpt-6",
        "terra", "sol", "luna", "astra",
    )
    for p in unsupported_patterns:
        if base.startswith(p) or f"-{p}" in base or f"_{p}" in base or f".{p}" in base:
            return False
    return True


def _is_temperature_unsupported_error(exc: Exception) -> bool:
    """Check if exception is caused by an unsupported temperature parameter value."""
    err_str = str(exc).lower()
    if "temperature" in err_str and (
        "unsupported" in err_str
        or "default (1)" in err_str
        or "default" in err_str
        or "does not support" in err_str
        or "invalid" in err_str
    ):
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            if err.get("param") == "temperature" or err.get("code") == "unsupported_value":
                return True
    return False


def _is_transcription_model(model: str) -> bool:
    """Check if model is an audio transcription (STT) model requiring /audio/transcriptions."""
    m = model.lower()
    return "whisper" in m or "transcribe" in m


def _is_transcription_endpoint_error(exc: Exception) -> bool:
    """Check if model error says to use audio/transcriptions endpoint."""
    err_str = str(exc).lower()
    return "transcription model" in err_str or "audio/transcriptions" in err_str


def _format_transcription_response(data: dict) -> LLMResponse:
    """Convert verbose_json transcription response into LLMResponse with JSON array for Utterance parsing."""
    items = []
    raw_segments = data.get("segments") or []
    if raw_segments:
        for idx, seg in enumerate(raw_segments, start=1):
            if isinstance(seg, dict):
                text = (seg.get("text") or seg.get("ja") or "").strip()
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
            else:
                text = (getattr(seg, "text", "") or getattr(seg, "ja", "") or "").strip()
                start = float(getattr(seg, "start", 0.0))
                end = float(getattr(seg, "end", 0.0))
            if not text:
                continue
            if end <= start:
                end = start + 0.5
            items.append({
                "id": idx,
                "start": start,
                "end": end,
                "type": "dialogue",
                "ja": text,
            })
    elif data.get("text"):
        raw_text = str(data["text"]).strip()
        sentences = [s.strip() for s in re.split(r"(?<=[。！？\n])", raw_text) if s.strip()]
        if not sentences:
            sentences = [raw_text]
        dur = float(data.get("duration", 0.0) or (len(sentences) * 3.0))
        dur = max(dur, len(sentences) * 1.5)
        step = dur / len(sentences)
        for idx, s in enumerate(sentences, start=1):
            s_start = round((idx - 1) * step, 2)
            s_end = round(min(idx * step, dur), 2)
            if s_end <= s_start:
                s_end = s_start + 0.5
            items.append({
                "id": idx,
                "start": s_start,
                "end": s_end,
                "type": "dialogue",
                "ja": s,
            })

    json_text = json.dumps(items, ensure_ascii=False)
    usage = data.get("usage") or {}
    in_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    if not in_tokens and not out_tokens and "seconds" in usage:
        sec = float(usage.get("seconds") or 0.0)
        in_tokens = int(sec * 10)
        out_tokens = int(sec * 5)
    return LLMResponse(
        text=json_text,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        truncated=False,
    )


def _extract_transcription_context(system: str) -> tuple[str, list[str]]:
    """Extract a concise prompt and clean keywords (no <, >, newlines) from system context."""
    if not system:
        return "", []
    keywords: list[str] = []
    for line in system.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells and cells[0] and not cells[0].startswith("---"):
                term = cells[0].replace("<", "").replace(">", "").strip()
                if term and term not in keywords and len(term) < 50:
                    keywords.append(term)
            continue
        if line.startswith("-"):
            clean = re.sub(r"[*_`]", "", line.lstrip("- ")).replace("<", "").replace(">", "").strip()
            clean_term = clean.split(":")[0].split("—")[0].split("(")[0].strip()
            if clean_term and clean_term not in keywords and len(clean_term) < 50:
                keywords.append(clean_term)
    keywords = keywords[:50]
    prompt = ""
    if keywords:
        prompt = f"Japanese variety show transcription. Key terms: {', '.join(keywords[:15])}"
    return prompt, keywords


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, max_retries=0)  # we retry ourselves
        self.model = model

    def upload_audio(self, path: Path):
        return path

    def _call_transcriptions_endpoint(self, audio_path: Path, system: str = "") -> LLMResponse:
        audio_path = Path(audio_path)
        model = self.model
        if model.startswith("openai/"):
            model = model.split("/", 1)[1]
        prompt, keywords = _extract_transcription_context(system)

        call_kwargs: dict = {
            "model": model,
        }
        if model == "gpt-transcribe":
            call_kwargs["response_format"] = "json"
            if prompt:
                call_kwargs["prompt"] = prompt
            if keywords:
                call_kwargs["extra_body"] = {"keywords": keywords, "languages": ["ja"]}
        else:
            call_kwargs["response_format"] = "verbose_json"
            call_kwargs["language"] = "ja"
            call_kwargs["timestamp_granularities"] = ["segment"]
            if prompt:
                call_kwargs["prompt"] = prompt[:200]

        try:
            with open(audio_path, "rb") as f:
                raw = self.client.audio.transcriptions.with_raw_response.create(
                    file=f,
                    **call_kwargs,
                )
            data = json.loads(raw.text)
            return _format_transcription_response(data)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "verbose_json" in err_msg and call_kwargs.get("response_format") == "verbose_json":
                call_kwargs["response_format"] = "json"
                call_kwargs.pop("timestamp_granularities", None)
                with open(audio_path, "rb") as f:
                    raw = self.client.audio.transcriptions.with_raw_response.create(
                        file=f,
                        **call_kwargs,
                    )
                data = json.loads(raw.text)
                return _format_transcription_response(data)
            if "extra_body" in str(exc) or "keywords" in err_msg or "languages" in err_msg:
                call_kwargs.pop("extra_body", None)
                with open(audio_path, "rb") as f:
                    raw = self.client.audio.transcriptions.with_raw_response.create(
                        file=f,
                        **call_kwargs,
                    )
                data = json.loads(raw.text)
                return _format_transcription_response(data)
            if (
                ("model" in err_msg and ("not exist" in err_msg or "not found" in err_msg or "does not have access" in err_msg or "invalid" in err_msg))
                and model != "gpt-transcribe"
                and model != "whisper-1"
            ):
                fb_model = "whisper-1" if "whisper" in model.lower() else "gpt-transcribe"
                fb_format = "json" if fb_model == "gpt-transcribe" else "verbose_json"
                with open(audio_path, "rb") as f:
                    raw = self.client.audio.transcriptions.with_raw_response.create(
                        model=fb_model,
                        file=f,
                        response_format=fb_format,
                    )
                data = json.loads(raw.text)
                return _format_transcription_response(data)
            raise

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        if audio is not None and _is_transcription_model(self.model):
            return self._call_transcriptions_endpoint(Path(audio), system=system)

        kwargs = {}
        if _supports_custom_temperature(self.model):
            kwargs["temperature"] = 0.3

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                **kwargs,
            )
            return _from_openai(resp)
        except Exception as exc:
            if audio is not None and _is_transcription_endpoint_error(exc):
                return self._call_transcriptions_endpoint(Path(audio))
            if "temperature" in kwargs and _is_temperature_unsupported_error(exc):
                kwargs.pop("temperature", None)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                return _from_openai(resp)
            raise


class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self.model = model

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=16384,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,
        )
        return _from_anthropic(resp)


class NineRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:20128/v1"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model = model

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        kwargs = {}
        if _supports_custom_temperature(self.model):
            kwargs["temperature"] = 0.3
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                **kwargs,
            )
            return _from_openai(resp)
        except Exception as exc:
            if "temperature" in kwargs and _is_temperature_unsupported_error(exc):
                kwargs.pop("temperature", None)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    **kwargs,
                )
                return _from_openai(resp)
            raise


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "SubtitledByAI",
            },
        )
        self.model = model

    def upload_audio(self, path: Path):
        return path

    def _call_transcriptions_endpoint(self, audio_path: Path) -> LLMResponse:
        audio_path = Path(audio_path)
        try:
            with open(audio_path, "rb") as f:
                raw = self.client.audio.transcriptions.with_raw_response.create(
                    model=self.model,
                    file=f,
                    response_format="verbose_json",
                )
            data = json.loads(raw.text)
            return _format_transcription_response(data)
        except Exception as exc:
            status = _status_of(exc)
            if status == 400 or "input_audio" in str(exc).lower():
                b64_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
                fmt = audio_path.suffix.lstrip(".").lower() or "mp3"
                resp = self.client._client.post(
                    f"{str(self.client.base_url).rstrip('/')}/audio/transcriptions",
                    json={
                        "model": self.model,
                        "input_audio": {"data": b64_audio, "format": fmt},
                        "response_format": "verbose_json",
                    },
                    headers={
                        "Authorization": f"Bearer {self.client.api_key}",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "SubtitledByAI",
                    },
                )
                if resp.is_success:
                    return _format_transcription_response(resp.json())
            raise

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        if audio is not None and _is_transcription_model(self.model):
            return self._call_transcriptions_endpoint(Path(audio))

        kwargs = {}
        if _supports_custom_temperature(self.model):
            kwargs["temperature"] = 0.3

        if audio is not None:
            audio_path = Path(audio)
            b64_audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            user_content = [
                {"type": "text", "text": user},
                {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "mp3"}},
            ]
        else:
            user_content = user
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_content}],
                **kwargs,
            )
            return _from_openai(resp)
        except Exception as exc:
            if audio is not None and _is_transcription_endpoint_error(exc):
                return self._call_transcriptions_endpoint(Path(audio))
            if "temperature" in kwargs and _is_temperature_unsupported_error(exc):
                kwargs.pop("temperature", None)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user_content}],
                    **kwargs,
                )
                return _from_openai(resp)
            raise


def make_transcriber(cfg: Config, model: str | None = None) -> GeminiClient:
    return GeminiClient(cfg.gemini_api_key, model or cfg.transcribe_model)


def make_audio_client(
    cfg: Config, provider: str = "gemini", model: str | None = None, key_label: str | None = None
) -> GeminiClient | OpenRouterClient | OpenAIClient:
    if provider == "openrouter":
        if not cfg.openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY belum diisi di .env")
        return OpenRouterClient(cfg.openrouter_api_key, model or "qwen/qwen3.8-omni-flash", cfg.openrouter_base_url)
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY belum diisi di .env")
        return OpenAIClient(cfg.openai_api_key, model or "whisper-1")
    label = (key_label or "").strip() or None
    if label is None:
        return make_transcriber(cfg, model=model)
    api_key = resolve_gemini_key(label, fallback_key=cfg.gemini_api_key)
    return GeminiClient(api_key, model or cfg.transcribe_model)


def make_translator(cfg: Config, provider: str, model: str | None = None):
    if provider == "gemini":
        return GeminiClient(cfg.gemini_api_key, model or cfg.translate_models["gemini"])
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY belum diisi di .env")
        return OpenAIClient(cfg.openai_api_key, model or cfg.translate_models["openai"])
    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY belum diisi di .env")
        return AnthropicClient(cfg.anthropic_api_key, model or cfg.translate_models["anthropic"])
    raise ConfigError(f"Penerjemah tidak dikenal: {provider}")


def make_text_client(
    cfg: Config,
    provider: str,
    model: str | None = None,
    key_label: str | None = None,
) -> GeminiClient | OpenAIClient | AnthropicClient | NineRouterClient | OpenRouterClient:
    if provider == "9router":
        if not cfg.ninerouter_api_key:
            raise ConfigError("NINEROUTER_API_KEY belum diisi di .env")
        return NineRouterClient(cfg.ninerouter_api_key, model or "ag/gemini-3.8-flash-high", cfg.ninerouter_base_url)
    if provider == "openrouter":
        if not cfg.openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY belum diisi di .env")
        return OpenRouterClient(cfg.openrouter_api_key, model or "deepseek/deepseek-chat", cfg.openrouter_base_url)
    label = (key_label or "").strip() or None
    if provider == "gemini" and label is not None:
        api_key = resolve_gemini_key(label, fallback_key=cfg.gemini_api_key)
        return GeminiClient(api_key, model or cfg.translate_models["gemini"])
    return make_translator(cfg, provider, model=model)


@dataclass(frozen=True)
class AudioFallbackSlot:
    key_label: str
    model: str
    provider: str = "gemini"


@dataclass(frozen=True)
class TranslateFallbackSlot:
    provider: str
    model: str
    key_label: str | None = None


def _is_fallback_trigger(exc: Exception) -> bool:
    status = _status_of(exc)
    return status in FALLBACK_TRIGGER_STATUS or isinstance(exc, NETWORK_ERRORS)


def _execute_slot_chain(
    *,
    primary,
    fallback_slots: list,
    build_client,
    invoke_slot,
    attempts_per_slot: int,
    base_delay: float,
    on_fallback,
    sleep,
):
    if attempts_per_slot < 1:
        raise ValueError("attempts_per_slot must be >= 1")
    chain = [primary, *itertools.islice(fallback_slots or (), 3)]
    last_exc: Exception | None = None
    for idx, slot in enumerate(chain):
        if idx > 0 and on_fallback is not None:
            on_fallback(idx, slot, last_exc)
        try:
            client = build_client(slot)
            result = call_with_retries(
                lambda c=client, s=slot: invoke_slot(c, s),
                attempts=attempts_per_slot,
                base_delay=base_delay,
                sleep=sleep,
            )
            return result, client
        except Exception as exc:
            last_exc = exc
            if not _is_fallback_trigger(exc) or idx == len(chain) - 1:
                raise
    assert last_exc is not None
    raise last_exc


def execute_with_fallback(
    *,
    primary,
    fallback_slots: list,
    build_client,
    call_fn,
    attempts_per_slot: int = 2,
    base_delay: float = 2.0,
    on_fallback=None,
    sleep=time.sleep,
):
    """Run call_fn on primary, stepping through up to 3 fallback slots on busy/network errors.

    Returns (result, winning_client). Raises immediately on non-fallback errors (e.g. 400),
    or re-raises the final error once all slots are exhausted.
    """
    return _execute_slot_chain(
        primary=primary,
        fallback_slots=fallback_slots,
        build_client=build_client,
        invoke_slot=lambda client, _slot: call_fn(client),
        attempts_per_slot=attempts_per_slot,
        base_delay=base_delay,
        on_fallback=on_fallback,
        sleep=sleep,
    )


def execute_audio_with_fallback(
    *,
    chunk_path: Path,
    primary: AudioFallbackSlot,
    fallback_slots: list[AudioFallbackSlot],
    build_client,
    resolve_key,
    call_fn,
    attempts_per_slot: int = 2,
    base_delay: float = 2.0,
    on_upload_done=None,
    on_fallback=None,
    sleep=time.sleep,
):
    """Like execute_with_fallback, caching Gemini Files API uploads per resolved API key."""
    upload_cache: dict[str, object] = {}

    def _run_for_client(client, slot: AudioFallbackSlot):
        api_key = resolve_key(slot)
        if api_key not in upload_cache:
            ref = client.upload_audio(chunk_path)
            if ref is None:
                raise RuntimeError(f"upload_audio mengembalikan None untuk {chunk_path.name}")
            upload_cache[api_key] = ref
            if on_upload_done is not None:
                on_upload_done()
        return call_fn(client, upload_cache[api_key])

    return _execute_slot_chain(
        primary=primary,
        fallback_slots=fallback_slots,
        build_client=build_client,
        invoke_slot=_run_for_client,
        attempts_per_slot=attempts_per_slot,
        base_delay=base_delay,
        on_fallback=on_fallback,
        sleep=sleep,
    )
