# Phase 2 Design: 9router & OpenRouter Multi-Provider Architecture

**Date:** 2026-11-10  
**Scope:** Phase 2 of the major system update — adding **9router** (local OpenAI-compatible proxy at `http://localhost:20128/v1` for text translation & QC Pass 1B with live `/v1/models` discovery) and **OpenRouter** (curated multimodal audio models + curated text models + custom model ID input).

---

## 1. Architecture Overview & Provider Separation

The system divides pipeline stages strictly into two categories:

```
[Audio Stages]                                 [Text Stages]
- transcribe                                   - translate
- timing                                       - qc (Pass 1B Whole-Episode Text QC)
- qc (Pass 2 Targeted/Deep Audio QC)           

Supported Audio Providers:                     Supported Text Providers:
1. gemini (Google Files API)                   1. 9router (Local proxy via OpenAI client)
2. openrouter (base64 input_audio)             2. openrouter (OpenAI-compatible client)
                                               3. gemini (Native GeminiClient)
                                               4. openai (Native OpenAIClient)
                                               5. anthropic (Native AnthropicClient)
```

### Key Principles
1. **Inheritance by Stage Type:**
   - On the main UI form, the user selects:
     - **Audio Provider** (`gemini` | `openrouter`) + **Transcribe Model**
     - **Text Provider** (`9router` | `openrouter` | `gemini` | `openai` | `anthropic`) + **Translate Model**
   - `timing` and Pass 2 Audio QC automatically inherit the selected **Audio Provider + Model**.
   - Pass 1B Text QC automatically inherits the selected **Text Provider + Model**.
   - Redundant `timing_model` and `qc_model` dropdowns are removed from the main form layout.
2. **Stage-Aware Retry Control:**
   - In the event of a failure, the `#retry-box` dynamically displays controls matching the failed stage:
     - `transcribe` or `timing` failed: allows changing **Audio Provider**, Audio Key, and Audio Model.
     - `translate` or `qc` (Pass 1B) failed: allows changing **Text Provider**, Text Key, and Text Model.
3. **Optional API Keys:**
   - `GEMINI_API_KEY` is no longer globally mandatory for server boot.
   - At job submission and retry, the server verifies only the API key required for the chosen provider(s):
     - `transcription_only`: requires Audio Provider API key only.
     - `translation_only`: requires Text Provider API key only (e.g. `NINEROUTER_API_KEY`).
     - `full`: requires both Audio Provider and Text Provider API keys.

---

## 2. Configuration & Provider Clients (`app/providers.py`)

### 2.1 Environment Variables
Add to `.env` / `.env.example`:
```env
# 9router (Local AI Gateway)
NINEROUTER_API_KEY=
NINEROUTER_BASE_URL=http://localhost:20128/v1

# OpenRouter (Cloud AI Gateway)
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 2.2 Client Implementation
Both `9router` and `OpenRouter` use the `openai.OpenAI` SDK with custom `base_url`:

1. **`NineRouterClient` (Text Only):**
   ```python
   class NineRouterClient:
       def __init__(self, api_key: str, model: str, base_url: str):
           from openai import OpenAI
           self.client = OpenAI(api_key=api_key or "dummy", base_url=base_url, max_retries=0)
           self.model = model

       def generate(self, system: str, user: str, audio=None) -> LLMResponse:
           # Standard chat completions with JSON mode
           resp = self.client.chat.completions.create(
               model=self.model,
               messages=[
                   {"role": "system", "content": system},
                   {"role": "user", "content": user},
               ],
               response_format={"type": "json_object"},
               temperature=0.3,
           )
           return _from_openai(resp)
   ```

2. **`OpenRouterClient` (Text & Multimodal Audio):**
   ```python
   class OpenRouterClient:
       def __init__(self, api_key: str, model: str, base_url: str):
           from openai import OpenAI
           self.client = OpenAI(
               api_key=api_key,
               base_url=base_url,
               max_retries=0,
               default_headers={"HTTP-Referer": "https://github.com/SubtitledByAI", "X-Title": "SubtitledByAI"},
           )
           self.model = model

       def upload_audio(self, path: Path):
           # OpenRouter has no Files API; return the local Path directly
           return path

       def generate(self, system: str, user: str, audio=None) -> LLMResponse:
           messages = [{"role": "system", "content": system}]
           if audio is not None:
               import base64
               audio_path = Path(audio)
               b64_audio = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
               content = [
                   {"type": "text", "text": user},
                   {"type": "input_audio", "input_audio": {"data": b64_audio, "format": "mp3"}},
               ]
               messages.append({"role": "user", "content": content})
           else:
               messages.append({"role": "user", "content": user})

           resp = self.client.chat.completions.create(
               model=self.model,
               messages=messages,
               response_format={"type": "json_object"},
               temperature=0.3,
           )
           return _from_openai(resp)
   ```

3. **Updated Factory Seams:**
   - `make_audio_client(cfg: Config, provider: str = "gemini", model: str | None = None, key_label: str | None = None)`:
     - `provider == "gemini"`: returns `GeminiClient` with key isolation.
     - `provider == "openrouter"`: returns `OpenRouterClient` using `cfg.openrouter_api_key`.
   - `make_text_client(cfg: Config, provider: str, model: str | None = None, key_label: str | None = None)`:
     - `provider == "9router"`: returns `NineRouterClient`.
     - `provider == "openrouter"`: returns `OpenRouterClient`.
     - `provider in ("gemini", "openai", "anthropic")`: returns existing clients.

---

## 3. Model Catalog & Live 9router Discovery (`config/models.json`, `app/main.py`)

### 3.1 Catalog Layout (`config/models.json`)
```json
{
  "transcription": {
    "gemini": [
      "gemini-3.8-flash",
      "gemini-3.7-flash",
      "gemini-3.6-flash",
      "gemini-3.5-flash",
      "gemini-3.1-pro-preview",
      "gemini-2.5-flash",
      "gemini-2.5-pro"
    ],
    "openrouter": [
      "qwen/qwen3.8-omni-flash",
      "google/gemini-2.5-flash-lite",
      "google/gemini-2.5-flash",
      "google/gemini-3.8-flash",
      "openai/gpt-audio-mini",
      "xiaomi/mimo-v2.6-flash"
    ]
  },
  "translation": {
    "9router": [
      "ag/gemini-3.8-flash-high",
      "subtitle-gen-combo",
      "gc/gemini-2.5-flash",
      "gc/gemini-3-flash-preview"
    ],
    "openrouter": [
      "deepseek/deepseek-chat",
      "anthropic/claude-3.7-sonnet",
      "google/gemini-2.5-flash",
      "openai/gpt-4o-mini",
      "qwen/qwen-2.5-72b-instruct"
    ],
    "gemini": [
      "gemini-3.8-flash",
      "gemini-3.7-flash",
      "gemini-3.6-flash",
      "gemini-3.5-flash",
      "gemini-2.5-flash"
    ],
    "openai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
  },
  "defaults": {
    "audio_provider": "gemini",
    "transcription": {
      "gemini": "gemini-3.6-flash",
      "openrouter": "qwen/qwen3.8-omni-flash"
    },
    "text_provider": "9router",
    "translation": {
      "9router": "ag/gemini-3.8-flash-high",
      "openrouter": "deepseek/deepseek-chat",
      "gemini": "gemini-3.6-flash",
      "openai": "gpt-4o",
      "anthropic": "claude-sonnet-4-6"
    }
  }
}
```

### 3.2 Live Discovery for 9router (`app/main.py`)
In `GET /api/config`:
```python
@app.get("/api/config")
def get_config() -> dict:
    ...
    models_cfg = providers.load_models_config()
    # Query 9router live models if configured
    if cfg.ninerouter_api_key and cfg.ninerouter_base_url:
        live_models = fetch_9router_models(cfg.ninerouter_base_url, cfg.ninerouter_api_key)
        if live_models:
            models_cfg["translation"]["9router"] = live_models
    ...
```
`fetch_9router_models` uses `httpx.get(f"{base_url}/models", timeout=2.0)`. If 9router is offline or times out, it gracefully falls back to the static list in `config/models.json`.

---

## 4. UI & Form Layout (`static/index.html`)

### 4.1 Symmetric Dropdowns
1. **Audio Stage Section:**
   - `<select id="audio_provider">`: options `Gemini`, `OpenRouter`.
   - `<select id="transcribe_model">`: updates dynamically based on `audio_provider`.
   - If `openrouter` is chosen, the list contains the curated audio models plus `[Ketik model lain...]`.
   - Selecting `[Ketik model lain...]` reveals `<input type="text" id="custom_transcribe_model">`.
2. **Text Stage Section:**
   - `<select id="translator">`: options `9router`, `OpenRouter`, `Gemini`, `OpenAI`, `Anthropic`.
   - `<select id="translate_model">`: updates dynamically based on `translator`. For `9router`, populated with live models. For `openrouter`, includes `[Ketik model lain...]` revealing `<input type="text" id="custom_translate_model">`.

### 4.2 Fallback Chains Setup
- **Audio Fallback Slots (1–3):**
  - Dropdowns: `Audio Provider` (`Gemini` | `OpenRouter`), then `API Key` (visible if Gemini), and `Model` (curated + custom).
- **Translation Fallback Slots (1–3):**
  - Dropdowns: `Provider` (`9router` | `OpenRouter` | `Gemini` | `OpenAI` | `Anthropic`), `API Key` (if Gemini), and `Model`.

---

## 5. Pipeline Orchestration & Checkpoints (`app/pipeline.py`)

1. **Parameters Stored in `job.params`:**
   - `audio_provider`: `str` (`"gemini"` or `"openrouter"`, default `"gemini"`)
   - `transcribe_model`: `str`
   - `translator`: `str` (`"9router"`, `"openrouter"`, `"gemini"`, `"openai"`, `"anthropic"`)
   - `translate_model`: `str`
   - `audio_fallback`: `list[dict]` (slot schema: `{"provider": str, "key_label": str | None, "model": str}`)
   - `translate_fallback`: `list[dict]`
2. **Execution Mapping:**
   - `transcribe`: uses `audio_provider` + `transcribe_model`.
   - `translate`: uses `translator` + `translate_model`.
   - `timing`: uses `audio_provider` + `job.params.get("timing_model") or transcribe_model`.
   - `qc`:
     - Pass 1B (Text): uses `translator` + `translate_model`.
     - Pass 2 (Audio): uses `audio_provider` + `job.params.get("qc_model") or transcribe_model`.
3. **Audio Upload Caching with OpenRouter:**
   - In `execute_audio_with_fallback`, `OpenRouterClient.upload_audio(path)` returns `path`.
   - Audio chunks are base64-encoded on demand when dispatched to OpenRouter without making any Google Files API calls.

---

## 6. Verification & Test Plan

1. **`tests/test_providers.py`:**
   - Test `NineRouterClient` request format and JSON response unwrapping.
   - Test `OpenRouterClient` text request and base64 audio request creation.
   - Test `Config.load()` without `GEMINI_API_KEY` (passes when other keys are present).
   - Test `make_audio_client` and `make_text_client` for `"9router"` and `"openrouter"`.
2. **`tests/test_api.py`:**
   - Test live 9router model discovery mock in `GET /api/config` (and fallback on timeout).
   - Test `POST /api/jobs` validation: checks chosen provider's key presence in `.env`; allows custom OpenRouter model string.
   - Test `POST /api/jobs/{id}/retry` overriding `audio_provider` and `translator`.
3. **`tests/test_pipeline.py`:**
   - Test end-to-end inheritance: `timing` and Pass 2 `qc` inherit `audio_provider`.
   - Test `translation_only` runs with `translator="9router"` without requiring any audio keys.
   - Full test suite passes offline with 0 network calls.
