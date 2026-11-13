# Phase 2: 9router & OpenRouter Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate **9router** (local gateway for text translation and QC Pass 1B with live `/v1/models` auto-discovery) and **OpenRouter** (curated multimodal audio models + curated text models + custom model ID inputs), relax the mandatory `GEMINI_API_KEY` requirement at boot, and implement symmetric Audio Provider and Text Provider selection across the application.

**Architecture:** `app/providers.py` adds `NineRouterClient` and `OpenRouterClient` (with base64 `input_audio` payload formatting for audio chunk calls). `Config.load()` no longer crashes if `GEMINI_API_KEY` is missing, reporting granular provider availability. `config/models.json` partitions transcription by audio provider (`gemini`, `openrouter`) and translation by text provider (`9router`, `openrouter`, `gemini`, `openai`, `anthropic`). `app/main.py` fetches live 9router models on `/api/config`, validates keys on demand per selected provider, and `app/pipeline.py` wires inheritance so `timing` and Pass 2 Audio QC automatically inherit the Audio Provider and model, while Pass 1B Text QC inherits the Text Provider and model.

**Tech Stack:** Python 3.11+, FastAPI, `openai`, `httpx`, vanilla HTML/JS (`static/index.html`), `pytest`.

---

## Global Constraints & File Map

- **Spec reference:** `docs/superpowers/specs/2026-11-10-phase2-9router-openrouter-design.md`
- **Isolation:** `tests/conftest.py` continues to isolate tests from real `.env`. All OpenRouter and 9router calls in tests must use offline mocks/fakes.
- **Backward compatibility:** Existing jobs with `audio_fallback`, `translate_fallback`, or legacy `{id: text}` checkpoints remain functional. When `audio_provider` is omitted, it defaults to `"gemini"`.
- **Full test suite command:** `python -m pytest -q` (currently 397 tests passing).

**Files touched:**
- Modify: `config/models.json` (partition transcription & translation models by provider, add OpenRouter audio & text curated models, add 9router models)
- Modify: `app/providers.py` (`NineRouterClient`, `OpenRouterClient`, `Config.load` optional Gemini key, `make_audio_client`, `make_text_client`, updated `PRICING`)
- Modify: `app/main.py` (`fetch_9router_models`, dynamic live 9router discovery in `/api/config`, on-demand key validation in `create_job` and `retry`, custom OpenRouter model string support)
- Modify: `app/pipeline.py` (`audio_provider` param, pipeline stage inheritance for `timing` and `qc`, OpenRouter audio runner integration)
- Modify: `static/index.html` (symmetric Audio Provider + Model, Text Provider + Model, OpenRouter custom model input, 9router live model populator, removed main-form Timing & QC dropdowns)
- Modify: `tests/test_providers.py`, `tests/test_api.py`, `tests/test_pipeline.py`

---

### Task 1: Update `config/models.json` & Add Pricing (`config/models.json`, `app/providers.py`, `tests/test_providers.py`)

**Files:**
- Modify: `config/models.json`
- Modify: `app/providers.py:217-245`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
def test_models_config_phase2_structure():
    cfg = load_models_config()
    assert isinstance(cfg["transcription"], dict)
    assert "gemini" in cfg["transcription"]
    assert "openrouter" in cfg["transcription"]
    assert "qwen/qwen3.8-omni-flash" in cfg["transcription"]["openrouter"]

    assert isinstance(cfg["translation"], dict)
    assert "9router" in cfg["translation"]
    assert "openrouter" in cfg["translation"]
    assert "ag/gemini-3.8-flash-high" in cfg["translation"]["9router"]

    assert cfg["defaults"]["audio_provider"] == "gemini"
    assert cfg["defaults"]["text_provider"] == "9router"
    assert cfg["defaults"]["transcription"]["gemini"] in cfg["transcription"]["gemini"]
    assert cfg["defaults"]["transcription"]["openrouter"] in cfg["transcription"]["openrouter"]
    assert cfg["defaults"]["translation"]["9router"] in cfg["translation"]["9router"]


def test_pricing_includes_new_gemini_and_qwen_models():
    from app.providers import PRICING
    assert "gemini-3.8-flash" in PRICING
    assert "qwen/qwen3.8-omni-flash" in PRICING
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers.py -k "phase2_structure or pricing_includes_new" -v`
Expected: FAIL (`AssertionError` because `cfg["transcription"]` is currently a list and models are missing from `PRICING`).

- [ ] **Step 3: Update `config/models.json` and `PRICING` in `app/providers.py`**

Replace `config/models.json` with the partitioned provider catalog:

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
    },
    "timing": "gemini-3.6-flash",
    "qc": "gemini-3.6-flash"
  }
}
```

In `app/providers.py`, add the new models to `PRICING`:

```python
    "gemini-3.8-flash": (0.825, 4.125),
    "gemini-3.5-live-translate-preview": (0.30, 2.50),
    "qwen/qwen3.8-omni-flash": (0.15, 0.47),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-3.8-flash": (0.825, 4.125),
    "openai/gpt-audio-mini": (0.60, 2.40),
    "xiaomi/mimo-v2.6-flash": (0.14, 0.28),
    "deepseek/deepseek-chat": (0.14, 0.28),
    "anthropic/claude-3.7-sonnet": (3.00, 15.00),
    "qwen/qwen-2.5-72b-instruct": (0.35, 0.40),
```

And in `tests/test_providers.py`, update `test_models_config_defaults_match_lists` so it accommodates the dictionary structure:

```python
def test_models_config_defaults_match_lists():
    from app.providers import load_models_config
    cfg = load_models_config()
    assert cfg["defaults"]["transcription"]["gemini"] in cfg["transcription"]["gemini"]
    assert cfg["defaults"]["transcription"]["openrouter"] in cfg["transcription"]["openrouter"]
    assert cfg["defaults"]["translation"]["gemini"] in cfg["translation"]["gemini"]
    assert cfg["defaults"]["translation"]["openai"] in cfg["translation"]["openai"]
    assert cfg["defaults"]["translation"]["anthropic"] in cfg["translation"]["anthropic"]
    assert cfg["defaults"]["translation"]["9router"] in cfg["translation"]["9router"]
    assert cfg["defaults"]["translation"]["openrouter"] in cfg["translation"]["openrouter"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -k "models_config or pricing" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/models.json app/providers.py tests/test_providers.py
git commit -m "feat(config): update models.json with 9router and OpenRouter models and pricing"
```

---

### Task 2: Provider Clients & Optional Gemini Key at Boot (`app/providers.py`, `tests/test_providers.py`)

**Files:**
- Modify: `app/providers.py:80-160, 360-460`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
from app.providers import NineRouterClient, OpenRouterClient


def test_config_loads_without_gemini_key_if_other_providers_set(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("NINEROUTER_API_KEY", "9r-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    cfg = Config.load()
    assert cfg.gemini_api_key == ""
    assert cfg.ninerouter_api_key == "9r-key"
    assert cfg.openrouter_api_key == "or-key"
    assert cfg.available_providers()["gemini"] is False
    assert cfg.available_providers()["9router"] is True
    assert cfg.available_providers()["openrouter"] is True
    assert cfg.available_audio_providers()["gemini"] is False
    assert cfg.available_audio_providers()["openrouter"] is True


def test_ninerouter_client_generates_json_response(monkeypatch):
    client = NineRouterClient(api_key="test-key", model="ag/gemini-3.8-flash-high", base_url="http://localhost:20128/v1")
    fake_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"translated": "halo"}'), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kw: fake_resp)
    resp = client.generate(system="sys", user="user")
    assert resp.text == '{"translated": "halo"}'
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5


def test_openrouter_client_handles_text_and_audio_content(tmp_path, monkeypatch):
    client = OpenRouterClient(api_key="test-key", model="qwen/qwen3.8-omni-flash", base_url="https://openrouter.ai/api/v1")
    audio_file = tmp_path / "chunk_001.mp3"
    audio_file.write_bytes(b"dummy-mp3-bytes")

    called_messages = []
    def fake_create(**kw):
        called_messages.append(kw.get("messages"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='[{"id": 1, "ja": "test"}]'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
        )

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)

    # 1. Test text-only
    resp_text = client.generate(system="sys", user="text prompt")
    assert resp_text.text == '[{"id": 1, "ja": "test"}]'
    assert len(called_messages[0]) == 2
    assert called_messages[0][1]["content"] == "text prompt"

    # 2. Test audio multimodal
    resp_audio = client.generate(system="sys", user="audio prompt", audio=audio_file)
    assert resp_audio.text == '[{"id": 1, "ja": "test"}]'
    user_content = called_messages[1][1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "audio prompt"}
    assert user_content[1]["type"] == "input_audio"
    assert user_content[1]["input_audio"]["format"] == "mp3"
    assert len(user_content[1]["input_audio"]["data"]) > 0


def test_make_audio_client_supports_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = Config.load()
    client = make_audio_client(cfg, provider="openrouter", model="qwen/qwen3.8-omni-flash")
    assert isinstance(client, OpenRouterClient)
    assert client.model == "qwen/qwen3.8-omni-flash"


def test_make_text_client_supports_9router_and_openrouter(monkeypatch):
    monkeypatch.setenv("NINEROUTER_API_KEY", "9r-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = Config.load()
    c_9r = make_text_client(cfg, provider="9router", model="ag/gemini-3.8-flash-high")
    assert isinstance(c_9r, NineRouterClient)
    c_or = make_text_client(cfg, provider="openrouter", model="deepseek/deepseek-chat")
    assert isinstance(c_or, OpenRouterClient)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -k "ninerouter_client or openrouter_client or supports_openrouter" -v`
Expected: FAIL (`ImportError: cannot import name 'NineRouterClient'`).

- [ ] **Step 3: Implement client classes and update `Config` in `app/providers.py`**

In `app/providers.py`:
1. Update `Config` dataclass and `Config.load()`:
   - Add fields:
     - `ninerouter_api_key: str | None`
     - `ninerouter_base_url: str`
     - `openrouter_api_key: str | None`
     - `openrouter_base_url: str`
   - In `Config.load()`:
     - Read `gemini_key = all_keys.get(_active_gemini_label, "")`
     - Read `ninerouter_api_key = (os.getenv("NINEROUTER_API_KEY") or "").strip() or None`
     - Read `ninerouter_base_url = (os.getenv("NINEROUTER_BASE_URL") or "http://localhost:20128/v1").strip()`
     - Read `openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None`
     - Read `openrouter_base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip()`
     - Handle `transcribe_model` fallback: if `_defaults["transcription"]` is a dict, use `_defaults["transcription"].get("gemini", "gemini-3.6-flash")`
     - Update check: Only raise `ConfigError` if NO provider has any API key configured at all (`not gemini_key and not ninerouter_api_key and not openrouter_api_key and not openai_key and not anthropic_key`).
   - Add `available_audio_providers(self) -> dict[str, bool]`:
     ```python
     def available_audio_providers(self) -> dict[str, bool]:
         return {
             "gemini": bool(self.gemini_api_key),
             "openrouter": bool(self.openrouter_api_key),
         }
     ```
   - In `available_providers(self) -> dict[str, bool]`:
     ```python
     def available_providers(self) -> dict[str, bool]:
         return {
             "9router": bool(self.ninerouter_api_key),
             "openrouter": bool(self.openrouter_api_key),
             "gemini": bool(self.gemini_api_key),
             "openai": bool(self.openai_api_key),
             "anthropic": bool(self.anthropic_api_key),
         }
     ```
2. Implement `NineRouterClient` and `OpenRouterClient` (with `upload_audio` returning the path so `execute_audio_with_fallback` works seamlessly).
3. Update `make_audio_client(cfg: Config, provider: str = "gemini", model: str | None = None, key_label: str | None = None)`:
   - If `provider == "openrouter"`:
     - If not `cfg.openrouter_api_key`: raise `ConfigError("OPENROUTER_API_KEY belum diisi di .env")`
     - Return `OpenRouterClient(cfg.openrouter_api_key, model or "qwen/qwen3.8-omni-flash", cfg.openrouter_base_url)`
   - Else (`provider == "gemini"`): existing Gemini logic.
4. Update `make_text_client(cfg: Config, provider: str, model: str | None = None, key_label: str | None = None)`:
   - If `provider == "9router"`:
     - If not `cfg.ninerouter_api_key`: raise `ConfigError("NINEROUTER_API_KEY belum diisi di .env")`
     - Return `NineRouterClient(cfg.ninerouter_api_key, model or "ag/gemini-3.8-flash-high", cfg.ninerouter_base_url)`
   - If `provider == "openrouter"`:
     - If not `cfg.openrouter_api_key`: raise `ConfigError("OPENROUTER_API_KEY belum diisi di .env")`
     - Return `OpenRouterClient(cfg.openrouter_api_key, model or "deepseek/deepseek-chat", cfg.openrouter_base_url)`
   - Else: existing `gemini`, `openai`, `anthropic` logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/providers.py tests/test_providers.py
git commit -m "feat(providers): add NineRouterClient, OpenRouterClient and make Gemini key optional"
```

---

### Task 3: Live 9router Discovery & API Endpoints (`app/main.py`, `tests/test_api.py`)

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_get_config_discovers_live_9router_models(client, monkeypatch):
    monkeypatch.setenv("NINEROUTER_API_KEY", "test-key")
    monkeypatch.setenv("NINEROUTER_BASE_URL", "http://localhost:20128/v1")

    def fake_get(url, **kw):
        class Resp:
            status_code = 200
            def json(self):
                return {"data": [{"id": "model-1"}, {"id": "model-2"}]}
        return Resp()

    monkeypatch.setattr("httpx.get", fake_get)
    body = client.get("/api/config").json()
    assert "9router" in body["providers"]
    assert "model-1" in body["models"]["translation"]["9router"]
    assert "model-2" in body["models"]["translation"]["9router"]


def test_create_job_accepts_audio_provider_and_openrouter_custom_model(client, fake_run, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    resp = client.post(
        "/api/jobs",
        data={
            "source": "url",
            "url": "https://youtu.be/x",
            "audio_provider": "openrouter",
            "transcribe_model": "custom/my-audio-model",
            "translator": "9router",
            "translate_model": "ag/gemini-3.8-flash-high",
            "output_format": "both",
        },
    )
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["audio_provider"] == "openrouter"
    assert job.params["transcribe_model"] == "custom/my-audio-model"
    assert job.params["translator"] == "9router"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py -k "discovers_live_9router or accepts_audio_provider" -v`
Expected: FAIL (`AssertionError` / missing keys).

- [ ] **Step 3: Implement live discovery and endpoint validation in `app/main.py`**

In `app/main.py`:
1. Add helper `fetch_9router_models(base_url: str, api_key: str | None = None) -> list[str]`:
   ```python
   def fetch_9router_models(base_url: str, api_key: str | None = None) -> list[str]:
       headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
       try:
           resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=2.0)
           if resp.status_code == 200:
               data = resp.json().get("data", [])
               return [m["id"] for m in data if isinstance(m, dict) and "id" in m]
       except Exception:
           pass
       return []
   ```
2. In `get_config()`:
   - Call `fetch_9router_models` if `cfg.ninerouter_api_key and cfg.ninerouter_base_url`, and update `models_cfg["translation"]["9router"]` if returned list is non-empty.
   - Return `"audio_providers": cfg.available_audio_providers()` alongside `"providers": cfg.available_providers()`.
3. In `create_job()`:
   - Accept `audio_provider: str = Form("gemini")`.
   - Validate `audio_provider in ("gemini", "openrouter")`.
   - Validate provider key requirements:
     - If `mode != "translation_only"`: check that `cfg.available_audio_providers().get(audio_provider)` is True (raise `422` with `"API key untuk audio provider {audio_provider!r} belum diisi di .env."`).
     - If `mode != "transcription_only"`: check that `cfg.available_providers().get(translator)` is True.
   - For `transcribe_model`: if `audio_provider == "openrouter"`, accept any non-empty stripped string (custom model support); if `audio_provider == "gemini"`, validate against `models_cfg["transcription"]["gemini"]`.
   - For `translate_model`: if `translator == "openrouter"`, accept any non-empty stripped string; if `translator == "9router"`, accept any model in `models_cfg["translation"]["9router"]` or stripped non-empty string; if other providers, validate against their list.
   - Store `"audio_provider": audio_provider` in `job.params`.
4. In `RetryRequest` & `retry()`:
   - Add `audio_provider: str | None = None`.
   - If provided, validate and update `job.params["audio_provider"]`.

- [ ] **Step 4: Run API tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat(api): add live 9router discovery, audio_provider validation, and custom OpenRouter models"
```

---

### Task 4: Pipeline Inheritance for Timing & QC (`app/pipeline.py`, `tests/test_pipeline.py`)

**Files:**
- Modify: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_pipeline_inherits_audio_provider_and_transcribe_model(env, monkeypatch):
    timing_calls = []
    qc_calls = []

    def fake_make_audio_client(cfg, provider="gemini", model=None, key_label=None):
        timing_calls.append({"provider": provider, "model": model})
        return object()

    def fake_qc(audio_client, rows, audio_path, tmp_dir, system, tracker, **kw):
        qc_calls.append({"deep": kw.get("deep_audio_qc")})
        return rows, {"mode": "hybrid", "summary": {}, "issues": []}

    monkeypatch.setattr("app.providers.make_audio_client", fake_make_audio_client)
    monkeypatch.setattr("app.qc.run_qc_all", fake_qc)

    params = make_params(
        audio_provider="openrouter",
        transcribe_model="qwen/qwen3.8-omni-flash",
        timing_model="",
        qc_model="",
    )
    job = pipeline.create_job(params)
    pipeline.run_job(job.id)

    assert job.status == "done"
    # Timing should have inherited openrouter and qwen3.8-omni-flash
    assert any(c["provider"] == "openrouter" and c["model"] == "qwen/qwen3.8-omni-flash" for c in timing_calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -k "inherits_audio_provider" -v`
Expected: FAIL (`timing_calls` contains `"gemini"` instead of `"openrouter"`).

- [ ] **Step 3: Update `_make_audio_runner`, `_stage_transcribe`, `_stage_timing`, and `_stage_qc` in `app/pipeline.py`**

In `app/pipeline.py`:
1. Update `_make_audio_runner(job, cfg, stage, primary_provider, primary_model, primary_client=None)`:
   - Build client via `providers.make_audio_client(cfg, provider=slot.provider if hasattr(slot, "provider") else primary_provider, model=slot.model, key_label=slot.key_label)`.
2. In `_stage_transcribe`:
   - Read `audio_provider = job.params.get("audio_provider") or "gemini"`.
   - Read `primary_model = job.params.get("transcribe_model") or ...`.
   - Pass `audio_provider` to `make_audio_client` and `_make_audio_runner`.
3. In `_stage_timing`:
   - Read `audio_provider = job.params.get("audio_provider") or "gemini"`.
   - Inherit model: `timing_model = job.params.get("timing_model") or job.params.get("transcribe_model") or ...`.
   - Pass `audio_provider` to `make_audio_client` and `_make_audio_runner`.
4. In `_stage_qc`:
   - Inherit audio model: `qc_model = job.params.get("qc_model") or job.params.get("transcribe_model") or ...`.
   - Read `audio_provider = job.params.get("audio_provider") or "gemini"`.
   - Pass `audio_provider` to `make_audio_client` and `_make_audio_runner`.
   - Pass `translator = job.params.get("translator") or "9router"` and `translate_model` to `_make_text_runner`.

- [ ] **Step 4: Run pipeline tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All pipeline tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): wire audio provider and model inheritance to timing and QC stages"
```

---

### Task 5: Web UI Redesign with 9router & OpenRouter Dropdowns (`static/index.html`)

**Files:**
- Modify: `static/index.html`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write test assertion for UI elements**

In `tests/test_api.py`, update `test_index_contains_fallback_retry_and_qc_elements` to verify:
- `#audio_provider` exists
- `#custom_transcribe_model` and `#custom_translate_model` exist
- `#timing_model` and `#qc_model` are removed from the main form

- [ ] **Step 2: Update HTML layout in `static/index.html`**

1. Replace the model selection rows with:
   ```html
   <!-- Audio Stage -->
   <div class="field-row mode-full mode-trans">
     <div class="field">
       <label class="field-label" for="audio_provider">Audio Provider</label>
       <select id="audio_provider">
         <option value="gemini">Google Gemini</option>
         <option value="openrouter">OpenRouter (Multimodal)</option>
       </select>
     </div>
     <div class="field">
       <label class="field-label" for="transcribe_model">Model transkripsi</label>
       <select id="transcribe_model"></select>
       <input type="text" id="custom_transcribe_model" class="hidden" style="margin-top:6px;" placeholder="Ketik model OpenRouter (contoh: qwen/qwen3.8-omni-flash)">
     </div>
   </div>

   <!-- Text Stage -->
   <div class="field-row mode-full mode-tl">
     <div class="field">
       <label class="field-label" for="translator">Model translasi</label>
       <select id="translator"></select>
     </div>
     <div class="field">
       <label class="field-label" for="translate_model">Varian model terjemahan</label>
       <select id="translate_model"></select>
       <input type="text" id="custom_translate_model" class="hidden" style="margin-top:6px;" placeholder="Ketik model OpenRouter (contoh: deepseek/deepseek-chat)">
     </div>
   </div>
   ```
2. Remove the standalone `#timing_model` and `#qc_model` selects from the main form (leaving only `#deep_audio_qc`).
3. Update `<script>`:
   - In `loadConfig()`:
     - Populate `#audio_provider` based on `cfg.audio_providers`.
     - Update `populateTranscribeModelDropdown(provider)`: populate Gemini models or OpenRouter models + `[Ketik model lain...]`.
     - Update `populateTranslateModelDropdown(provider)`: for 9router, populate live models; for OpenRouter, add curated models + `[Ketik model lain...]`.
     - Listen for `change` on `#transcribe_model` and `#translate_model` to toggle `#custom_transcribe_model` and `#custom_translate_model`.
   - On form submit:
     - Append `audio_provider = $("audio_provider").value`.
     - If custom model selected, append the text input value instead of the select value.
   - In `showRetryBox(stage)`:
     - Keep stage-aware overrides for failed stages.

- [ ] **Step 3: Run Node syntax check and test suite**

Run:
```bash
python -m pytest tests/test_api.py -v
python -m pytest -q
```
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add static/index.html tests/test_api.py
git commit -m "feat(ui): add audio_provider selector, custom OpenRouter inputs, and remove redundant timing/qc dropdowns"
```

---

## Plan Self-Review Checklist

- **Spec coverage check:**
  - 9router text provider + live `/v1/models` discovery -> Tasks 2, 3, 5
  - OpenRouter audio provider (base64 `input_audio`) + text provider -> Tasks 2, 3, 4
  - Curated models in `models.json` + `qwen/qwen3.8-omni-flash` -> Task 1
  - Optional `GEMINI_API_KEY` at boot + on-demand key validation -> Task 2, 3
  - Audio Provider & Text Provider inheritance across Timing & QC -> Task 4
  - Custom OpenRouter model text input in UI -> Tasks 3, 5
  - Removed redundant Timing & QC model dropdowns from main form -> Task 5
- **No placeholders check:** All signatures, test functions, and data structures are specified in full.
