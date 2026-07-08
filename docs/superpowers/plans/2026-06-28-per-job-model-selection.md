# Per-Job Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-job transcription and translation model dropdowns so users can choose from a curated list of models on each job submission.

**Architecture:** A single `config/models.json` file is the source of truth for all model lists and defaults. The backend loads it at startup and exposes it via `/api/config`. `POST /api/jobs` accepts `transcribe_model` and `translate_model` form fields, validates them against the list, and stores them in `job.params`. The pipeline passes per-job models to `make_transcriber` / `make_translator`. The frontend populates two dropdowns from the API response.

**Tech Stack:** Python (FastAPI, dataclasses), JSON config, vanilla JS (existing index.html pattern)

## Global Constraints

- All new form fields default gracefully: empty string → fall back to `models.json` default
- Validation: reject unknown model names with HTTP 422
- `make_transcriber` and `make_translator` keep backward-compatible signatures (optional `model` param)
- Tests use `pytest` + `monkeypatch`; no real API calls
- Full test suite must stay green (`pytest` — currently 88+ tests)

---

### Task 1: `config/models.json` + load helper in `providers.py`

**Files:**
- Create: `config/models.json`
- Modify: `app/providers.py` — add `load_models_config()`, `MODELS_CONFIG_PATH` constant
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces: `load_models_config() -> dict` — returns full parsed JSON; `MODELS_CONFIG_PATH: Path`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_providers.py`:

```python
from pathlib import Path
from app.providers import load_models_config, MODELS_CONFIG_PATH

def test_load_models_config_returns_expected_keys():
    cfg = load_models_config()
    assert "transcription" in cfg
    assert "translation" in cfg
    assert "defaults" in cfg

def test_models_config_transcription_list_nonempty():
    cfg = load_models_config()
    assert len(cfg["transcription"]) > 0

def test_models_config_defaults_match_lists():
    cfg = load_models_config()
    assert cfg["defaults"]["transcription"] in cfg["transcription"]
    assert cfg["defaults"]["translation"]["gemini"] in cfg["translation"]["gemini"]
    assert cfg["defaults"]["translation"]["openai"] in cfg["translation"]["openai"]
    assert cfg["defaults"]["translation"]["anthropic"] in cfg["translation"]["anthropic"]

def test_models_config_path_exists():
    assert MODELS_CONFIG_PATH.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_providers.py::test_load_models_config_returns_expected_keys -v
```
Expected: `FAIL` — `cannot import name 'load_models_config'`

- [ ] **Step 3: Create `config/models.json`**

```json
{
  "transcription": [
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro"
  ],
  "translation": {
    "gemini": [
      "gemini-3.5-flash",
      "gemini-3.1-pro-preview",
      "gemini-3-flash-preview",
      "gemini-3.1-flash-lite",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
      "gemini-2.5-pro"
    ],
    "openai": [
      "gpt-4.1",
      "gpt-4.1-mini",
      "gpt-4o",
      "gpt-4o-mini"
    ],
    "anthropic": [
      "claude-opus-4-8",
      "claude-sonnet-4-6",
      "claude-haiku-4-5"
    ]
  },
  "defaults": {
    "transcription": "gemini-3.1-pro-preview",
    "translation": {
      "gemini": "gemini-3.5-flash",
      "openai": "gpt-4o",
      "anthropic": "claude-sonnet-4-6"
    }
  }
}
```

- [ ] **Step 4: Add `load_models_config` to `providers.py`**

Add near the top of `app/providers.py` (after imports):

```python
MODELS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.json"


def load_models_config() -> dict:
    return json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_providers.py -v -k "models_config"
```
Expected: 4 PASS

- [ ] **Step 6: Commit**

```
git add config/models.json app/providers.py tests/test_providers.py
git commit -m "feat: add config/models.json and load_models_config() helper"
```

---

### Task 2: `providers.py` — update `make_transcriber` / `make_translator` signatures

**Files:**
- Modify: `app/providers.py` — add optional `model` param to both factory functions
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `load_models_config()` from Task 1
- Produces:
  - `make_transcriber(cfg: Config, model: str | None = None) -> GeminiClient`
  - `make_translator(cfg: Config, provider: str, model: str | None = None) -> GeminiClient | OpenAIClient | AnthropicClient`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_providers.py`:

```python
from unittest.mock import MagicMock, patch
from app.providers import make_transcriber, make_translator, Config

def _fake_cfg():
    return Config(
        gemini_api_key="key",
        openai_api_key="okey",
        anthropic_api_key="akey",
        transcribe_model="gemini-2.5-pro",
        translate_models={"gemini": "gemini-2.5-flash", "openai": "gpt-4o", "anthropic": "claude-sonnet-4-6"},
        ytdlp_cookies_file=None,
        sub_min_duration=0.7,
        sub_max_duration=7.5,
        sub_merge_gap=0.5,
        sub_cps_flag=25,
    )

def test_make_transcriber_uses_model_override():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_transcriber(cfg, model="gemini-3.1-pro-preview")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-3.1-pro-preview")

def test_make_transcriber_falls_back_to_cfg_model():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_transcriber(cfg)
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-2.5-pro")

def test_make_translator_uses_model_override():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_translator(cfg, "gemini", model="gemini-3.5-flash")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-3.5-flash")

def test_make_translator_falls_back_to_cfg_model():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_translator(cfg, "gemini")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-2.5-flash")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_providers.py::test_make_transcriber_uses_model_override -v
```
Expected: `FAIL` — unexpected keyword argument `model`

- [ ] **Step 3: Update `make_transcriber` and `make_translator` in `providers.py`**

Replace the current implementations at the bottom of `providers.py`:

```python
def make_transcriber(cfg: Config, model: str | None = None) -> GeminiClient:
    return GeminiClient(cfg.gemini_api_key, model or cfg.transcribe_model)


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
```

- [ ] **Step 4: Run all new tests**

```
pytest tests/test_providers.py -v -k "make_transcriber or make_translator"
```
Expected: 4 PASS

- [ ] **Step 5: Run full suite to verify nothing broke**

```
pytest -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```
git add app/providers.py tests/test_providers.py
git commit -m "feat: add optional model override to make_transcriber and make_translator"
```

---

### Task 3: `main.py` — expose models in `/api/config`, accept + validate model fields in `POST /api/jobs`

**Files:**
- Modify: `app/main.py` — update `get_config()`, update `create_job()`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `load_models_config()` from Task 1
- Produces:
  - `GET /api/config` returns additional key `"models"` with full `load_models_config()` payload
  - `POST /api/jobs` accepts `transcribe_model: str = Form("")` and `translate_model: str = Form("")`; validates against lists; stores in `job.params`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_api.py`:

```python
from app.providers import load_models_config

def test_config_includes_models_list(client, monkeypatch):
    monkeypatch.setattr("app.audio.ensure_ffmpeg", lambda: True)
    body = client.get("/api/config").json()
    assert "models" in body
    models = body["models"]
    assert "transcription" in models
    assert "translation" in models
    assert "defaults" in models
    assert "gemini-3.1-pro-preview" in models["transcription"]

def test_create_job_accepts_transcribe_model(client, fake_run):
    resp = post_url_job(client, transcribe_model="gemini-2.5-flash")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["transcribe_model"] == "gemini-2.5-flash"

def test_create_job_accepts_translate_model(client, fake_run):
    resp = post_url_job(client, translate_model="gemini-3.5-flash")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["translate_model"] == "gemini-3.5-flash"

def test_create_job_unknown_transcribe_model_rejected(client):
    resp = post_url_job(client, transcribe_model="gpt-999-turbo")
    assert resp.status_code == 422

def test_create_job_unknown_translate_model_rejected(client):
    resp = post_url_job(client, translator="gemini", translate_model="gpt-999-turbo")
    assert resp.status_code == 422

def test_create_job_empty_model_uses_default(client, fake_run):
    models_cfg = load_models_config()
    resp = post_url_job(client, transcribe_model="")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["transcribe_model"] == models_cfg["defaults"]["transcription"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_api.py::test_config_includes_models_list -v
```
Expected: `FAIL` — `"models" not in body`

- [ ] **Step 3: Update `get_config()` in `main.py`**

In `app/main.py`, update the `get_config()` function:

```python
@app.get("/api/config")
def get_config() -> dict:
    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "providers": cfg.available_providers(),
        "translate_models": cfg.translate_models,
        "transcribe_model": cfg.transcribe_model,
        "ffmpeg": audio.ensure_ffmpeg(),
        "gemini_keys": list(providers.get_all_gemini_keys().keys()),
        "active_gemini_key": providers.get_active_gemini_label(),
        "models": providers.load_models_config(),
    }
```

- [ ] **Step 4: Update `create_job()` in `main.py`**

Replace the current `create_job` signature and body:

```python
@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     source: str = Form(...),
                     url: str = Form(""),
                     translator: str = Form("gemini"),
                     output_format: str = Form("both"),
                     save_mp4: bool = Form(False),
                     group: str = Form("sakurazaka"),
                     context_preset: str = Form("sakurazaka_sokomagattara"),
                     context_override: str = Form(""),
                     additional_context: str = Form(""),
                     transcribe_model: str = Form(""),
                     translate_model: str = Form(""),
                     file: UploadFile | None = File(None)) -> dict:
    if output_format not in VALID_FORMATS:
        raise HTTPException(422, f"output_format tidak dikenal: {output_format!r}")

    models_cfg = providers.load_models_config()

    resolved_transcribe_model = transcribe_model.strip() or models_cfg["defaults"]["transcription"]
    if resolved_transcribe_model not in models_cfg["transcription"]:
        raise HTTPException(422, f"transcribe_model tidak dikenal: {resolved_transcribe_model!r}")

    resolved_translate_model = translate_model.strip() or models_cfg["defaults"]["translation"].get(translator, "")
    provider_model_list = models_cfg["translation"].get(translator, [])
    if resolved_translate_model and resolved_translate_model not in provider_model_list:
        raise HTTPException(422, f"translate_model tidak dikenal untuk provider {translator!r}: {resolved_translate_model!r}")

    upload_bytes: bytes | None = None
    upload_name: str | None = None
    if source == "url":
        if not url.strip():
            raise HTTPException(422, "URL YouTube belum diisi.")
    elif source == "file":
        if file is None:
            raise HTTPException(422, "File belum dipilih.")
        upload_bytes = await file.read()
        upload_name = file.filename
    else:
        raise HTTPException(422, f"source tidak dikenal: {source!r}")

    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        raise HTTPException(500, str(exc)) from exc
    if not cfg.available_providers().get(translator):
        raise HTTPException(
            422, f"API key untuk provider {translator!r} belum diisi di .env.")

    params = {"source": source, "url": url.strip() or None,
              "translator": translator, "output_format": output_format,
              "save_mp4": save_mp4, "original_filename": upload_name,
              "group": group.strip() or "sakurazaka",
              "context_preset": context_preset.strip() or "sakurazaka_sokomagattara",
              "context_override": context_override.strip() or None,
              "additional_context": additional_context.strip() or None,
              "transcribe_model": resolved_transcribe_model,
              "translate_model": resolved_translate_model}
    job = pipeline.create_job(params, upload_bytes=upload_bytes,
                              upload_filename=upload_name)
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id}
```

- [ ] **Step 5: Run new API tests**

```
pytest tests/test_api.py -v -k "model"
```
Expected: all PASS

- [ ] **Step 6: Run full suite**

```
pytest -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```
git add app/main.py tests/test_api.py
git commit -m "feat: expose models list in /api/config and validate per-job model fields"
```

---

### Task 4: `pipeline.py` — pass per-job model to `make_transcriber` / `make_translator`

**Files:**
- Modify: `app/pipeline.py` — update `_stage_transcribe` and `_stage_translate`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `make_transcriber(cfg, model=None)` and `make_translator(cfg, provider, model=None)` from Task 2
- Consumes: `job.params["transcribe_model"]` and `job.params["translate_model"]` from Task 3

- [ ] **Step 1: Update pipeline mocks in `tests/test_pipeline.py`**

In `tests/test_pipeline.py`, update the monkeypatches in the `env` fixture to accept the new optional `model` parameter:

```python
monkeypatch.setattr("app.providers.make_transcriber", lambda cfg, model=None: object())
monkeypatch.setattr("app.providers.make_translator", lambda cfg, p, model=None: object())
```

Also update `make_params` to include the new fields:

```python
def make_params(**overrides):
    params = {"source": "url", "url": "https://youtu.be/x", "save_mp4": False,
              "translator": "gemini", "output_format": "both",
              "original_filename": None,
              "transcribe_model": "gemini-3.1-pro-preview",
              "translate_model": "gemini-3.5-flash"}
    params.update(overrides)
    return params
```

- [ ] **Step 2: Write new failing test**

Add to `tests/test_pipeline.py`:

```python
def test_pipeline_passes_transcribe_model_to_maker(env, tmp_path, monkeypatch):
    called_with = {}

    def fake_maker(cfg, model=None):
        called_with["model"] = model
        return object()

    monkeypatch.setattr("app.providers.make_transcriber", fake_maker)
    params = make_params(transcribe_model="gemini-2.5-flash")
    job = pipeline.create_job(params)
    pipeline.run_job(job.id)
    assert called_with["model"] == "gemini-2.5-flash"


def test_pipeline_passes_translate_model_to_maker(env, tmp_path, monkeypatch):
    called_with = {}

    def fake_maker(cfg, provider, model=None):
        called_with["model"] = model
        return object()

    monkeypatch.setattr("app.providers.make_translator", fake_maker)
    params = make_params(translate_model="gemini-3.5-flash")
    job = pipeline.create_job(params)
    pipeline.run_job(job.id)
    assert called_with["model"] == "gemini-3.5-flash"
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_pipeline.py::test_pipeline_passes_transcribe_model_to_maker -v
```
Expected: `FAIL` — `called_with["model"]` is `None` (model not passed yet)

- [ ] **Step 4: Update `_stage_transcribe` in `pipeline.py`**

Find the line `gemini = providers.make_transcriber(cfg)` (around line 210) and replace it:

```python
gemini = providers.make_transcriber(cfg, model=job.params.get("transcribe_model"))
```

- [ ] **Step 5: Update `_stage_translate` in `pipeline.py`**

Find the line `client = providers.make_translator(cfg, job.params["translator"])` (around line 283) and replace it:

```python
client = providers.make_translator(
    cfg, job.params["translator"],
    model=job.params.get("translate_model") or None,
)
```

- [ ] **Step 6: Run new pipeline tests**

```
pytest tests/test_pipeline.py -v -k "model"
```
Expected: 2 PASS

- [ ] **Step 7: Run full suite**

```
pytest -v
```
Expected: all PASS

- [ ] **Step 8: Commit**

```
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: pass per-job transcribe_model and translate_model through pipeline"
```

---

### Task 5: `static/index.html` — add model dropdowns

**Files:**
- Modify: `static/index.html` — add two `<select>` elements, populate from `/api/config`, wire into form submission

**Interfaces:**
- Consumes: `GET /api/config` → `body.models` (from Task 3)
- Produces: `transcribe_model` and `translate_model` fields appended to `FormData` on submit

- [ ] **Step 1: Add transcription model `<select>` to the HTML**

Find the settings panel section containing `<select id="translator">`. Add a transcription model row before it (before the "Model translasi" label). The transcription section currently has `<div class="pipe-stage" id="pipe-transcribe">` in the pipeline display — add the select near the other settings fields:

Locate this block:
```html
          <label class="field-label" for="translator">Model translasi</label>
          <select id="translator"></select>
```

Insert before it:
```html
          <label class="field-label" for="transcribe_model">Model transkripsi</label>
          <select id="transcribe_model" style="margin-bottom:8px;"></select>
```

- [ ] **Step 2: Add translation model `<select>` after the provider dropdown**

After `<select id="translator"></select>`, insert:
```html
          <label class="field-label" for="translate_model" style="margin-top:6px;">Model terjemahan</label>
          <select id="translate_model" style="margin-bottom:8px;"></select>
```

- [ ] **Step 3: Populate dropdowns in `loadConfig()`**

Find the `loadConfig()` function (it calls `/api/config` and populates `#translator`). Add model population after the translator loop. Locate the block that starts with:

```js
  const sel = $("translator");
  for (const [prov, available] of Object.entries(cfg.providers)) {
```

After that block (after the translator `for` loop closes), add:

```js
  // Populate transcription model dropdown
  const modelsConfig = cfg.models || {};
  const tmSel = $("transcribe_model");
  tmSel.innerHTML = "";
  for (const m of (modelsConfig.transcription || [])) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === (modelsConfig.defaults?.transcription)) opt.selected = true;
    tmSel.appendChild(opt);
  }

  // Populate translation model dropdown (initially for gemini)
  populateTranslateModelDropdown(modelsConfig, $("translator").value);

  // Re-populate when provider changes
  $("translator").addEventListener("change", () => {
    populateTranslateModelDropdown(modelsConfig, $("translator").value);
  });
```

- [ ] **Step 4: Add `populateTranslateModelDropdown` helper function**

Near the other JS helper functions (e.g., near `PROVIDER_LABELS`), add:

```js
function populateTranslateModelDropdown(modelsConfig, provider) {
  const sel = $("translate_model");
  sel.innerHTML = "";
  const list = (modelsConfig.translation || {})[provider] || [];
  const def = (modelsConfig.defaults?.translation || {})[provider] || "";
  for (const m of list) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === def) opt.selected = true;
    sel.appendChild(opt);
  }
}
```

- [ ] **Step 5: Append model fields to FormData on submit**

Find the form submission block that builds `fd` (FormData). It currently has lines like:

```js
  fd.append("translator", $("translator").value);
```

Add after them:
```js
  fd.append("transcribe_model", $("transcribe_model").value);
  fd.append("translate_model", $("translate_model").value);
```

- [ ] **Step 6: Manual smoke test**

Start the server:
```
uvicorn app.main:app --reload
```
Open http://localhost:8000 and verify:
1. "Model transkripsi" dropdown appears with all 7 Gemini models; `gemini-3.1-pro-preview` selected by default
2. "Model terjemahan" dropdown appears; shows Gemini models with `gemini-3.5-flash` selected by default
3. Switching the provider to OpenAI repopulates the translation dropdown with `gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini`, defaulting to `gpt-4o`
4. Switching to Anthropic shows Claude models, defaulting to `claude-sonnet-4-6`
5. Submitting a job stores the chosen models in `job.params` (check `/api/jobs/{id}`)

- [ ] **Step 7: Commit**

```
git add static/index.html
git commit -m "feat: add per-job transcription and translation model dropdowns"
```
