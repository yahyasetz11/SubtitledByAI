# Sokomagattara SubGen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Localhost web app that transcribes (JP) and translates (ID) episodes of *Soko Magattara, Sakurazaka?* and renders HikaLeon-style `.ass`/`.srt` subtitles.

**Architecture:** FastAPI backend serving a single vanilla-JS `index.html`. Jobs run async via `BackgroundTasks`; progress streams over SSE. Audio is normalized and chunked with ffmpeg (silencedetect cascade), transcribed chunk-by-chunk with Gemini 2.5 Pro via the Files API, translated as numbered JSON batches (Gemini Flash default, GPT/Claude optional), then deterministically post-processed and rendered from `context/template.ass`. Every stage checkpoints to `output/{job_id}/` so retries never redo completed work.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, google-genai, openai, anthropic, yt-dlp, ffmpeg (external binary), python-dotenv, pytest + httpx for tests. Frontend: one static HTML file, no build step.

---

## Context for the implementing engineer

- **Project root** is the current repo root: `C:\Users\YAHYASETZ\Documents\SubtitledByAI`. The spec shows a folder named `sokomagattara-subgen/`; we use the existing repo root instead (user request: "implement in this directory"). All paths below are relative to the repo root.
- **Platform is Windows 11 / PowerShell.** Test commands use `python -m pytest`. ffmpeg and ffprobe must be on PATH (`winget install Gyan.FFmpeg` if missing). Subprocess code must work without a shell (pass arg lists, never shell strings).
- **The spec is `SPEC-sokomagattara-subgen.md`** at the repo root. Read it if a requirement seems ambiguous — section numbers below (§5, §7, etc.) refer to it.
- **Run tests** with: `python -m pytest -q` (pytest.ini sets testpaths). Run a single test: `python -m pytest tests/test_audio.py::test_name -v`.
- The git repo exists but has **no commits yet** — Task 1's commit is the initial commit.
- **No real API keys or network in tests.** All LLM/ffmpeg interactions are behind small seams (pure parsers, arg-list builders, duck-typed clients) and tests use fakes. `tests/conftest.py` sets a dummy `GEMINI_API_KEY`.

### Documented deviations from the spec (intentional)

1. **yt-dlp version floor, not exact pin.** An exact pin from today would go stale and break YouTube downloads within months; we use `yt-dlp>=2025.1.26` plus the spec-mandated error message suggesting `pip install -U yt-dlp`. (Spec §1 asks for a pin; the floor preserves the intent — reproducible installs that can still be upgraded when YouTube changes.)
2. **Translation batches are groups of ≤100 utterances** carved from the merged transcript, not literal audio-chunk groupings (spec §6b says "per chunk"). After merge+dedupe the chunk grouping is gone; fixed-size batches give the same alignment-validation property with more uniform request sizes. Checkpointing is per batch, same as per chunk.
3. **`template.ass` style values are sensible HikaLeon-like defaults** (Comic Sans MS, PlayRes 1920×1080, bold white w/ black outline). The real HikaLeon episode-#288 file isn't in the repo; the user can paste exact style lines into `context/template.ass` later — the renderer copies the template header verbatim, so no code change is needed.

### Final file map

```
app/__init__.py
app/main.py          # FastAPI app: /config, POST /jobs, SSE, downloads, retry, static
app/pipeline.py      # Job registry + stage orchestration + checkpoints + SSE events
app/audio.py         # ffmpeg/yt-dlp wrappers, silencedetect parse, chunk planner
app/transcribe.py    # prompts, transcript parsing, merge + overlap dedupe, truncation split
app/translate.py     # batch JSON translation, alignment validation, per-line fallback
app/subtitle.py      # deterministic post-processing (§7), .ass/.srt rendering
app/providers.py     # Config, retry wrapper, LLM clients, UsageTracker
context/members.md   # moved from repo root, fixed per §4
context/context.md   # moved from repo root
context/template.ass # HikaLeon-style header + styles
static/index.html    # the whole UI
tests/conftest.py
tests/test_providers.py
tests/test_audio.py
tests/test_transcribe.py
tests/test_translate.py
tests/test_subtitle.py
tests/test_pipeline.py
tests/test_api.py
requirements.txt, pytest.ini, .env.example, .gitignore, README.md
output/              # gitignored, created at runtime
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.env.example`, `.gitignore`, `app/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create directories**

```powershell
New-Item -ItemType Directory -Force app, tests, static, context | Out-Null
```

(`output/` is created at runtime by the pipeline; don't commit it.)

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi>=0.115
uvicorn[standard]>=0.32
python-dotenv>=1.0
python-multipart>=0.0.12
google-genai>=1.0
openai>=1.60
anthropic>=0.45
# Floor, not exact pin: YouTube changes frequently; on download failure the app
# tells the user to run: pip install -U yt-dlp
yt-dlp>=2025.1.26
pytest>=8.3
httpx>=0.27
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 4: Write `.env.example`** (copy of spec §2; user copies to `.env` and fills keys)

```env
GEMINI_API_KEY=            # WAJIB — transkripsi + translasi default
OPENAI_API_KEY=            # opsional
ANTHROPIC_API_KEY=         # opsional

TRANSCRIBE_MODEL=gemini-2.5-pro
TRANSLATE_MODEL_GEMINI=gemini-2.5-flash
TRANSLATE_MODEL_OPENAI=gpt-4o
TRANSLATE_MODEL_ANTHROPIC=claude-sonnet-4-6

YTDLP_COOKIES_FILE=        # opsional, untuk video region-locked/membership

SUB_MIN_DURATION=0.7
SUB_MAX_DURATION=7.5
SUB_MERGE_GAP=0.5
SUB_CPS_FLAG=25
```

- [ ] **Step 5: Write `.gitignore`**

```
.env
output/
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
```

- [ ] **Step 6: Create empty `app/__init__.py` and `tests/__init__.py`**

Both files are empty (zero bytes is fine).

- [ ] **Step 7: Install dependencies and verify pytest runs**

Run: `python -m pip install -r requirements.txt`
Then: `python -m pytest -q`
Expected: `no tests ran` (exit code 5 is fine at this point).

- [ ] **Step 8: Commit**

```powershell
git add requirements.txt pytest.ini .env.example .gitignore app/__init__.py tests/__init__.py SPEC-sokomagattara-subgen.md docs/
git commit -m "chore: scaffold sokomagattara-subgen project"
```

---

### Task 2: Fix and relocate context files (§4)

The repo root already has `members.md` and `context.md`. They belong in `context/`. `context.md` is already correct per spec. `members.md` needs three fixes (§4): the roster itself is already correct (Morita Hikaru & Tamura Hono are in the active 2nd-gen table; Saito Fuyuka is in graduated) — verify, don't change.

**Files:**
- Move: `members.md` → `context/members.md`
- Move: `context.md` → `context/context.md`
- Modify: `context/members.md`

- [ ] **Step 1: Move both files**

```powershell
Move-Item members.md context/members.md
Move-Item context.md context/context.md
```

- [ ] **Step 2: Fix the 握手会 typo in `context/members.md`**

In the "Show-Specific Vocabulary" table, the row

```
| 握手会                 | akushukai                 | sesi握手 / meet & greet                     |
```

becomes

```
| 握手会                 | akushukai                 | meet & greet                                |
```

- [ ] **Step 3: Remove the "please verify roster" note**

Delete this entire bullet from the "Notes for Translator" section (a context file must speak with certainty — §4):

```
- **Please verify the 3rd generation roster** — this list may be incomplete or contain errors; Yahya should confirm against the latest official lineup
```

- [ ] **Step 4: Remove stray tab characters**

Search for literal tabs and replace each with a single space (the spec calls out a stray tab; markdown tables must use spaces):

```powershell
(Get-Content context/members.md -Raw) -replace "`t", " " | Set-Content context/members.md -Encoding utf8 -NoNewline
```

Only run this if a tab is actually found (`Select-String -Path context/members.md -Pattern "`t"`); otherwise skip.

- [ ] **Step 5: Verify roster correctness (read-only check)**

Confirm in `context/members.md`: Morita Hikaru (森田ひかる) and Tamura Hono (田村保乃) appear under "Current Members / 2nd Generation"; Saito Fuyuka (齋藤冬優花) appears under "Graduated Members". All three are already correct — change nothing.

- [ ] **Step 6: Commit**

```powershell
git add context/ members.md context.md
git commit -m "fix: relocate context files and correct members.md per spec section 4"
```

### Task 3: HikaLeon `.ass` template

**Files:**
- Create: `context/template.ass`

- [ ] **Step 1: Write `context/template.ass`**

Exact content (the renderer in Task 15 copies everything through the `[Events]` `Format:` line verbatim, then appends `Dialogue:` lines — so the user can later overwrite styles with the real HikaLeon values without code changes):

```
[Script Info]
; HikaLeon Subs style template — Sokomagattara SubGen
; Styles are sensible defaults; paste exact style lines from a real HikaLeon .ass to match perfectly.
Title: Soko Magattara, Sakurazaka?
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Comic Sans MS,66,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,3.6,1.8,2,60,60,45,1
Style: Narrator,Comic Sans MS,60,&H00F5F5F5,&H000000FF,&H00000000,&H96000000,-1,-1,0,0,100,100,0,0,1,3.6,1.8,8,60,60,45,1
Style: yt_default,Roboto Medium,57,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,3,3,0,2,60,60,45,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text
```

Notes: `Default` = bottom-center bold white (dialogue), `Narrator` = top-center italic (voice-over), `yt_default` kept for compatibility with HikaLeon files. Save as UTF-8 **without BOM** (use the Write tool, not `Out-File`).

- [ ] **Step 2: Commit**

```powershell
git add context/template.ass
git commit -m "feat: add HikaLeon-style .ass template (PlayRes 1920x1080)"
```

---

### Task 4: Config loading + key validation (`providers.py` part 1)

`Config` reads `.env`/environment once per call (cheap, supports hot-ish reload), validates the mandatory Gemini key, and reports which translator providers are available (§2).

**Files:**
- Create: `app/providers.py`
- Create: `tests/conftest.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import os

# Set BEFORE app modules import, so startup validation in tests never trips
# on a missing key. Individual tests override with monkeypatch.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
```

- [ ] **Step 2: Write the failing tests**

`tests/test_providers.py`:

```python
import pytest

from app.providers import Config, ConfigError


def test_config_loads_keys_and_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_MODEL", raising=False)

    cfg = Config.load()

    assert cfg.gemini_api_key == "g-key"
    assert cfg.transcribe_model == "gemini-2.5-pro"
    assert cfg.translate_models == {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
    }
    assert cfg.available_providers() == {
        "gemini": True, "openai": False, "anthropic": False,
    }
    assert cfg.sub_min_duration == pytest.approx(0.7)
    assert cfg.sub_max_duration == pytest.approx(7.5)
    assert cfg.sub_merge_gap == pytest.approx(0.5)
    assert cfg.sub_cps_flag == pytest.approx(25.0)


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "gemini-3.0-pro")
    monkeypatch.setenv("SUB_CPS_FLAG", "30")

    cfg = Config.load()

    assert cfg.transcribe_model == "gemini-3.0-pro"
    assert cfg.sub_cps_flag == pytest.approx(30.0)
    assert cfg.available_providers()["openai"] is True


def test_config_missing_gemini_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        Config.load()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.providers'` (or ImportError).

- [ ] **Step 4: Write `app/providers.py` (Config section)**

```python
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
```

Note: `load_dotenv(override=False)` means real environment variables win over `.env`, and the tests' `monkeypatch.delenv` works as long as no `.env` file with keys exists on the test machine. The repo's `.gitignore` keeps `.env` untracked; tests on this machine may have a `.env` — if `test_config_missing_gemini_key_raises` fails because a local `.env` provides the key, change the test to also `monkeypatch.setattr("app.providers.load_dotenv", lambda **kw: None)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```powershell
git add app/providers.py tests/conftest.py tests/test_providers.py
git commit -m "feat: config loading with provider availability and key validation"
```

### Task 5: Transport retry wrapper (`providers.py` part 2) — §9 layer 1

Exponential backoff ×3 retries (4 attempts total), honoring `Retry-After`, for 429/5xx/network errors from any provider SDK. Provider-agnostic: inspects common exception attributes instead of importing provider exception types.

**Files:**
- Modify: `app/providers.py` (append)
- Test: `tests/test_providers.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_providers.py`)

```python
from app.providers import call_with_retries


class FakeHTTPError(Exception):
    """Mimics provider SDK errors: .status_code + .response.headers."""

    def __init__(self, status_code, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code

        class _Resp:
            headers = {"retry-after": retry_after} if retry_after else {}
        self.response = _Resp()


def test_retry_succeeds_after_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeHTTPError(503)
        return "ok"

    assert call_with_retries(flaky, sleep=lambda s: None) == "ok"
    assert calls["n"] == 3


def test_retry_respects_retry_after_header():
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeHTTPError(429, retry_after="7")
        return "ok"

    assert call_with_retries(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [7.0]


def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_500():
        calls["n"] += 1
        raise FakeHTTPError(500)

    with pytest.raises(FakeHTTPError):
        call_with_retries(always_500, sleep=lambda s: None)
    assert calls["n"] == 4  # 1 attempt + 3 retries


def test_non_retryable_error_raises_immediately():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise FakeHTTPError(400)

    with pytest.raises(FakeHTTPError):
        call_with_retries(bad_request, sleep=lambda s: None)
    assert calls["n"] == 1


def test_network_errors_are_retryable():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return "ok"

    assert call_with_retries(flaky, sleep=lambda s: None) == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v -k retry or network`
Expected: FAIL — `ImportError: cannot import name 'call_with_retries'`.

- [ ] **Step 3: Implement** (append to `app/providers.py`; add `import time` and `import httpx` to the imports block)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/providers.py tests/test_providers.py
git commit -m "feat: provider-agnostic retry with backoff and retry-after"
```

---

### Task 6: Usage tracking (`providers.py` part 3) — §10

Accumulates token usage per model into `output/{job_id}/usage.json` and estimates cost from a pricing table.

**Files:**
- Modify: `app/providers.py` (append)
- Test: `tests/test_providers.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_providers.py`)

```python
import json

from app.providers import UsageTracker


def test_usage_accumulates_and_persists(tmp_path):
    path = tmp_path / "usage.json"
    tracker = UsageTracker(path)
    tracker.add("gemini-2.5-pro", input_tokens=40_000, output_tokens=10_000)
    tracker.add("gemini-2.5-pro", input_tokens=12_000, output_tokens=2_000)
    tracker.add("gemini-2.5-flash", input_tokens=8_000, output_tokens=3_000)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["models"]["gemini-2.5-pro"]["input_tokens"] == 52_000
    assert data["models"]["gemini-2.5-pro"]["output_tokens"] == 12_000

    summary = tracker.summary()
    assert summary["input_tokens"] == 60_000
    assert summary["output_tokens"] == 15_000
    # pro: 52k*1.25/1M + 12k*10/1M = 0.065 + 0.12 = 0.185
    # flash: 8k*0.30/1M + 3k*2.50/1M = 0.0024 + 0.0075 = 0.0099
    assert summary["cost_usd"] == pytest.approx(0.1949, abs=1e-4)
    assert "60.0K in / 15.0K out" in summary["line"]
    assert "$0.19" in summary["line"]


def test_usage_reloads_existing_file(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(path).add("gpt-4o", 1000, 500)
    tracker2 = UsageTracker(path)  # e.g. after a checkpoint retry
    tracker2.add("gpt-4o", 1000, 500)
    assert tracker2.summary()["input_tokens"] == 2000


def test_usage_unknown_model_costs_zero(tmp_path):
    tracker = UsageTracker(tmp_path / "usage.json")
    tracker.add("some-future-model", 1_000_000, 1_000_000)
    summary = tracker.summary()
    assert summary["cost_usd"] == 0.0
    assert summary["unpriced_models"] == ["some-future-model"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -k usage -v`
Expected: FAIL — `ImportError: cannot import name 'UsageTracker'`.

- [ ] **Step 3: Implement** (append to `app/providers.py`; add `import json` and `from pathlib import Path` to imports)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/providers.py tests/test_providers.py
git commit -m "feat: token usage tracking with cost estimation"
```

### Task 7: LLM clients (`providers.py` part 4)

Thin clients with one shared shape so `transcribe.py`/`translate.py` stay provider-agnostic and tests can use fakes. Each `generate()` returns an `LLMResponse(text, input_tokens, output_tokens, truncated)`. SDK-response→`LLMResponse` mapping lives in module-level functions so it's testable without network.

**Files:**
- Modify: `app/providers.py` (append)
- Test: `tests/test_providers.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_providers.py`)

```python
from types import SimpleNamespace

from app.providers import (
    LLMResponse, _from_anthropic, _from_gemini, _from_openai, make_translator,
)


def _gemini_resp(text="hi", finish="STOP"):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20,
        ),
        candidates=[SimpleNamespace(
            finish_reason=SimpleNamespace(name=finish),
        )],
    )


def test_from_gemini_maps_fields():
    resp = _from_gemini(_gemini_resp())
    assert resp == LLMResponse(text="hi", input_tokens=100, output_tokens=20,
                               truncated=False)


def test_from_gemini_detects_truncation():
    assert _from_gemini(_gemini_resp(finish="MAX_TOKENS")).truncated is True


def test_from_openai_maps_fields():
    raw = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="halo"), finish_reason="length",
        )],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
    )
    resp = _from_openai(raw)
    assert resp == LLMResponse(text="halo", input_tokens=50, output_tokens=10,
                               truncated=True)


def test_from_anthropic_maps_fields():
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ya")],
        usage=SimpleNamespace(input_tokens=30, output_tokens=5),
        stop_reason="end_turn",
    )
    resp = _from_anthropic(raw)
    assert resp == LLMResponse(text="ya", input_tokens=30, output_tokens=5,
                               truncated=False)


def test_make_translator_requires_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config.load()
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_translator(cfg, "openai")


def test_make_translator_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = Config.load()
    with pytest.raises(ConfigError):
        make_translator(cfg, "deepseek")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -v`
Expected: new tests FAIL with ImportError.

- [ ] **Step 3: Implement** (append to `app/providers.py`)

```python
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


class OpenAIClient:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, max_retries=0)  # we retry ourselves
        self.model = model

    def generate(self, system: str, user: str, audio=None) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return _from_openai(resp)


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


def make_transcriber(cfg: Config) -> GeminiClient:
    return GeminiClient(cfg.gemini_api_key, cfg.transcribe_model)


def make_translator(cfg: Config, provider: str):
    if provider == "gemini":
        return GeminiClient(cfg.gemini_api_key, cfg.translate_models["gemini"])
    if provider == "openai":
        if not cfg.openai_api_key:
            raise ConfigError("OPENAI_API_KEY belum diisi di .env")
        return OpenAIClient(cfg.openai_api_key, cfg.translate_models["openai"])
    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY belum diisi di .env")
        return AnthropicClient(cfg.anthropic_api_key, cfg.translate_models["anthropic"])
    raise ConfigError(f"Penerjemah tidak dikenal: {provider}")
```

Notes for the OpenAI JSON mode: `response_format={"type": "json_object"}` requires the word "JSON" in the prompt — the translate prompt (Task 13) includes it. `_from_gemini` reads `resp.text`; for JSON-mode Gemini responses that is the concatenated JSON string.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/providers.py tests/test_providers.py
git commit -m "feat: Gemini/OpenAI/Anthropic clients with shared LLMResponse shape"
```

### Task 8: Silence parsing + chunk planner (`audio.py` part 1) — §5

Pure logic first: parse ffmpeg `silencedetect` stderr, then plan 10–14-minute chunks cut mid-silence with the −30→−25→−20 dB cascade and the hard-cut+overlap fallback.

**Files:**
- Create: `app/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:

```python
import pytest

from app.audio import (
    OVERLAP_SECONDS, PlannedChunk, Silence, parse_silences, plan_chunks,
)

SAMPLE_STDERR = """\
[mp3float @ 000001] Header missing
[silencedetect @ 000002] silence_start: 612.501
[silencedetect @ 000002] silence_end: 613.4 | silence_duration: 0.899
size=N/A time=00:14:00.00 bitrate=N/A speed= 980x
[silencedetect @ 000002] silence_start: 1305.2
[silencedetect @ 000002] silence_end: 1306.0 | silence_duration: 0.8
"""


def test_parse_silences_pairs_start_end():
    silences = parse_silences(SAMPLE_STDERR)
    assert silences == [Silence(612.501, 613.4), Silence(1305.2, 1306.0)]
    assert silences[0].duration == pytest.approx(0.899)
    assert silences[0].midpoint == pytest.approx(612.9505)


def test_parse_silences_ignores_unmatched_trailing_start():
    silences = parse_silences("[x] silence_start: 5.0\n")
    assert silences == []


def make_lookup(by_threshold):
    """dict {-30: [Silence...]} -> get_silences callable; missing -> []"""
    return lambda threshold: by_threshold.get(threshold, [])


def test_plan_single_chunk_for_short_audio():
    chunks = plan_chunks(700.0, make_lookup({}))
    assert chunks == [PlannedChunk(index=1, start=0.0, end=700.0,
                                   overlap_prev=False)]


def test_plan_cuts_at_longest_silence_in_window():
    silences = [
        Silence(300.0, 301.0),    # outside window (before min 10)
        Silence(650.0, 650.6),    # in window, 0.6s
        Silence(700.0, 701.2),    # in window, 1.2s  <- best
        Silence(900.0, 901.0),    # outside window (after min 14)
    ]
    chunks = plan_chunks(1400.0, make_lookup({-30: silences}))
    assert chunks[0].end == pytest.approx(700.6)   # midpoint of best silence
    assert chunks[1] == PlannedChunk(2, pytest.approx(700.6), 1400.0, False)


def test_plan_cascades_to_quieter_thresholds():
    chunks = plan_chunks(1400.0, make_lookup({
        -30: [],
        -25: [],
        -20: [Silence(720.0, 720.8)],
    }))
    assert chunks[0].end == pytest.approx(720.4)
    assert chunks[1].overlap_prev is False


def test_plan_hard_cut_with_overlap_when_no_silence():
    chunks = plan_chunks(1400.0, make_lookup({}))
    assert chunks[0] == PlannedChunk(1, 0.0, 840.0, False)
    # next chunk starts OVERLAP_SECONDS before the hard cut and is flagged
    assert chunks[1].start == pytest.approx(840.0 - OVERLAP_SECONDS)
    assert chunks[1].overlap_prev is True
    assert chunks[1].end == 1400.0


def test_plan_ignores_too_short_silences():
    chunks = plan_chunks(1400.0, make_lookup({-30: [Silence(700.0, 700.3)]}))
    assert chunks[0].end == 840.0  # 0.3s < 0.5s minimum -> hard cut


def test_plan_long_audio_produces_sequential_chunks():
    silences = [Silence(660.0, 661.0), Silence(1380.0, 1381.0)]
    chunks = plan_chunks(2000.0, make_lookup({-30: silences}))
    assert [c.index for c in chunks] == [1, 2, 3]
    assert chunks[0].end == chunks[1].start
    assert chunks[1].end == chunks[2].start
    assert chunks[2].end == 2000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audio'`.

- [ ] **Step 3: Write `app/audio.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/audio.py tests/test_audio.py
git commit -m "feat: silencedetect parsing and 10-14min chunk planner with cascade"
```

---

### Task 9: ffmpeg + yt-dlp runners (`audio.py` part 2) — §3, §5

Command **builders** are pure (tested); **runners** are thin subprocess wrappers (not unit-tested — exercised in the final smoke test).

**Files:**
- Modify: `app/audio.py` (append)
- Test: `tests/test_audio.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_audio.py`)

```python
from app.audio import (
    build_cut_args, build_normalize_args, build_silence_args, build_ytdlp_args,
)


def test_build_normalize_args_mono_16k_64kbps():
    args = build_normalize_args("in.mp4", "out.mp3")
    assert args == ["ffmpeg", "-y", "-i", "in.mp4", "-vn", "-ac", "1",
                    "-ar", "16000", "-b:a", "64k", "out.mp3"]


def test_build_silence_args_uses_threshold():
    args = build_silence_args("audio.mp3", -25)
    assert "-af" in args
    assert "silencedetect=noise=-25dB:d=0.5" in args
    assert args[-2:] == ["-f", "null"] or args[-3:] == ["-f", "null", "-"]


def test_build_cut_args_seeks_before_input_and_copies():
    args = build_cut_args("audio.mp3", "chunks/chunk_001.mp3", 10.0, 614.25)
    i_input = args.index("-i")
    assert args.index("-ss") < i_input          # fast seek
    assert "604.250" in args                    # -t duration, not -to
    assert "copy" in args


def test_build_ytdlp_args_audio_only_default():
    args = build_ytdlp_args("https://youtu.be/x", "out/source.%(ext)s",
                            save_mp4=False, cookies_file=None)
    assert "bestaudio/best" in args
    assert "--cookies" not in args


def test_build_ytdlp_args_mp4_and_cookies():
    args = build_ytdlp_args("https://youtu.be/x", "out/source.%(ext)s",
                            save_mp4=True, cookies_file="c.txt")
    assert "bestvideo*+bestaudio/best" in args
    assert "mp4" in args
    assert "--cookies" in args and "c.txt" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py -v`
Expected: new tests FAIL with ImportError.

- [ ] **Step 3: Implement** (append to `app/audio.py`; add `import subprocess`, `import sys`, `import shutil` to imports)

```python
def build_normalize_args(src: str, dst: str) -> list[str]:
    """Any input -> mp3 mono 16kHz 64kbps (§3: ~3-4x smaller uploads)."""
    return ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1",
            "-ar", "16000", "-b:a", "64k", str(dst)]


def build_silence_args(path: str, threshold_db: int) -> list[str]:
    return ["ffmpeg", "-i", str(path), "-af",
            f"silencedetect=noise={threshold_db}dB:d={MIN_SILENCE_DURATION}",
            "-f", "null", "-"]


def build_cut_args(src: str, dst: str, start: float, end: float) -> list[str]:
    # -ss before -i = fast input seek; stream copy keeps it instant.
    return ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(src), "-c", "copy", str(dst)]


def build_ytdlp_args(url: str, output_template: str, *, save_mp4: bool,
                     cookies_file: str | None) -> list[str]:
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist",
            "-o", output_template]
    if save_mp4:
        args += ["-f", "bestvideo*+bestaudio/best", "--merge-output-format", "mp4"]
    else:
        args += ["-f", "bestaudio/best"]
    if cookies_file:
        args += ["--cookies", cookies_file]
    args.append(url)
    return args


def _run(args: list[str], error_hint: str = "") -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise AudioError(f"Program tidak ditemukan: {args[0]}. "
                         "Pastikan ffmpeg ada di PATH.") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise AudioError(f"{args[0]} gagal (exit {proc.returncode}). "
                         f"{error_hint}\n{tail}")
    return proc


def ensure_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def normalize_audio(src: Path, dst: Path) -> None:
    _run(build_normalize_args(str(src), str(dst)))


def get_duration(path: Path) -> float:
    proc = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(proc.stdout.strip())


def detect_silences(path: Path, threshold_db: int) -> list[Silence]:
    # silencedetect logs to stderr; exit code 0 either way.
    proc = _run(build_silence_args(str(path), threshold_db))
    return parse_silences(proc.stderr)


def cut_chunk(src: Path, dst: Path, start: float, end: float) -> None:
    _run(build_cut_args(str(src), str(dst), start, end))


def split_audio(chunk_path: Path) -> tuple[Path, Path, float]:
    """Halve a chunk (for transcription-truncation recovery). Returns
    (left_path, right_path, right_offset_seconds)."""
    mid = get_duration(chunk_path) / 2
    left = chunk_path.with_name(chunk_path.stem + "_a.mp3")
    right = chunk_path.with_name(chunk_path.stem + "_b.mp3")
    cut_chunk(chunk_path, left, 0.0, mid)
    cut_chunk(chunk_path, right, mid, mid * 2 + 1.0)
    return left, right, mid


def download_youtube(url: str, job_dir: Path, *, save_mp4: bool,
                     cookies_file: str | None) -> Path:
    template = str(job_dir / "source.%(ext)s")
    args = build_ytdlp_args(url, template, save_mp4=save_mp4,
                            cookies_file=cookies_file)
    _run(args, error_hint=(
        "Jika YouTube berubah, perbarui yt-dlp: pip install -U yt-dlp."))
    matches = sorted(job_dir.glob("source.*"))
    if not matches:
        raise AudioError("yt-dlp selesai tetapi file source.* tidak ditemukan.")
    return matches[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/audio.py tests/test_audio.py
git commit -m "feat: ffmpeg/yt-dlp command builders and runners"
```

### Task 10: Transcript parsing + prompts (`transcribe.py` part 1) — §6a

**Files:**
- Create: `app/transcribe.py`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_transcribe.py`:

```python
import pytest

from app.transcribe import (
    TranscriptParseError, Utterance, build_transcribe_prompts,
    parse_timestamp, parse_transcript,
)


def test_parse_timestamp_mm_ss_mmm():
    assert parse_timestamp("01:23.450") == pytest.approx(83.45)


def test_parse_timestamp_h_mm_ss():
    assert parse_timestamp("1:02:03.5") == pytest.approx(3723.5)


def test_parse_timestamp_plain_seconds():
    assert parse_timestamp("45.2") == pytest.approx(45.2)


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("abc")


VALID = """[
  {"id": 1, "start": "00:01.000", "end": "00:02.500",
   "type": "dialogue", "ja": "こんにちは"},
  {"id": 2, "start": "00:03.000", "end": "00:04.000",
   "type": "narration", "ja": "今日の企画は"}
]"""


def test_parse_transcript_valid():
    utts = parse_transcript(VALID)
    assert utts == [
        Utterance(id=1, start=1.0, end=2.5, type="dialogue", ja="こんにちは"),
        Utterance(id=2, start=3.0, end=4.0, type="narration", ja="今日の企画は"),
    ]


def test_parse_transcript_strips_code_fences_and_wrapper_dict():
    fenced = '```json\n{"utterances": ' + VALID + '}\n```'
    assert len(parse_transcript(fenced)) == 2


def test_parse_transcript_skips_bad_items_and_normalizes():
    text = """[
      {"id": 1, "start": "00:01.0", "end": "00:00.5", "type": "vo", "ja": "あ"},
      {"id": 2, "start": "bad", "end": "00:04.0", "type": "dialogue", "ja": "い"},
      {"id": 3, "start": "00:05.0", "end": "00:06.0", "type": "dialogue", "ja": ""}
    ]"""
    utts = parse_transcript(text)
    assert len(utts) == 1                    # items 2 (bad time) & 3 (empty) skipped
    assert utts[0].type == "dialogue"        # unknown type normalized
    assert utts[0].end == pytest.approx(1.5) # end<=start clamped to start+0.5


def test_parse_transcript_raises_on_unparseable_json():
    with pytest.raises(TranscriptParseError):
        parse_transcript('[{"id": 1, "start": "00:01.0", "end": "00:0')  # truncated


def test_parse_transcript_raises_when_empty():
    with pytest.raises(TranscriptParseError):
        parse_transcript("[]")


def test_build_transcribe_prompts_injects_context():
    system, user = build_transcribe_prompts("CONTEXT-BODY", "MEMBERS-BODY")
    assert "CONTEXT-BODY" in system
    assert "MEMBERS-BODY" in system
    assert "MM:SS.mmm" in user
    assert "JSON" in user
    assert "narration" in user
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write `app/transcribe.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcript parsing, timestamp handling, transcription prompts"
```

---

### Task 11: Merge + overlap dedupe (`transcribe.py` part 2) — §5, §6a

Shift each chunk's utterances by its absolute offset; for hard-cut boundaries (`overlap_prev=True`), drop utterances at the start of the new chunk that duplicate the tail of the previous chunk (text similarity ≥ 0.8).

**Files:**
- Modify: `app/transcribe.py` (append)
- Test: `tests/test_transcribe.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_transcribe.py`)

```python
from app.audio import PlannedChunk
from app.transcribe import merge_transcripts


def U(id, start, end, ja, type="dialogue"):
    return Utterance(id=id, start=start, end=end, type=type, ja=ja)


def test_merge_shifts_by_offset_and_renumbers():
    chunks = [PlannedChunk(1, 0.0, 700.0, False),
              PlannedChunk(2, 700.0, 1400.0, False)]
    per_chunk = [
        [U(1, 1.0, 2.0, "あ"), U(2, 5.0, 6.0, "い")],
        [U(1, 0.5, 1.5, "う")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert [u.id for u in merged] == [1, 2, 3]
    assert merged[2].start == pytest.approx(700.5)
    assert merged[2].ja == "う"


def test_merge_dedupes_overlap_zone():
    # chunk 2 re-covers the last 4s of chunk 1 (hard cut at 840)
    chunks = [PlannedChunk(1, 0.0, 840.0, False),
              PlannedChunk(2, 836.0, 1400.0, True)]
    per_chunk = [
        [U(1, 830.0, 833.0, "今日は楽しかったですね")],
        [U(1, 0.2, 2.8, "今日は楽しかったですね。"),  # dupe (similar text)
         U(2, 10.0, 12.0, "次のコーナーです")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert [u.ja for u in merged] == ["今日は楽しかったですね", "次のコーナーです"]


def test_merge_keeps_different_text_in_overlap_zone():
    chunks = [PlannedChunk(1, 0.0, 840.0, False),
              PlannedChunk(2, 836.0, 1400.0, True)]
    per_chunk = [
        [U(1, 830.0, 833.0, "全然違う話")],
        [U(1, 0.2, 2.8, "新しいセリフです")],
    ]
    merged = merge_transcripts(per_chunk, chunks)
    assert len(merged) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: new tests FAIL with ImportError.

- [ ] **Step 3: Implement** (append to `app/transcribe.py`)

```python
SIMILARITY_THRESHOLD = 0.8
OVERLAP_MARGIN_SECONDS = 2.0


def _similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= SIMILARITY_THRESHOLD


def merge_transcripts(per_chunk: list[list[Utterance]],
                      chunks: list["audio.PlannedChunk"]) -> list[Utterance]:
    """Shift utterances to the absolute timeline; dedupe hard-cut overlaps."""
    merged: list[Utterance] = []
    for utterances, chunk in zip(per_chunk, chunks):
        shifted = [replace(u, start=u.start + chunk.start, end=u.end + chunk.start)
                   for u in utterances]
        if chunk.overlap_prev and merged:
            zone_end = chunk.start + audio.OVERLAP_SECONDS + OVERLAP_MARGIN_SECONDS
            tail = merged[-8:]
            shifted = [u for u in shifted
                       if not (u.start <= zone_end
                               and any(_similar(u.ja, t.ja) for t in tail))]
        merged.extend(shifted)
    return [replace(u, id=i) for i, u in enumerate(merged, start=1)]


def save_transcript(utterances: list[Utterance], path: Path) -> None:
    path.write_text(
        json.dumps([u.to_dict() for u in utterances], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_transcript(path: Path) -> list[Utterance]:
    return [Utterance.from_dict(d)
            for d in json.loads(path.read_text(encoding="utf-8"))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcript merging with absolute offsets and overlap dedupe"
```

---

### Task 12: Chunk transcription with truncation recovery (`transcribe.py` part 3) — §6a, §9

Upload chunk via Files API → generate → parse. If the output is truncated (MAX_TOKENS) or unparseable, split the chunk audio in half and transcribe each half (recursion, max depth 2).

**Files:**
- Modify: `app/transcribe.py` (append)
- Test: `tests/test_transcribe.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_transcribe.py`)

```python
from app.providers import LLMResponse
from app.transcribe import transcribe_chunk


class FakeGemini:
    """Returns queued LLMResponses; records uploads."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.uploads = []
        self.model = "gemini-2.5-pro"

    def upload_audio(self, path):
        self.uploads.append(path)
        return f"file-ref:{path}"

    def generate(self, system, user, audio=None):
        return self.responses.pop(0)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, input_tokens, output_tokens):
        self.calls.append((model, input_tokens, output_tokens))


GOOD = LLMResponse(text=VALID, input_tokens=100, output_tokens=50)
HALF = LLMResponse(
    text='[{"id":1,"start":"00:01.0","end":"00:02.0","type":"dialogue","ja":"あ"}]',
    input_tokens=10, output_tokens=5)


def test_transcribe_chunk_happy_path(tmp_path):
    gemini = FakeGemini([GOOD])
    tracker = FakeTracker()
    utts = transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                            "sys", "user", tracker)
    assert len(utts) == 2
    assert tracker.calls == [("gemini-2.5-pro", 100, 50)]
    assert gemini.uploads == [tmp_path / "chunk_001.mp3"]


def test_transcribe_chunk_splits_on_truncation(tmp_path, monkeypatch):
    truncated = LLMResponse(text=VALID, input_tokens=1, output_tokens=1,
                            truncated=True)
    gemini = FakeGemini([truncated, HALF, HALF])

    def fake_split(path):
        return (path.with_name(path.stem + "_a.mp3"),
                path.with_name(path.stem + "_b.mp3"), 420.0)

    monkeypatch.setattr("app.transcribe.audio.split_audio", fake_split)
    utts = transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                            "sys", "user", FakeTracker())
    assert len(utts) == 2
    assert utts[1].start == pytest.approx(421.0)  # second half shifted by 420
    assert len(gemini.uploads) == 3


def test_transcribe_chunk_gives_up_after_max_depth(tmp_path, monkeypatch):
    bad = LLMResponse(text="not json", input_tokens=1, output_tokens=1)
    gemini = FakeGemini([bad] * 7)
    monkeypatch.setattr(
        "app.transcribe.audio.split_audio",
        lambda p: (p.with_name(p.stem + "_a.mp3"),
                   p.with_name(p.stem + "_b.mp3"), 60.0))
    with pytest.raises(TranscriptParseError):
        transcribe_chunk(gemini, tmp_path / "chunk_001.mp3",
                         "sys", "user", FakeTracker())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: new tests FAIL with ImportError.

- [ ] **Step 3: Implement** (append to `app/transcribe.py`)

```python
MAX_SPLIT_DEPTH = 2


def transcribe_chunk(gemini, chunk_path: Path, system: str, user: str,
                     tracker, depth: int = 0) -> list[Utterance]:
    """Transcribe one audio chunk; timestamps relative to the chunk start."""
    audio_ref = gemini.upload_audio(chunk_path)
    response = call_with_retries(
        lambda: gemini.generate(system=system, user=user, audio=audio_ref))
    if tracker is not None:
        tracker.add(gemini.model, response.input_tokens, response.output_tokens)
    try:
        if response.truncated:
            raise TranscriptParseError("output terpotong (MAX_TOKENS)")
        return parse_transcript(response.text)
    except TranscriptParseError:
        if depth >= MAX_SPLIT_DEPTH:
            raise
        left, right, offset = audio.split_audio(chunk_path)
        first = transcribe_chunk(gemini, left, system, user, tracker, depth + 1)
        second = transcribe_chunk(gemini, right, system, user, tracker, depth + 1)
        shifted = [replace(u, start=u.start + offset, end=u.end + offset)
                   for u in second]
        combined = first + shifted
        return [replace(u, id=i) for i, u in enumerate(combined, start=1)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat: chunk transcription with truncation split recovery"
```

### Task 13: Translation with alignment validation (`translate.py`) — §6b, §9

Numbered-JSON batches; the model must echo the exact same ids and count with an added `id_text`. Validation failure → one retry carrying the error message → per-line fallback for the still-missing ids only.

**Files:**
- Create: `app/translate.py`
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_translate.py`:

```python
import json

import pytest

from app.providers import LLMResponse
from app.transcribe import Utterance
from app.translate import (
    TranslateError, build_translate_system, parse_translation,
    translate_batch, validate_alignment,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.model = "fake-flash"

    def generate(self, system, user, audio=None):
        self.prompts.append(user)
        return self.responses.pop(0)


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, i, o):
        self.calls.append((model, i, o))


def U(id, ja):
    return Utterance(id=id, start=float(id), end=float(id) + 1, type="dialogue",
                     ja=ja)


def R(items):
    return LLMResponse(text=json.dumps(items, ensure_ascii=False),
                       input_tokens=10, output_tokens=10)


UTTS = [U(1, "こんにちは"), U(2, "ありがとう")]
GOOD_ITEMS = [{"id": 1, "id_text": "Halo"}, {"id": 2, "id_text": "Makasih"}]


def test_parse_translation_tolerates_fences_and_wrapper():
    text = '```json\n{"items": [{"id": 1, "id_text": "Halo"}]}\n```'
    assert parse_translation(text) == {1: "Halo"}


def test_validate_alignment_passes_on_exact_match():
    assert validate_alignment({1: "a", 2: "b"}, [1, 2]) is None


def test_validate_alignment_reports_missing_and_extra():
    error = validate_alignment({1: "a", 3: "c"}, [1, 2])
    assert "2" in error and "3" in error


def test_translate_batch_happy_path():
    client = FakeClient([R(GOOD_ITEMS)])
    tracker = FakeTracker()
    result = translate_batch(client, UTTS, "SYSTEM", tracker)
    assert result == {1: "Halo", 2: "Makasih"}
    assert tracker.calls == [("fake-flash", 10, 10)]
    assert "JSON" in client.prompts[0]
    assert "こんにちは" in client.prompts[0]


def test_translate_batch_retries_with_error_feedback():
    bad = R([{"id": 1, "id_text": "Halo"}])  # missing id 2
    client = FakeClient([bad, R(GOOD_ITEMS)])
    result = translate_batch(client, UTTS, "SYSTEM", FakeTracker())
    assert result == {1: "Halo", 2: "Makasih"}
    assert len(client.prompts) == 2
    assert "id" in client.prompts[1]  # retry prompt carries the error


def test_translate_batch_falls_back_per_line_for_missing_ids():
    bad = R([{"id": 1, "id_text": "Halo"}])
    single = R([{"id": 2, "id_text": "Makasih"}])
    client = FakeClient([bad, bad, single])  # batch, retry, then per-line id 2
    result = translate_batch(client, UTTS, "SYSTEM", FakeTracker())
    assert result == {1: "Halo", 2: "Makasih"}
    assert len(client.prompts) == 3
    assert "ありがとう" in client.prompts[2]
    assert "こんにちは" not in client.prompts[2]  # only the missing line is resent


def test_translate_batch_raises_when_fallback_fails():
    garbage = LLMResponse(text="not json", input_tokens=1, output_tokens=1)
    client = FakeClient([garbage] * 6)
    with pytest.raises(TranslateError):
        translate_batch(client, UTTS, "SYSTEM", FakeTracker())


def test_build_translate_system_injects_context():
    system = build_translate_system("CTX", "MEMBERS")
    assert "CTX" in system and "MEMBERS" in system
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_translate.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write `app/translate.py`**

```python
"""JP -> ID translation in numbered JSON batches (§6b)."""
from __future__ import annotations

import json
import re

from app.providers import call_with_retries
from app.transcribe import Utterance

BATCH_SIZE = 100  # merged utterances per request (spec: "per chunk"; see plan notes)


class TranslateError(Exception):
    """Translation failed after retry + per-line fallback."""


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_translation(text: str) -> dict[int, str]:
    """Model output -> {id: id_text}. Tolerates fences and wrapper dicts."""
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("output bukan array JSON")
    result: dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item["id"])
            id_text = str(item["id_text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if id_text:
            result[item_id] = id_text
    return result


def validate_alignment(mapping: dict[int, str],
                       expected_ids: list[int]) -> str | None:
    """None when aligned; otherwise a human-readable error for the retry prompt."""
    expected = set(expected_ids)
    got = set(mapping)
    problems = []
    if missing := sorted(expected - got):
        problems.append(f"id hilang: {missing}")
    if extra := sorted(got - expected):
        problems.append(f"id tidak dikenal: {extra}")
    if len(mapping) != len(expected):
        problems.append(f"jumlah item {len(mapping)} != {len(expected)}")
    return "; ".join(problems) if problems else None


def build_translate_system(context_md: str, members_md: str) -> str:
    return (
        "Kamu adalah penerjemah subtitle JP->ID untuk variety show "
        "Soko Magattara, Sakurazaka?. Patuhi style guide, glossary, dan "
        "ejaan nama persis seperti referensi berikut.\n\n"
        "=== KONTEKS & STYLE GUIDE ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )


def _build_user_prompt(utterances: list[Utterance],
                       error_feedback: str | None = None) -> str:
    payload = json.dumps([{"id": u.id, "ja": u.ja} for u in utterances],
                         ensure_ascii=False)
    prompt = (
        "Terjemahkan tiap item berikut ke Bahasa Indonesia sesuai style guide.\n"
        "Balas HANYA array JSON dengan id yang SAMA PERSIS dan jumlah item "
        "yang SAMA PERSIS, format per item: "
        '{"id": <id sama>, "id_text": "<terjemahan>"}.\n'
        "Jangan menggabung, memecah, menambah, atau membuang item.\n\n" + payload
    )
    if error_feedback:
        prompt = (f"Output sebelumnya SALAH ({error_feedback}). Perbaiki dan "
                  f"ikuti instruksi dengan tepat.\n\n" + prompt)
    return prompt


def _request_mapping(client, utterances, system, tracker,
                     error_feedback=None) -> dict[int, str]:
    response = call_with_retries(
        lambda: client.generate(system=system,
                                user=_build_user_prompt(utterances,
                                                        error_feedback)))
    if tracker is not None:
        tracker.add(client.model, response.input_tokens, response.output_tokens)
    try:
        return parse_translation(response.text)
    except (json.JSONDecodeError, ValueError):
        return {}


def translate_batch(client, utterances: list[Utterance], system: str,
                    tracker) -> dict[int, str]:
    expected_ids = [u.id for u in utterances]

    mapping = _request_mapping(client, utterances, system, tracker)
    error = validate_alignment(mapping, expected_ids)
    if error is None:
        return mapping

    retry = _request_mapping(client, utterances, system, tracker,
                             error_feedback=error)
    if validate_alignment(retry, expected_ids) is None:
        return retry
    mapping = {**retry, **{k: v for k, v in mapping.items() if k not in retry}}
    mapping = {k: v for k, v in mapping.items() if k in set(expected_ids)}

    # Granular fallback: only ids that are still missing, one by one (§9).
    by_id = {u.id: u for u in utterances}
    for missing_id in sorted(set(expected_ids) - set(mapping)):
        single = _request_mapping(client, [by_id[missing_id]], system, tracker)
        if missing_id not in single:
            raise TranslateError(
                f"Gagal menerjemahkan id={missing_id} "
                f"({by_id[missing_id].ja[:40]}...) setelah fallback per-baris.")
        mapping[missing_id] = single[missing_id]
    return mapping


def translate_all(client, utterances: list[Utterance], system: str, tracker,
                  on_batch=None, load_checkpoint=None,
                  save_checkpoint=None) -> dict[int, str]:
    """Translate in BATCH_SIZE groups with optional per-batch checkpoints.

    load_checkpoint(batch_index) -> dict | None
    save_checkpoint(batch_index, mapping) -> None
    on_batch(done_count, total_count) -> None   (progress callback)
    """
    batches = [utterances[i:i + BATCH_SIZE]
               for i in range(0, len(utterances), BATCH_SIZE)]
    result: dict[int, str] = {}
    for idx, batch in enumerate(batches, start=1):
        mapping = load_checkpoint(idx) if load_checkpoint else None
        if mapping is None:
            mapping = translate_batch(client, batch, system, tracker)
            if save_checkpoint:
                save_checkpoint(idx, mapping)
        result.update(mapping)
        if on_batch:
            on_batch(idx, len(batches))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_translate.py -v`
Expected: 9 passed.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass (≈39 at this point).

- [ ] **Step 6: Commit**

```powershell
git add app/translate.py tests/test_translate.py
git commit -m "feat: batch translation with alignment validation and per-line fallback"
```

### Task 14: Deterministic post-processing (`subtitle.py` part 1) — §7

Backend rules, no LLM. Order matters: sort → merge short neighbors → extend remaining short events → split long events → fix overlaps → CPS flags.

**Files:**
- Create: `app/subtitle.py`
- Test: `tests/test_subtitle.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_subtitle.py`:

```python
import pytest

from app.subtitle import SubEvent, postprocess


def E(start, end, text, type="dialogue"):
    return SubEvent(start=start, end=end, text=text, type=type)


def run(events, **kw):
    defaults = dict(min_duration=0.7, max_duration=7.5, merge_gap=0.5,
                    cps_threshold=25.0)
    defaults.update(kw)
    return postprocess(events, **defaults)


def test_merges_short_reaction_with_continuation():
    # spec §7 example: "Aku senang sekali. Impianku jadi kenyataan."
    events, _ = run([E(0.0, 0.5, "Aku senang sekali."),
                     E(0.8, 2.5, "Impianku jadi kenyataan.")])
    assert len(events) == 1
    assert events[0].text == "Aku senang sekali. Impianku jadi kenyataan."
    assert events[0].start == 0.0 and events[0].end == 2.5


def test_does_not_merge_across_types():
    events, _ = run([E(0.0, 0.5, "Eh!?"),
                     E(0.8, 2.5, "Episode hari ini...", type="narration")])
    assert len(events) == 2


def test_does_not_merge_when_gap_too_large():
    events, _ = run([E(0.0, 0.5, "Eh!?"), E(1.2, 3.0, "Lanjut.")])
    assert len(events) == 2


def test_extends_short_event_to_min_duration():
    events, _ = run([E(0.0, 0.4, "Oke."), E(5.0, 7.0, "Lanjut.")])
    assert events[0].end == pytest.approx(0.7)


def test_extension_clamped_by_next_event():
    events, _ = run([E(0.0, 0.4, "Oke.", type="narration"),
                     E(0.6, 3.0, "Lanjut.")])  # different type: no merge
    assert events[0].end == pytest.approx(0.6)


def test_splits_long_event_at_sentence_pause():
    text = "Halo semua, selamat datang. Hari ini kita main game seru banget loh."
    events, _ = run([E(0.0, 10.0, text)])
    assert len(events) == 2
    assert events[0].text == "Halo semua, selamat datang."
    assert events[1].text == "Hari ini kita main game seru banget loh."
    assert events[0].end == pytest.approx(events[1].start)
    assert events[0].end == pytest.approx(10.0 * len("Halo semua, selamat datang.")
                                          / len(text.replace(" ", " ")), abs=0.6)
    assert events[1].end == pytest.approx(10.0)


def test_split_falls_back_to_space():
    text = "kata " * 40  # no punctuation, 200 chars
    events, _ = run([E(0.0, 9.0, text.strip())])
    assert len(events) >= 2
    assert all(e.duration <= 7.5 for e in events)


def test_unsplittable_long_event_left_alone():
    events, _ = run([E(0.0, 8.0, "a" * 50)])  # no punct, no space
    assert len(events) == 1


def test_overlap_shifts_next_block():
    events, _ = run([E(0.0, 5.0, "Pertama panjang sekali ya."),
                     E(4.0, 6.0, "Kedua juga lumayan.")])
    assert events[0].end == pytest.approx(5.0)
    assert events[1].start == pytest.approx(5.0)
    assert events[1].end == pytest.approx(6.0)


def test_cps_flags_fast_lines_without_cutting():
    fast = "x" * 100
    events, flags = run([E(0.0, 2.0, fast)])  # 50 cps
    assert events[0].text == fast              # never auto-cut (editorial call)
    assert len(flags) == 1
    assert flags[0]["cps"] == pytest.approx(50.0)
    assert flags[0]["index"] == 1


def test_events_sorted_by_start():
    events, _ = run([E(5.0, 6.0, "Kedua."), E(0.0, 1.0, "Pertama.")])
    assert [e.text for e in events] == ["Pertama.", "Kedua."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write `app/subtitle.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: 11 passed. (If `test_splits_long_event_at_sentence_pause` is off by the proportional-time tolerance, the assertion with `abs=0.6` allows for strip()-induced drift — do not loosen further; check the ratio math instead.)

- [ ] **Step 5: Commit**

```powershell
git add app/subtitle.py tests/test_subtitle.py
git commit -m "feat: deterministic subtitle post-processing per spec section 7"
```

---

### Task 15: Render `.ass`/`.srt` (`subtitle.py` part 2) — §8

**Files:**
- Modify: `app/subtitle.py` (append)
- Test: `tests/test_subtitle.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_subtitle.py`)

```python
from pathlib import Path

from app.subtitle import format_ass_time, format_srt_time, render_ass, render_srt

TEMPLATE = Path("context/template.ass").read_text(encoding="utf-8")


def test_format_ass_time():
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(3661.456) == "1:01:01.46"
    assert format_ass_time(59.999) == "0:01:00.00"  # centisecond carry


def test_format_srt_time():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(83.4567) == "00:01:23,457"
    assert format_srt_time(3600) == "01:00:00,000"


def test_render_ass_copies_template_header_and_appends_dialogue():
    events = [E(1.0, 2.5, "Halo semua!"),
              E(3.0, 4.0, "Episode kali ini...", type="narration")]
    out = render_ass(events, TEMPLATE)
    assert "[Script Info]" in out
    assert "PlayResX: 1920" in out
    assert "Style: Default,Comic Sans MS" in out
    lines = out.splitlines()
    fmt_i = next(i for i, l in enumerate(lines) if l.startswith("Format: Layer"))
    assert lines[fmt_i + 1] == \
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Halo semua!"
    assert lines[fmt_i + 2] == \
        "Dialogue: 0,0:00:03.00,0:00:04.00,Narrator,,0,0,0,,Episode kali ini..."


def test_render_ass_escapes_newlines():
    out = render_ass([E(0.0, 1.0, "baris satu\nbaris dua")], TEMPLATE)
    assert "baris satu\\Nbaris dua" in out


def test_render_srt_format():
    events = [E(1.0, 2.5, "Halo semua!"), E(3.0, 4.0, "Lanjut.")]
    assert render_srt(events) == (
        "1\n00:00:01,000 --> 00:00:02,500\nHalo semua!\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nLanjut.\n"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: new tests FAIL with ImportError. (They also require `context/template.ass` from Task 3 to exist.)

- [ ] **Step 3: Implement** (append to `app/subtitle.py`)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```powershell
git add app/subtitle.py tests/test_subtitle.py
git commit -m "feat: .ass/.srt rendering from HikaLeon template"
```

### Task 16: Job pipeline with checkpoints (`pipeline.py`) — §1, §9 layer 3

Orchestrates stages Download → Normalize → Chunk → Transcribe → Translate → Format. Every stage skips itself when its artifact already exists in `output/{job_id}/`, so a retry never redoes finished work (especially transcription, the expensive stage — it also checkpoints **per chunk**). Emits SSE-ready event dicts into `job.events`.

**Files:**
- Create: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
import json

import pytest

from app import pipeline
from app.transcribe import Utterance

FAKE_UTTS = [
    Utterance(id=1, start=1.0, end=2.0, type="dialogue", ja="こんにちは"),
    Utterance(id=2, start=3.0, end=4.0, type="narration", ja="次のコーナーです"),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated OUTPUT_DIR + all heavy externals faked. Yields a counters dict."""
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    pipeline.JOBS.clear()
    counters = {"download": 0, "normalize": 0, "transcribe": 0, "translate": 0}

    def fake_download(url, job_dir, *, save_mp4, cookies_file):
        counters["download"] += 1
        path = job_dir / ("source.mp4" if save_mp4 else "source.m4a")
        path.write_bytes(b"av-data")
        return path

    def fake_normalize(src, dst):
        counters["normalize"] += 1
        dst.write_bytes(b"mp3-data")

    def fake_cut(src, dst, start, end):
        dst.write_bytes(b"chunk-data")

    def fake_transcribe_chunk(gemini, chunk_path, system, user, tracker,
                              depth=0):
        counters["transcribe"] += 1
        return FAKE_UTTS

    def fake_translate_batch(client, utterances, system, tracker):
        counters["translate"] += 1
        return {u.id: f"ID-{u.id}" for u in utterances}

    monkeypatch.setattr("app.audio.download_youtube", fake_download)
    monkeypatch.setattr("app.audio.normalize_audio", fake_normalize)
    monkeypatch.setattr("app.audio.get_duration", lambda p: 700.0)
    monkeypatch.setattr("app.audio.detect_silences", lambda p, t: [])
    monkeypatch.setattr("app.audio.cut_chunk", fake_cut)
    monkeypatch.setattr("app.transcribe.transcribe_chunk", fake_transcribe_chunk)
    monkeypatch.setattr("app.translate.translate_batch", fake_translate_batch)
    monkeypatch.setattr("app.providers.make_transcriber", lambda cfg: object())
    monkeypatch.setattr("app.providers.make_translator",
                        lambda cfg, p: object())
    return counters


def make_params(**overrides):
    params = {"source": "url", "url": "https://youtu.be/x", "save_mp4": False,
              "translator": "gemini", "output_format": "both",
              "original_filename": None}
    params.update(overrides)
    return params


def test_full_url_job_produces_artifacts_and_events(env):
    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)

    assert job.status == "done"
    for name in ("audio.mp3", "offsets.json", "transcript_jp.json",
                 "translated_id.json", "result.ass", "result.srt",
                 "flags.json", "job.json"):
        assert (job.dir / name).exists(), name

    stages = [e["stage"] for e in job.events]
    for stage in ("download", "normalize", "chunk", "transcribe",
                  "translate", "format", "done"):
        assert stage in stages
    assert job.events[-1]["stage"] == "done"
    assert "usage" in job.events[-1]["result"]
    assert "result.ass" in job.events[-1]["result"]["files"]

    translated = json.loads((job.dir / "translated_id.json")
                            .read_text(encoding="utf-8"))
    assert translated[0]["id_text"] == "ID-1"
    ass_text = (job.dir / "result.ass").read_text(encoding="utf-8")
    assert "ID-1" in ass_text and "Narrator" in ass_text


def test_uploaded_file_job_skips_download(env):
    job = pipeline.create_job(make_params(source="file", url=None,
                                          original_filename="ep288.mp4"),
                              upload_bytes=b"vid", upload_filename="ep288.mp4")
    assert (job.dir / "source.mp4").read_bytes() == b"vid"
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert env["download"] == 0


def test_checkpoints_skip_completed_stages(env):
    job = pipeline.create_job(make_params(output_format="ass"))
    pipeline.run_job(job.id)
    assert env["transcribe"] == 1 and env["translate"] == 1

    # Simulate a later-stage redo: delete translation + result, keep transcript
    (job.dir / "translated_id.json").unlink()
    for f in job.dir.glob("chunks/translated_*.json"):
        f.unlink()
    (job.dir / "result.ass").unlink()
    job.events.clear()
    pipeline.run_job(job.id)

    assert job.status == "done"
    assert env["transcribe"] == 1          # §9: transcription never repeated
    assert env["translate"] == 2
    assert (job.dir / "result.ass").exists()


def test_failure_marks_job_and_retry_resumes(env, monkeypatch):
    def boom(client, utterances, system, tracker):
        raise RuntimeError("translasi meledak")

    monkeypatch.setattr("app.translate.translate_batch", boom)
    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert "translasi meledak" in job.error
    assert job.events[-1]["status"] == "error"

    # retry with translation fixed
    monkeypatch.setattr("app.translate.translate_batch",
                        lambda c, u, s, t: {x.id: "OK" for x in u})
    job.events.clear()
    job.error = None
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert env["transcribe"] == 1  # resumed from checkpoint, not re-transcribed


def test_load_job_from_disk(env):
    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    pipeline.JOBS.clear()
    loaded = pipeline.load_job(job.id)
    assert loaded is not None
    assert loaded.params["url"] == "https://youtu.be/x"
    assert loaded.status == "done"
    assert pipeline.load_job("nonexistent") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Write `app/pipeline.py`**

```python
"""Job orchestration: stages, checkpoints, SSE-ready events (§1, §9 layer 3).

Every stage skips itself when its artifact already exists in output/{job_id}/,
so run_job() after a failure resumes instead of redoing work. Transcription
additionally checkpoints per chunk and translation per batch.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app import audio, providers, subtitle, transcribe, translate
from app.providers import UsageTracker
from app.subtitle import SubEvent

OUTPUT_DIR = Path("output")
CONTEXT_DIR = Path("context")
JOBS: dict[str, "Job"] = {}


class PipelineError(Exception):
    """Job cannot proceed (bad params, missing source file, ...)."""


@dataclass
class Job:
    id: str
    dir: Path
    params: dict
    status: str = "queued"  # queued | running | done | failed
    error: str | None = None
    events: list[dict] = field(default_factory=list)


def create_job(params: dict, upload_bytes: bytes | None = None,
               upload_filename: str | None = None) -> Job:
    job_id = (datetime.now().strftime("%Y%m%d-%H%M%S")
              + "-" + uuid.uuid4().hex[:6])
    job_dir = OUTPUT_DIR / job_id
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, dir=job_dir, params=params)
    if upload_bytes is not None:
        suffix = Path(upload_filename or "source.bin").suffix or ".bin"
        (job_dir / f"source{suffix}").write_bytes(upload_bytes)
    JOBS[job_id] = job
    _save(job)
    return job


def load_job(job_id: str) -> Job | None:
    meta_path = OUTPUT_DIR / job_id / "job.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job = Job(id=meta["id"], dir=OUTPUT_DIR / job_id, params=meta["params"],
              status=meta["status"], error=meta.get("error"),
              events=meta.get("events", []))
    JOBS[job.id] = job
    return job


def _save(job: Job) -> None:
    payload = {"id": job.id, "params": job.params, "status": job.status,
               "error": job.error, "events": job.events}
    (job.dir / "job.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit(job: Job, stage: str, status: str, message: str | None = None,
          result: dict | None = None) -> None:
    event: dict = {"stage": stage, "status": status}
    if message:
        event["message"] = message
    if result is not None:
        event["result"] = result
    job.events.append(event)
    _save(job)


def run_job(job_id: str) -> None:
    job = JOBS.get(job_id) or load_job(job_id)
    if job is None:
        raise KeyError(f"Job tidak ditemukan: {job_id}")
    cfg = providers.Config.load()
    tracker = UsageTracker(job.dir / "usage.json")
    job.status, job.error = "running", None
    _save(job)
    stage = "init"
    try:
        stage = "download"
        source = _stage_download(job, cfg)
        stage = "normalize"
        audio_path = _stage_normalize(job, source)
        stage = "chunk"
        chunks = _stage_chunk(job, audio_path)
        stage = "transcribe"
        utterances = _stage_transcribe(job, cfg, tracker, chunks)
        stage = "translate"
        rows = _stage_translate(job, cfg, tracker, utterances)
        stage = "format"
        files = _stage_format(job, cfg, rows)
        job.status = "done"
        _emit(job, "done", "done",
              result={"usage": tracker.summary(), "files": files})
    except Exception as exc:  # noqa: BLE001 — any stage error fails the job
        job.status = "failed"
        job.error = str(exc)
        _emit(job, stage, "error", message=str(exc))


def _read_context() -> tuple[str, str]:
    # §4: hot-reload per job — never cached across jobs.
    context_md = (CONTEXT_DIR / "context.md").read_text(encoding="utf-8")
    members_md = (CONTEXT_DIR / "members.md").read_text(encoding="utf-8")
    return context_md, members_md


def _find_source(job: Job) -> Path | None:
    matches = sorted(job.dir.glob("source.*"))
    return matches[0] if matches else None


def _stage_download(job: Job, cfg) -> Path:
    existing = _find_source(job)
    if existing is not None:
        _emit(job, "download", "done",
              f"{existing.name} sudah ada (checkpoint)")
        return existing
    if job.params.get("source") != "url" or not job.params.get("url"):
        raise PipelineError("File sumber tidak ditemukan dan tidak ada URL.")
    _emit(job, "download", "start", "mengunduh dari YouTube...")
    path = audio.download_youtube(
        job.params["url"], job.dir,
        save_mp4=bool(job.params.get("save_mp4", False)),
        cookies_file=cfg.ytdlp_cookies_file)
    _emit(job, "download", "done", path.name)
    return path


def _stage_normalize(job: Job, source: Path) -> Path:
    dst = job.dir / "audio.mp3"
    if dst.exists():
        _emit(job, "normalize", "done", "audio.mp3 sudah ada (checkpoint)")
        return dst
    _emit(job, "normalize", "start",
          "normalisasi audio (mp3 mono 16kHz 64kbps)")
    audio.normalize_audio(source, dst)
    _emit(job, "normalize", "done", "audio.mp3")
    return dst


def _stage_chunk(job: Job, audio_path: Path) -> list[audio.PlannedChunk]:
    offsets_path = job.dir / "offsets.json"
    chunks_dir = job.dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    if offsets_path.exists():
        chunks = audio.load_offsets(offsets_path)
        _emit(job, "chunk", "done", f"{len(chunks)} chunk (checkpoint)")
    else:
        _emit(job, "chunk", "start",
              "deteksi keheningan & perencanaan chunk")
        duration = audio.get_duration(audio_path)
        silence_cache: dict[int, list] = {}

        def get_silences(threshold_db: int):
            if threshold_db not in silence_cache:
                silence_cache[threshold_db] = audio.detect_silences(
                    audio_path, threshold_db)
            return silence_cache[threshold_db]

        chunks = audio.plan_chunks(duration, get_silences)
        audio.write_offsets(chunks, offsets_path)
        _emit(job, "chunk", "done", f"{len(chunks)} chunk")
    for c in chunks:
        dst = chunks_dir / f"chunk_{c.index:03d}.mp3"
        if not dst.exists():
            audio.cut_chunk(audio_path, dst, c.start, c.end)
    return chunks


def _stage_transcribe(job: Job, cfg, tracker,
                      chunks: list[audio.PlannedChunk]
                      ) -> list[transcribe.Utterance]:
    final_path = job.dir / "transcript_jp.json"
    if final_path.exists():
        utterances = transcribe.load_transcript(final_path)
        _emit(job, "transcribe", "done",
              f"{len(utterances)} ujaran (checkpoint)")
        return utterances
    context_md, members_md = _read_context()
    system, user = transcribe.build_transcribe_prompts(context_md, members_md)
    gemini = providers.make_transcriber(cfg)
    _emit(job, "transcribe", "start", f"transkripsi {len(chunks)} chunk")
    per_chunk: list[list[transcribe.Utterance]] = []
    for c in chunks:
        ckpt = job.dir / "chunks" / f"transcript_{c.index:03d}.json"
        if ckpt.exists():
            utts = transcribe.load_transcript(ckpt)
        else:
            chunk_path = job.dir / "chunks" / f"chunk_{c.index:03d}.mp3"
            utts = transcribe.transcribe_chunk(gemini, chunk_path, system,
                                               user, tracker)
            transcribe.save_transcript(utts, ckpt)
        per_chunk.append(utts)
        _emit(job, "transcribe", "progress",
              f"chunk {c.index}/{len(chunks)} selesai")
    merged = transcribe.merge_transcripts(per_chunk, chunks)
    transcribe.save_transcript(merged, final_path)
    _emit(job, "transcribe", "done", f"{len(merged)} ujaran")
    return merged


def _stage_translate(job: Job, cfg, tracker,
                     utterances: list[transcribe.Utterance]) -> list[dict]:
    out_path = job.dir / "translated_id.json"
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        _emit(job, "translate", "done", f"{len(rows)} baris (checkpoint)")
        return rows
    context_md, members_md = _read_context()
    system = translate.build_translate_system(context_md, members_md)
    client = providers.make_translator(cfg, job.params["translator"])
    _emit(job, "translate", "start",
          f"terjemahkan {len(utterances)} ujaran via "
          f"{job.params['translator']}")

    def load_ckpt(batch_index: int) -> dict[int, str] | None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        if not p.exists():
            return None
        return {int(k): v
                for k, v in json.loads(p.read_text(encoding="utf-8")).items()}

    def save_ckpt(batch_index: int, mapping: dict[int, str]) -> None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def on_batch(done: int, total: int) -> None:
        _emit(job, "translate", "progress", f"batch {done}/{total}")

    mapping = translate.translate_all(client, utterances, system, tracker,
                                      on_batch=on_batch,
                                      load_checkpoint=load_ckpt,
                                      save_checkpoint=save_ckpt)
    rows = [{**u.to_dict(), "id_text": mapping[u.id]} for u in utterances]
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    _emit(job, "translate", "done", f"{len(rows)} baris")
    return rows


def _stage_format(job: Job, cfg, rows: list[dict]) -> list[str]:
    fmt = job.params.get("output_format", "both")
    _emit(job, "format", "start", f"post-processing & render ({fmt})")
    events = [SubEvent(start=float(r["start"]), end=float(r["end"]),
                       text=r["id_text"], type=r.get("type", "dialogue"))
              for r in rows]
    processed, flags = subtitle.postprocess(
        events, min_duration=cfg.sub_min_duration,
        max_duration=cfg.sub_max_duration, merge_gap=cfg.sub_merge_gap,
        cps_threshold=cfg.sub_cps_flag)
    (job.dir / "flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    files = ["transcript_jp.json", "flags.json"]
    if fmt in ("ass", "both"):
        template = (CONTEXT_DIR / "template.ass").read_text(encoding="utf-8")
        (job.dir / "result.ass").write_text(
            subtitle.render_ass(processed, template), encoding="utf-8")
        files.append("result.ass")
    if fmt in ("srt", "both"):
        (job.dir / "result.srt").write_text(
            subtitle.render_srt(processed), encoding="utf-8")
        files.append("result.srt")
    if (job.dir / "source.mp4").exists():
        files.append("source.mp4")
    _emit(job, "format", "done",
          ", ".join(f for f in files if f.startswith("result")))
    return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 5 passed.

Note: these tests exercise checkpoint semantics heavily. If `test_checkpoints_skip_completed_stages` fails with `env["translate"] == 1` instead of 2, the per-batch checkpoint files in `chunks/translated_*.json` were not deleted by the orchestration path you wrote — check that `_stage_translate` only short-circuits on `translated_id.json` and that `load_ckpt` returns `None` for missing files.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: 76 passed.

- [ ] **Step 6: Commit**

```powershell
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: job pipeline with per-stage checkpoints and SSE-ready events"
```

---

### Task 17: FastAPI app (`main.py`) — §1, §3, §10

HTTP layer only — all logic lives in `pipeline.py`. Endpoints: serve the UI, report config/provider availability, create jobs (URL or upload), stream progress as SSE, download artifacts (allowlisted), retry failed jobs.

**Files:**
- Create: `app/main.py`
- Create: `static/index.html` (placeholder — Task 18 writes the real UI)
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the placeholder `static/index.html`**

The `GET /` route reads this file; Task 18 replaces it with the real UI.

```html
<!doctype html>
<html lang="id">
<head><meta charset="utf-8"><title>Sokomagattara SubGen</title></head>
<body><p>UI belum dibangun (Task 18).</p></body>
</html>
```

- [ ] **Step 2: Write the failing tests**

`tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app import main, pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    pipeline.JOBS.clear()
    return TestClient(main.app)


@pytest.fixture
def fake_run(monkeypatch):
    """pipeline.run_job replaced by an instant success; returns call list."""
    calls = []

    def run(job_id):
        calls.append(job_id)
        job = pipeline.JOBS[job_id]
        job.status = "done"
        job.events.append({
            "stage": "done", "status": "done",
            "result": {"usage": {"line": "Tokens: 0 in / 0 out (~$0.00)"},
                       "files": ["result.ass", "result.srt"]},
        })

    monkeypatch.setattr(pipeline, "run_job", run)
    return calls


def post_url_job(client, **overrides):
    data = {"source": "url", "url": "https://youtu.be/x",
            "translator": "gemini", "output_format": "both"}
    data.update(overrides)
    return client.post("/api/jobs", data=data)


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_config_reports_providers(client, monkeypatch):
    monkeypatch.setattr("app.audio.ensure_ffmpeg", lambda: True)
    body = client.get("/api/config").json()
    assert body["ok"] is True
    assert body["providers"]["gemini"] is True
    assert body["ffmpeg"] is True


def test_create_url_job_runs_pipeline(client, fake_run):
    resp = post_url_job(client)
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert fake_run == [job_id]
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done"


def test_create_file_job_saves_upload(client, fake_run):
    resp = client.post(
        "/api/jobs",
        data={"source": "file", "translator": "gemini",
              "output_format": "ass"},
        files={"file": ("ep288.mp4", b"vid-bytes", "video/mp4")})
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert (job.dir / "source.mp4").read_bytes() == b"vid-bytes"


def test_url_job_requires_url(client):
    resp = post_url_job(client, url="")
    assert resp.status_code == 422


def test_file_job_requires_file(client):
    resp = client.post("/api/jobs",
                       data={"source": "file", "translator": "gemini"})
    assert resp.status_code == 422


def test_translator_without_key_rejected(client, monkeypatch):
    # Prevent a local .env from re-supplying the key under test.
    monkeypatch.setattr("app.providers.load_dotenv", lambda **kw: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp = post_url_job(client, translator="openai")
    assert resp.status_code == 422


def test_sse_streams_events_then_end(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    resp = client.get(f"/api/jobs/{job_id}/events")
    assert resp.status_code == 200
    assert '"stage": "done"' in resp.text
    assert "event: end" in resp.text


def test_download_allowlist(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    (job.dir / "result.ass").write_text("[Script Info]\n", encoding="utf-8")
    ok = client.get(f"/api/jobs/{job_id}/files/result.ass")
    assert ok.status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/job.json").status_code == 404
    assert client.get(
        f"/api/jobs/{job_id}/files/result.srt").status_code == 404


def test_retry_requeues_job(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert fake_run == [job_id, job_id]


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
```

Notes for the implementer:
- `TestClient` executes `BackgroundTasks` synchronously before the response returns, so `fake_run` has already run when the POST comes back.
- The SSE test works on a finished job: the stream flushes every stored event, sends `event: end`, and closes — so `client.get()` terminates.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL — `ImportError: cannot import name 'main' from 'app'`.

- [ ] **Step 4: Write `app/main.py`**

```python
"""FastAPI app: UI, job API, SSE progress, artifact downloads (§1, §3)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app import audio, pipeline, providers

app = FastAPI(title="Sokomagattara SubGen")

STATIC_DIR = Path("static")
# Allowlist: never serve job.json (contains params) or arbitrary paths.
DOWNLOADABLE = {"result.ass", "result.srt", "transcript_jp.json",
                "translated_id.json", "flags.json", "usage.json",
                "source.mp4"}
VALID_FORMATS = {"ass", "srt", "both"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


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
    }


def _get_job_or_404(job_id: str) -> pipeline.Job:
    job = pipeline.JOBS.get(job_id) or pipeline.load_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan.")
    return job


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     source: str = Form(...),
                     url: str = Form(""),
                     translator: str = Form("gemini"),
                     output_format: str = Form("both"),
                     save_mp4: bool = Form(False),
                     file: UploadFile | None = File(None)) -> dict:
    if output_format not in VALID_FORMATS:
        raise HTTPException(422, f"output_format tidak dikenal: {output_format!r}")
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
              "save_mp4": save_mp4, "original_filename": upload_name}
    job = pipeline.create_job(params, upload_bytes=upload_bytes,
                              upload_filename=upload_name)
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    return {"id": job.id, "status": job.status, "error": job.error,
            "params": job.params, "events": job.events}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = _get_job_or_404(job_id)

    async def stream():
        idx = 0
        while True:
            while idx < len(job.events):
                payload = json.dumps(job.events[idx], ensure_ascii=False)
                yield f"data: {payload}\n\n"
                idx += 1
            if job.status in ("done", "failed"):
                end = json.dumps({"status": job.status, "error": job.error})
                yield f"event: end\ndata: {end}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/files/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if filename not in DOWNLOADABLE:
        raise HTTPException(404, "File tidak tersedia untuk diunduh.")
    path = job.dir / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} belum ada.")
    return FileResponse(path, filename=f"{job.id}_{filename}")


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str, background_tasks: BackgroundTasks) -> dict:
    job = _get_job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(409, "Job masih berjalan.")
    job.status = "queued"
    job.error = None
    job.events.clear()
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id, "status": "queued"}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```powershell
git add app/main.py static/index.html tests/test_api.py
git commit -m "feat: FastAPI endpoints with SSE progress and allowlisted downloads"
```

---

### Task 18: Frontend (`static/index.html`) — §1, §3, §10

The entire UI in one static file: source picker (URL/upload), translator dropdown filled from `/api/config`, output format, live progress log over `EventSource`, download links, usage line, and a checkpoint-aware Retry button. No build step, no frameworks.

**Files:**
- Modify: `static/index.html` (replace the Task 17 placeholder entirely)

- [ ] **Step 1: Replace `static/index.html` with the full UI**

```html
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>Sokomagattara SubGen</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#14151a; --card:#1e2027; --text:#e8e8ea; --muted:#9aa0ae;
          --accent:#f2a7c3; --err:#ff6b6b; --ok:#7bd88f; }
  * { box-sizing: border-box; }
  body { background:var(--bg); color:var(--text); margin:0;
         font-family:"Segoe UI",system-ui,sans-serif; }
  main { max-width:760px; margin:0 auto; padding:24px 16px 64px; }
  h1 { font-size:1.5rem; margin-bottom:4px; } h1 span { color:var(--accent); }
  .muted { color:var(--muted); margin-top:0; }
  .card { background:var(--card); border-radius:10px; padding:20px;
          margin-top:16px; }
  label { display:block; margin:12px 0 4px; color:var(--muted);
          font-size:.9rem; }
  .row { display:flex; gap:24px; align-items:center; }
  .row label { display:flex; gap:6px; align-items:center; margin:0; }
  input[type=text], select { width:100%; padding:8px 10px; border-radius:6px;
    border:1px solid #333; background:#101116; color:var(--text); }
  button { background:var(--accent); color:#222; font-weight:600; border:none;
    border-radius:6px; padding:10px 22px; cursor:pointer; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  #log { font-family:Consolas,monospace; font-size:.85rem;
    white-space:pre-wrap; max-height:320px; overflow-y:auto; }
  .error { color:var(--err); } .okline { color:var(--ok); }
  .banner { background:#3a1f27; border:1px solid var(--err);
    border-radius:8px; padding:12px; margin-top:16px; display:none; }
  #downloads a { display:inline-block; margin:6px 12px 0 0;
    color:var(--accent); }
  .hidden { display:none; }
</style>
</head>
<body>
<main>
  <h1>Sokomagattara <span>SubGen</span></h1>
  <p class="muted">Transkripsi JP → terjemahan ID → subtitle .ass / .srt</p>
  <div id="config-banner" class="banner"></div>

  <form id="job-form" class="card">
    <div class="row">
      <label><input type="radio" name="source" value="url" checked>
        YouTube URL</label>
      <label><input type="radio" name="source" value="file">
        Upload file</label>
    </div>
    <div id="url-box">
      <label for="url">URL YouTube</label>
      <input type="text" id="url"
             placeholder="https://www.youtube.com/watch?v=...">
      <label><input type="checkbox" id="save_mp4">
        Simpan video .mp4 (untuk ditonton dengan subtitle)</label>
    </div>
    <div id="file-box" class="hidden">
      <label for="file">File audio/video (mp3, m4a, wav, mp4, mkv)</label>
      <input type="file" id="file" accept=".mp3,.m4a,.wav,.mp4,.mkv">
    </div>
    <label for="translator">Model translasi</label>
    <select id="translator"></select>
    <label for="output_format">Format output</label>
    <select id="output_format">
      <option value="both">.ass + .srt</option>
      <option value="ass">.ass saja</option>
      <option value="srt">.srt saja</option>
    </select>
    <p><button type="submit" id="start-btn">Mulai</button></p>
  </form>

  <div class="card hidden" id="progress-card">
    <h2>Progress</h2>
    <div id="log"></div>
    <p id="usage" class="okline"></p>
    <div id="downloads"></div>
    <p><button id="retry-btn" class="hidden" type="button">
      Coba lagi (lanjut dari checkpoint)</button></p>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let jobId = null;
let es = null;

const PROVIDER_LABELS = { gemini: "Gemini", openai: "OpenAI",
                          anthropic: "Anthropic Claude" };

function showBanner(msg) {
  const b = $("config-banner");
  b.textContent = msg;
  b.style.display = "block";
}

async function loadConfig() {
  let cfg;
  try {
    cfg = await (await fetch("/api/config")).json();
  } catch (e) {
    showBanner("Tidak bisa menghubungi server: " + e);
    $("start-btn").disabled = true;
    return;
  }
  if (!cfg.ok) {
    showBanner(cfg.error);
    $("start-btn").disabled = true;
    return;
  }
  if (!cfg.ffmpeg) {
    showBanner("ffmpeg/ffprobe tidak ditemukan di PATH. " +
               "Install: winget install Gyan.FFmpeg lalu restart terminal.");
    $("start-btn").disabled = true;
  }
  const sel = $("translator");
  for (const [prov, available] of Object.entries(cfg.providers)) {
    if (!available) continue;
    const opt = document.createElement("option");
    opt.value = prov;
    opt.textContent = (PROVIDER_LABELS[prov] || prov) +
                      " — " + cfg.translate_models[prov];
    sel.appendChild(opt);
  }
}

function logLine(text, cls) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = text;
  $("log").appendChild(div);
  $("log").scrollTop = $("log").scrollHeight;
}

function resetProgress() {
  $("progress-card").classList.remove("hidden");
  $("log").innerHTML = "";
  $("usage").textContent = "";
  $("downloads").innerHTML = "";
  $("retry-btn").classList.add("hidden");
}

function finish(result) {
  $("usage").textContent = result.usage.line || "";
  for (const name of result.files) {
    const a = document.createElement("a");
    a.href = "/api/jobs/" + jobId + "/files/" + name;
    a.textContent = "⬇ " + name;
    $("downloads").appendChild(a);
  }
}

function listen(id) {
  if (es) es.close();
  es = new EventSource("/api/jobs/" + id + "/events");
  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.status === "error") {
      logLine("[" + ev.stage + "] GAGAL: " + (ev.message || ""), "error");
      $("retry-btn").classList.remove("hidden");
      return;
    }
    logLine("[" + ev.stage + "] " + (ev.message || ev.status));
    if (ev.stage === "done" && ev.result) finish(ev.result);
  };
  es.addEventListener("end", () => es.close());
  es.onerror = () => {};  // the server closes the stream when the job ends
}

$("job-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const source =
    document.querySelector('input[name="source"]:checked').value;
  const fd = new FormData();
  fd.append("source", source);
  fd.append("translator", $("translator").value);
  fd.append("output_format", $("output_format").value);
  if (source === "url") {
    fd.append("url", $("url").value.trim());
    fd.append("save_mp4", $("save_mp4").checked);
  } else {
    if (!$("file").files[0]) { alert("Pilih file dulu."); return; }
    fd.append("file", $("file").files[0]);
  }
  const resp = await fetch("/api/jobs", { method: "POST", body: fd });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert(err.detail || ("Gagal membuat job (HTTP " + resp.status + ")"));
    return;
  }
  jobId = (await resp.json()).job_id;
  resetProgress();
  logLine("Job " + jobId + " dimulai...");
  listen(jobId);
});

$("retry-btn").addEventListener("click", async () => {
  const resp = await fetch("/api/jobs/" + jobId + "/retry",
                           { method: "POST" });
  if (!resp.ok) { alert("Gagal retry."); return; }
  resetProgress();
  logLine("Job " + jobId + " diulang dari checkpoint...");
  listen(jobId);
});

for (const radio of document.querySelectorAll('input[name="source"]')) {
  radio.addEventListener("change", () => {
    const isUrl =
      document.querySelector('input[name="source"]:checked').value === "url";
    $("url-box").classList.toggle("hidden", !isUrl);
    $("file-box").classList.toggle("hidden", isUrl);
  });
}

loadConfig();
</script>
</body>
</html>
```

- [ ] **Step 2: Run the whole suite (UI must not break `GET /`)**

Run: `python -m pytest -q`
Expected: 87 passed.

- [ ] **Step 3: Manual smoke test**

```powershell
python -m uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000` in a browser and verify:
- The page loads with the form visible.
- With a valid `.env`, no red banner appears and the translator dropdown lists at least "Gemini — gemini-2.5-flash".
- Without ffmpeg on PATH (or with `GEMINI_API_KEY` unset), the red banner appears and the Mulai button is disabled.
- Switching the radio to "Upload file" hides the URL box and shows the file picker.

Stop the server with Ctrl+C when done. (A full end-to-end run with a real episode costs ~$0.20–$0.50 and is left to the user.)

- [ ] **Step 4: Commit**

```powershell
git add static/index.html
git commit -m "feat: single-file frontend with SSE progress and retry"
```

---

### Task 19: README + final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Sokomagattara SubGen

Aplikasi localhost untuk membuat subtitle Bahasa Indonesia (.ass / .srt)
dari episode *Soko Magattara, Sakurazaka?* — transkripsi Jepang via
Gemini 2.5 Pro, terjemahan via Gemini 2.5 Flash (atau GPT-4o / Claude).

## Persiapan

1. **Python 3.11+** dan **ffmpeg** harus terpasang:
   ```powershell
   winget install Gyan.FFmpeg   # lalu restart terminal
   ```
2. Install dependensi:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Salin `.env.example` menjadi `.env` lalu isi `GEMINI_API_KEY`
   (wajib). `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` opsional —
   provider tanpa key tidak muncul di dropdown.

## Menjalankan

```powershell
python -m uvicorn app.main:app --port 8000
```

Buka <http://127.0.0.1:8000>, pilih sumber (URL YouTube atau upload
mp3/m4a/wav/mp4/mkv), model translasi, dan format output, lalu klik
**Mulai**. Progress tampil live; setelah selesai muncul link unduhan
dan ringkasan biaya token (estimasi ~$0.20–$0.50 per episode).

## Output per job (`output/{job_id}/`)

| File | Isi |
|---|---|
| `result.ass` / `result.srt` | Subtitle final (style HikaLeon dari `context/template.ass`) |
| `transcript_jp.json` | Transkrip Jepang per ujaran (id, start, end, type, ja) |
| `translated_id.json` | Transkrip + terjemahan Indonesia |
| `flags.json` | Baris dengan CPS > 25 untuk dicek manual (tidak dipotong otomatis) |
| `usage.json` | Akumulasi token & estimasi biaya |
| `source.mp4` | Video sumber (hanya jika "Simpan video" dicentang) |

## Jika gagal di tengah jalan

Klik **Coba lagi** di UI. Setiap tahap (download, normalisasi, chunking,
transkripsi per chunk, translasi per batch) di-checkpoint ke disk —
retry melanjutkan dari tahap yang gagal dan **tidak pernah mengulang
transkripsi yang sudah selesai** (tahap paling mahal).

Jika download YouTube gagal karena yt-dlp usang:
```powershell
pip install -U yt-dlp
```

## Konteks terjemahan

`context/context.md` (style guide + glossary) dan `context/members.md`
(roster member) dibaca ulang **setiap job** — edit saja filenya, tidak
perlu restart server. `context/template.ass` menentukan style .ass;
tempel style dari episode HikaLeon asli untuk hasil yang identik.

## Menjalankan test

```powershell
python -m pytest -q
```
````

- [ ] **Step 2: Final verification — whole suite**

Run: `python -m pytest -q`
Expected: 87 passed, 0 failed.

Then verify the working tree is clean except `README.md`:

Run: `git status --short`
Expected: only `?? README.md` (plus this plan/spec if still uncommitted).

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "docs: README with setup, usage, and retry semantics"
```

---

## Done — definition of complete

- `python -m pytest -q` → 87 passed.
- `python -m uvicorn app.main:app --port 8000` serves the UI; `/api/config` reports provider availability and the ffmpeg check.
- A failed job retried from the UI resumes from its last checkpoint without re-transcribing.
- Spec coverage: §1 (Tasks 16–17), §2 (Task 1), §3 (Tasks 9, 17–18), §4 (Task 2), §5 (Tasks 8–9, 11), §6a (Tasks 10–12), §6b (Task 13), §7 (Task 14), §8 (Tasks 3, 15), §9 (Tasks 5, 12, 13, 16), §10 (Tasks 6, 16, 18).
