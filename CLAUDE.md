# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (requires ffmpeg in PATH and GEMINI_API_KEY in .env)
uvicorn app.main:app --reload

# Run full test suite (88 tests, ~1s)
pytest

# Run a single test file
pytest tests/test_pipeline.py

# Run a single test by name
pytest tests/test_translate.py -k "test_translate_batch_retries_with_error_feedback"

# Run with verbose output
pytest -v
```

## Architecture

**Request flow:** `static/index.html` → FastAPI (`app/main.py`) → `app/pipeline.py` → stages → LLM providers (`app/providers.py`)

### Pipeline stages (sequential, each checkpointed)

`download → normalize → chunk → transcribe → translate → format`

Each stage checks for its output artifact before running. If found, it emits `"done (checkpoint)"` and skips. This makes retry free — only the failed stage re-runs. Transcription checkpoints per-chunk (`chunks/transcript_NNN.json`); translation checkpoints per-batch (`chunks/translated_NNN.json`).

### Module responsibilities

| Module | Role |
|--------|------|
| `app/main.py` | 7 FastAPI endpoints: UI, config, job CRUD, SSE stream, file download (allowlisted), retry |
| `app/pipeline.py` | Stage orchestration, checkpoint logic, SSE event emission, `_read_context()` hot-reload |
| `app/providers.py` | `Config.load()`, LLM client wrappers (Gemini/OpenAI/Anthropic), pricing table, `_clean_cookies_value()` |
| `app/audio.py` | yt-dlp download with `--remote-components ejs:github`, ffmpeg normalize, silence-based chunking |
| `app/transcribe.py` | Gemini transcription, recursive chunk splitting on truncation, overlap deduplication |
| `app/translate.py` | 3-tier retry (batch → retry with error_feedback → per-line fallback), `BATCH_SIZE=100` |
| `app/subtitle.py` | 5-step postprocess: merge_short → extend_short → split_long → fix_overlaps → cps_flags; ASS/SRT render |

### Key design patterns

**Context hot-reload:** `_read_context()` re-reads `context/context.md` and `context/members.md` fresh on every job — never cached. Update the glossary or member roster without restarting the server.

**LLM clients:** All three providers use `temperature=0.3`, `max_retries=0`. Retries are handled by a `call_with_retries` wrapper (4 attempts, base delay 2s exponential backoff).

**File download allowlist:** `GET /api/jobs/{id}/files/{filename}` only serves files in the `DOWNLOADABLE` set — `job.json` is explicitly excluded.

**SSE stream:** Polls `job.events` every 500ms. Sends `event: end` when status is `done` or `failed`. The frontend's `onerror` is a no-op because stream close is expected.

### Configuration (`.env`)

`GEMINI_API_KEY` is the only required var. Providers without API keys are hidden from the UI dropdown. `YTDLP_COOKIES_FILE` must be a bare path with no inline comment (see `_clean_cookies_value()` in `providers.py`).

### Test setup

`tests/conftest.py` sets `GEMINI_API_KEY=test-gemini-key` before any app import and patches `load_dotenv` to a no-op via an `autouse` fixture — tests never read the real `.env`.

### Context files

`context/template.ass` — ASS subtitle template (Comic Sans MS 66pt, 1920×1080). `context/context.md` — style guide and glossary. `context/members.md` — member roster. These are read every job run.
