# Phase 1 Design: QC & Retry Stage Overhaul

**Date:** 2026-11-10
**Scope:** Phase 1 of the major system update — resilient per-chunk/batch auto-fallback + stage-aware manual retry, upstream consistency & multi-speaker prevention, and Two-Pass Hybrid QC with actionable fixes and UI reporting.
**Out of Scope (Deferred to Phase 2):** Custom OpenAI-compatible / `9router` endpoint configuration and dynamic remote `/v1/models` discovery. Phase 1 establishes clean `make_audio_client` vs `make_text_client` seams so Phase 2 plugs in without pipeline rewrites.

---

## 1. Goals & Problems Solved

1. **Stage-aware retry & `503` resilience:**
   - Today `static/index.html` `#retry-box` only allows switching `translate_model` and `gemini_key_label`, ignoring `transcribe_model` (even though `RetryRequest` in `app/main.py` already parses it), and `timing` / `qc` models are hardcoded from `config/models.json` without per-chunk/batch checkpoints.
   - Users experience `503` / overload errors mid-transcription or mid-translation and want **up to 3 customizable auto-fallback `(Key/Provider, Model)` attempts** per failing chunk/batch before halting for stage-aware manual retry.
2. **Episode-wide translation term consistency:**
   - `translate.py` translates 100-line batches independently with no shared memory, causing recurring show/segment names to drift across chunks (e.g., `"Pertandingan Ratu Serigala"` vs `"Wolf Queen"` vs `"Ratu Serigala Grand Prix"`).
3. **Max 2 speakers per subtitle cue:**
   - Rapid variety-show banter causes `transcribe.py` to bundle 3–4 speakers into one utterance, and `subtitle.py`'s `_merge_short` chains unlimited `<0.7s` blocks together with spaces.
4. **Effective, visible QC (`app/qc.py`):**
   - Current `qc.py` re-uploads 100% of audio a third time in 20-line slices, only flags `timing_ok: false` without modifying timestamps, never repairs misheard Japanese (`ja`), silently ignores batch failures, skips `translation_only` jobs, and hides `qc/report.json` from the UI and downloads.

---

## 2. Architecture Overview

```
                    +------------------------------------------+
                    |         Web UI (static/index.html)       |
                    | - Primary model/key selectors            |
                    | - Audio 3-slot fallback chain            |
                    | - Translation 3-slot fallback chain      |
                    | - Deep Audio QC toggle                   |
                    | - Stage-aware Retry box & QC Report card |
                    +--------------------+---------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         FastAPI & Pipeline (main.py / pipeline.py)              |
|                                                                                 |
|  Audio Stages (Gemini + Audio Fallback)     Text Stages (Provider + TL Fallback)|
|  +------------------------------------+     +--------------------------------+  |
|  | 1. transcribe (chunks/transcript_*) |--->| 2. translate (chunks/translated)|  |
|  |    - Max 2 speakers (- prefix)     |     |    - Rolling episode_glossary  |  |
|  +------------------------------------+     +----------------+---------------+  |
|                                                              |                  |
|  +------------------------------------+                      v                  |
|  | 3. timing (chunks/timing_*)        |<---------------------+                  |
|  |    - Per-chunk checkpoints         |                                         |
|  +-----------------+------------------+                                         |
|                    |                                                            |
|                    v                                                            |
|  +---------------------------------------------------------------------------+  |
|  | 4. qc (Two-Pass Hybrid QC)                                                |  |
|  |    Pass 1A: Free local audit (timing fallbacks, silencedetect, rate, lint)|  |
|  |    Pass 1B: Whole-episode text consistency QC (Text Fallback chain)       |  |
|  |    Pass 2 : Targeted Audio QC on flagged slices / all batches if Deep     |  |
|  |             (Audio Fallback chain; fixes ja, id_text, start/end, splits)  |  |
|  +------------------------------------+--------------------------------------+  |
|                                       |                                         |
|                                       v                                         |
|  +---------------------------------------------------------------------------+  |
|  | 5. format (subtitle.py)                                                   |  |
|  |    - _merge_short capped at max 2 turns (- Line 1\N- Line 2)              |  |
|  |    - Emits result.ass, result.srt, flags.json, qc_report.json             |  |
|  +---------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------+
```

---

## 3. Component Design

### 3.1 Fallback Chains, Key Isolation & Stage-Aware Retry (`providers.py`, `pipeline.py`, `main.py`, `index.html`)

#### Data Model
Define normalized fallback slot payloads stored in `job.params` and persisted in `job.json`:
- `audio_fallback`: list of up to 3 objects `{"key_label": str, "model": str}`
- `translate_fallback`: list of up to 3 objects `{"provider": str, "key_label": str | None, "model": str}`
- `timing_model`: `str` (defaults to `models.json["defaults"]["timing"]`)
- `qc_model`: `str` (defaults to `models.json["defaults"]["qc"]`)
- `deep_audio_qc`: `bool` (default `False`)
- `failed_stage`: `str | None` persisted on `Job` and `job.json` whenever a job fails (`"download"`, `"normalize"`, `"chunk"`, `"transcribe"`, `"translate"`, `"timing"`, `"qc"`, `"format"`).

#### Validation (`app/main.py`)
- `POST /api/jobs` accepts `audio_fallback: str = Form("[]")` and `translate_fallback: str = Form("[]")` as JSON-encoded strings (parsed via `json.loads`; malformed JSON returns HTTP `422`).
- `POST /api/jobs/{id}/retry` (`RetryRequest`) accepts optional `audio_fallback: list[dict] | None = None` and `translate_fallback: list[dict] | None = None`, plus optional `transcribe_model`, `timing_model`, `qc_model`, `translator`, `translate_model`, `gemini_key_label`, and `deep_audio_qc`.
- Both endpoints validate:
  - `len(slots) <= 3` (HTTP `422` if exceeded).
  - Audio slots: `key_label` exists in `get_all_gemini_keys()`, `model` exists in `models_cfg["transcription"]`.
  - Translation slots: `provider` is enabled in `cfg.available_providers()`, `model` exists in `models_cfg["translation"][provider]`, and if `provider == "gemini"`, `key_label` (defaulting to active key if omitted) exists in `get_all_gemini_keys()`.
  - `timing_model` and `qc_model` (if supplied) exist in `models_cfg["transcription"]`.
- Reject invalid values with HTTP `422`.

#### `FallbackExecutor` (`app/providers.py`)
- Decouples client construction into two factories (Phase 2 extension seam):
  - `make_audio_client(cfg: Config, model: str | None = None, key_label: str | None = None) -> GeminiClient`
  - `make_text_client(cfg: Config, provider: str, model: str | None = None, key_label: str | None = None)`
  - Keep existing `make_transcriber` and `make_translator` as thin wrappers so existing callers and tests remain compatible.
- **Thread-safe per-slot key resolution:** Resolve `key_label` directly from `get_all_gemini_keys()` and construct `GeminiClient(api_key, model)` directly. Never mutate the process-global `_active_gemini_label` during fallback execution.
- **Retry & Fallback Execution Flow (`execute_with_fallback`):**
  1. Attempt the **primary** client via `call_with_retries(..., attempts=2, base_delay=2.0)` so brief network hiccups recover quickly without burning fallback slots (`RETRYABLE_STATUS = {429, 500, 502, 503, 504}` and `NETWORK_ERRORS`).
  2. Trigger fallback progression when primary fails with `RETRYABLE_STATUS`, `NETWORK_ERRORS`, or HTTP `403` (so a busy/blocked key or model still steps cleanly to alternate key/model slots rather than aborting). Non-retryable client errors such as `400` bad request still raise immediately.
  3. Step sequentially through `Slot 1 -> Slot 2 -> Slot 3` for **that failing chunk/batch only** (each fallback slot uses `attempts=2`; subsequent chunks/batches start fresh on the user's selected Primary configuration unless the user manually retried with a changed primary model).
  4. On each slot switch, emit an SSE progress event via optional `on_fallback(slot_idx, slot_spec, exc)` callback (e.g., `"chunk 3/8 gagal (503) di gemini-3.6-flash → mencoba Fallback 1: key WORK / gemini-3.5-flash"`).
  5. Record usage in `UsageTracker` using `winning_client.model`.
- **Audio reference handling across Gemini keys:** Gemini Files API URIs are scoped to the API key/project that uploaded the file. `execute_audio_with_fallback` caches `audio_ref` per `(chunk_path, resolved_api_key)` so switching models on the **same** API key reuses the upload, whereas switching to a **different** Gemini API key re-uploads the chunk cleanly before `generate()`.

#### Checkpointing & Error Propagation (`app/timing.py`, `app/qc.py`, `app/pipeline.py`)
- **`timing.py`:** Add per-chunk checkpoint files `chunks/timing_{chunk.index:03d}.json` storing `{"utterances": [...], "report_entry": {...}}`.
  - Distinguish **API/transport failures** from **LLM parse/alignment mismatch**:
    - API/transport failures invoke the Audio Fallback Chain (`FallbackExecutor`); if all slots fail, raise so the job halts cleanly at `failed_stage = "timing"` with completed chunks preserved on disk.
    - Parse/ID mismatch after fallback keeps the current `"status": "fallback", "reason": "gemini_parse_error"` entry in `timing/report.json` so Pass 1A + Pass 2 QC will target that chunk.
- **`qc.py`:** Add checkpoints for Pass 1 (`chunks/qc_pass1.json`) and per-slice Pass 2 (`chunks/qc_audio_{idx:03d}.json`). Remove silent `except Exception:` batch skipping; if a slice exhausts the fallback chain, raise and halt at `failed_stage = "qc"` for manual retry.

#### Stage-Aware Retry UI (`static/index.html`)
- **Job Form:**
  - Add two collapsible/compact panels: **"Fallback Transkripsi / Audio (maks 3)"** (3 rows of `[API Key] + [Gemini Model]`) and **"Fallback Translasi (maks 3)"** (3 rows of `[Provider] + [Key if Gemini] + [Model]`).
  - Add `"Deep Audio QC (cek semua batch dengan audio)"` checkbox (unchecked by default) and optional `"Model Timing"` / `"Model QC"` selectors (defaulting cleanly to `models.json`).
- **Retry Panel (`#retry-box`):**
  - Tracks `lastFailedStage` from SSE `error` event (or `job.failed_stage` when resuming a past session).
  - Shows the relevant controls for `failed_stage`:
    - `transcribe`: `Model Transkripsi` + `API Key` (+ optional edit of Audio Fallback slots)
    - `timing`: `Model Timing` + `API Key`
    - `translate`: `Provider` + `Model Terjemahan` + `API Key` (when Gemini)
    - `qc`: `Model QC` + `API Key` + `Deep Audio QC` toggle
  - Clicking `"Coba Lagi"` sends only the active overrides in `POST /api/jobs/{id}/retry` and resumes from the failed chunk/batch checkpoint.

---

### 3.2 Upstream Consistency & Multi-Speaker Prevention

#### Transcription Prompt Rules (`app/transcribe.py`)
Update `build_transcribe_prompts` rules:
- Prefer **1 speaker per utterance** with accurate start/end bounds.
- Combine **at most 2 rapid/overlapping speakers** in a single cue only when their speech overlaps inseparably, formatted as two lines starting with `- `:
  `"- <ucapan pembicara 1>\n- <ucapan pembicara 2>"`
- Explicitly forbid merging 3 or more speakers into a single `ja` entry.

#### Rolling Episode Glossary (`app/translate.py`)
- Maintain `episode_glossary: dict[str, str]` across `BATCH_SIZE=100` batches (capped at `MAX_GLOSSARY_TERMS = 80`, keeping most recently reinforced/first-seen canonical terms).
- **Prompt format:**
  - Inject existing terms before the batch JSON payload:
    `=== GLOSARIUM EPISODE (WAJIB KONSISTEN) ===` followed by `JP -> ID` pairs.
  - Instruct the model to return:
    `{"translations": [{"id": ..., "id_text": "..."}], "new_terms": {"<istilah JP baru>": "<terjemahan ID>"}}`
    for recurring segment titles, game names, catchphrases, or nicknames introduced in that batch, and to preserve `- ` two-line speaker prefixes when present in `ja`.
- **Backward-compatible `parse_translation_response(text)`:**
  - Accepts both the new object `{"translations": [...], "new_terms": {...}}` and legacy plain array `[{"id": ..., "id_text": ...}]` (or wrapped lists).
  - Existing `parse_translation(text) -> dict[int, str]` remains available as a wrapper so callers/tests relying on it keep working unchanged.
  - Existing 3-tier alignment recovery (`batch -> error_feedback retry -> per-line fallback`) operates over `translations`, never failing just because `new_terms` was omitted.
- **Checkpoint format (`chunks/translated_{idx:03d}.json` + `episode_glossary.json`):**
  - Save `{"translations": {id: text}, "glossary": {jp: id}}`.
  - When loading a checkpoint during resume, accept legacy flat `{id: text}` JSON gracefully (`glossary = {}`) and merge saved `glossary` snapshots forward before continuing to the next uncheckpointed batch.
  - Persist `job.dir / "episode_glossary.json"` when translation finishes so Pass 1B QC consumes the exact canonical glossary.

#### Post-Processing Merge Cap (`app/subtitle.py`)
- Extend `SubEvent` with `turns: int = 1` (initialized to `2` if `"- "` appears on 2 or more lines in `text`).
- In `_merge_short`:
  - Never merge if `prev.turns + ev.turns > 2` or either text already contains multi-line `- ` speaker formatting.
  - Keep space joining (`f"{prev.text} {ev.text}"`) for a single speaker's short reaction + continuation, setting `turns = prev.turns + ev.turns` so at most 2 short blocks can ever merge together.
  - When `transcribe` or `qc` outputs 2-speaker cues with `- Line 1\n- Line 2`, `render_ass` converts `\n` to `\N` cleanly, preserving exact two-line `- ` dialogue formatting.

---

### 3.3 Two-Pass Hybrid QC & Report UI (`app/qc.py`, `app/pipeline.py`, `app/main.py`, `static/index.html`)

#### Data Flow
`_stage_qc` runs for both `full` mode (`audio_path` present) and `translation_only` mode (`audio_path is None`, running Pass 1A non-acoustic checks + Pass 1B text consistency QC).

#### Pass 1A — Free Local Acoustic & Rule Audit (`audit_local`)
Runs deterministically in ~1–2s and produces `audit_flags: dict[int, list[str]]` (mapping `utterance_id -> [reason_codes]`):
1. `timing_fallback`: Any utterance belonging to a chunk marked `"status": "fallback"` in `timing/report.json` (when `timing/report.json` exists).
2. `in_silence` / `overlap`: Using `audio.detect_silences(audio_path, -30)` (cached or run once), flag any cue whose `[start, end]` lies wholly inside a detected silence interval (`silence.start <= u.start` and `u.end <= silence.end` with duration `>= 0.7s`) and any overlapping consecutive cues (`prev.end > cur.start + 0.05`).
3. `rate_outlier`: Japanese speaking rate outside normal range (`len(ja_no_space) / duration > 18.0` on cues `>= 0.4s`, or `< 1.2` on cues `> 3.0s`).
4. `speaker_overflow`: `id_text` or `ja` has `>= 3` lines starting with `- ` or `>= 3` speaker turns.
5. `untranslated_jp`: `id_text` contains Japanese Hiragana/Katakana characters (`[\u3040-\u30ff]`) outside known allowed romaji/symbols.
6. `high_cps`: `len(id_text) / duration > cfg.sub_cps_flag` (default `25`).
7. `pronoun_violation`: Regex match for standalone forbidden pronouns (`\bsaya\b`, `\banda\b`, `\bgue\b`, `\blo\b`) unless explicitly allowed by show context.

#### Pass 1B — Whole-Episode Text Consistency QC (`run_text_qc`, checkpointed at `chunks/qc_pass1.json`)
- Runs via `make_text_client` + `translate_fallback` chain.
- Sends `episode_glossary.json`, show context + member roster, local `audit_flags`, and the episode rows (in one call when `<= 250` lines, or in 200-line windows with 30-line overlap and shared glossary when `> 250` lines).
- Returns structured fixes + audio flags:
  ```json
  [
    {
      "id": 14,
      "fixed_id_text": "Pertandingan Ratu Serigala dimulai!",
      "needs_audio_check": false,
      "reason": "Seragamkan istilah dengan glosarium episode"
    },
    {
      "id": 42,
      "fixed_id_text": "- Tunggu!\n- Eh, seriusan?!",
      "needs_audio_check": true,
      "reason": "Terlalu banyak pembicara dalam 1 baris; ja mencurigakan"
    }
  ]
  ```
- Updates `id_text` immediately for consistency/CPS/style/pronoun fixes, logs each diff to `issues`, and merges `needs_audio_check` IDs into the Pass 2 queue.

#### Pass 2 — Targeted Audio QC (`run_audio_qc`, checkpointed at `chunks/qc_audio_{idx:03d}.json`)
- Skipped automatically when `audio_path is None` (`translation_only` mode).
- **Slice construction (`build_audio_qc_windows`):**
  - If `deep_audio_qc=False` (default): collect all IDs flagged by Pass 1A (`timing_fallback`, `in_silence`, `overlap`, `rate_outlier`, `speaker_overflow`) or Pass 1B (`needs_audio_check`). Group nearby flagged IDs (gap `<= 15s` and span `<= 90s`) into contiguous batches with `±1.5s` audio buffer and `±5` surrounding context lines. If zero IDs are flagged, Pass 2 finishes with 0 audio calls.
  - If `deep_audio_qc=True`: build slices covering all rows in `cfg.qc_batch_size` batches (`window_size` context), matching full-episode coverage.
- **Actionable Audio Fixes (`apply_audio_qc_results`):**
  For each item in a slice, Gemini listens to the audio slice and can emit:
  - `fixed_ja`: repairs misheard Japanese / member name.
  - `fixed_id_text`: repairs Indonesian translation to match audio + glossary.
  - `fixed_start` / `fixed_end`: slice-relative timestamps (validated `0 <= start < end <= slice_duration + 0.5`), shifted back by `+ slice_start` to absolute episode time.
  - `split`: list of 2–4 sub-utterances `[{"start": ..., "end": ..., "ja": ..., "id_text": ...}]` to break apart 3–4 speaker merged bundles; replaces the original item.
- After Pass 2 completes, re-index `id` sequentially (`1..N`) sorted by `start` before saving `qc/utterances.json` and `qc/report.json`.

#### Report Format, Download & UI Display
- `qc/report.json` schema:
  ```json
  {
    "mode": "hybrid",
    "deep_audio_qc": false,
    "summary": {
      "total_rows": 240,
      "local_flags": 18,
      "audio_slices_checked": 4,
      "translation_fixes": 12,
      "ja_fixes": 3,
      "timing_fixes": 5,
      "splits": 2,
      "unresolved_warnings": 1
    },
    "issues": [
      {
        "id": 14,
        "pass": "text_qc",
        "type": "translation",
        "original": "Pertandingan Wolf Queen dimulai!",
        "fixed": "Pertandingan Ratu Serigala dimulai!",
        "reason": "Konsistensi glosarium"
      }
    ]
  }
  ```
- **Download allowlist (`app/main.py`):** Add `"qc_report.json"` to `DOWNLOADABLE`, resolve `"qc_report.json"` in `download()` to `job.dir / "qc" / "report.json"`, and include `"qc_report.json"` in `result.files` whenever `qc/report.json` exists.
- **UI QC Report Card (`static/index.html`):** Render a collapsible **"Laporan QC"** panel beside the downloads card showing the summary pills (translation fixes, JP fixes, timing fixes, speaker splits, warnings) and a scrollable diff list (`original -> fixed` + timestamp diffs + remaining warnings).

---

## 4. Error Handling Summary

| Failure Scenario | Behavior |
|---|---|
| `503` / `429` / `5xx` / network error in `transcribe`, `timing`, or Pass 2 `qc` | 2 short retries on primary -> auto-step through `audio_fallback` slots 1..3 for the current chunk/slice -> if all fail, save `failed_stage` + preserve checkpoints, emit SSE `error`, display stage-aware Retry UI |
| `503` / `429` / `5xx` / network error in `translate` or Pass 1B `qc` | 2 short retries on primary -> auto-step through `translate_fallback` slots 1..3 for the current batch -> if all fail, save `failed_stage` + preserve checkpoints, emit SSE `error`, display stage-aware Retry UI |
| `timing` returns malformed JSON / mismatched IDs after fallback | Log `"status": "fallback"` in `timing/report.json` + chunk checkpoint; Pass 1A flags every utterance in that chunk for Pass 2 Audio QC |
| `translate` returns partial/mismatched IDs or omits `new_terms` | Existing 3-tier alignment recovery (`batch -> error_feedback -> per-line`); missing `new_terms` defaults cleanly to `{}` |
| `qc` batch/slice returns malformed JSON or invalid split timestamps | Retry once with error feedback; reject out-of-range timestamps (`start >= end` or outside slice bounds) and log as warning rather than corrupting timing |
| Legacy `job.json` / `chunks/translated_*.json` resume | Missing new params (`audio_fallback`, `translate_fallback`, `deep_audio_qc`) default safely to `[]` / `False`; legacy `{id: text}` checkpoints load with empty glossary |

---

## 5. Testing Plan

All tests remain 100% offline using `monkeypatch` and fakes per `tests/conftest.py`:

1. **`tests/test_providers.py`**
   - `FallbackExecutor` progression (`primary -> slot 1 -> slot 2 -> slot 3`) on `503`/`429` and immediate propagation on `400` after slots / non-retryable failure handling.
   - Key isolation across Gemini slots (does not mutate `_active_gemini_label`).
   - `UsageTracker` records tokens under the winning fallback slot's model.
   - `audio_ref` re-upload when switching to a different Gemini API key, reuse on same key.
2. **`tests/test_translate.py`**
   - Rolling glossary extraction (`new_terms`) injected into subsequent batch prompts and capped at `MAX_GLOSSARY_TERMS`.
   - Backward-compatible parsing for plain JSON array responses and legacy `{id: text}` checkpoint files.
   - `- ` multi-speaker formatting preserved across batches.
3. **`tests/test_subtitle.py`**
   - `_merge_short` stops at 2 turns (3–4 rapid `<0.7s` reactions never chain into one block).
   - Multi-speaker `- Line 1\n- Line 2` renders cleanly as `- Line 1\N- Line 2` in `.ass`.
4. **`tests/test_timing.py`** *(new)*
   - Per-chunk `chunks/timing_NNN.json` checkpoint write and skip-on-resume.
   - Fallback chain invocation on `503` vs `"fallback"` report entry on unparseable LLM alignment.
5. **`tests/test_qc.py`** *(new)*
   - Pass 1A local audit catches `timing/report.json` fallbacks, silence dead-air, overlaps, JP speaking-rate outliers, >2 `- ` lines, untranslated JP kana/kanji, high CPS, and forbidden pronouns.
   - Pass 1B text consistency QC applies term fixes and flags `needs_audio_check`.
   - Pass 2 targeted window grouping (`<= 90s` merged slices), skipping audio when zero flags in default mode, checking all batches when `deep_audio_qc=True`, and applying `fixed_ja`, `fixed_id_text`, `fixed_start`/`fixed_end`, and `split` with sequential ID re-indexing.
   - `translation_only` mode still runs Pass 1A (non-acoustic) + Pass 1B text QC.
6. **`tests/test_api.py` & `tests/test_pipeline.py`**
   - `POST /api/jobs` and `POST /api/jobs/{id}/retry` accept and validate `audio_fallback`, `translate_fallback`, `timing_model`, `qc_model`, `deep_audio_qc` (HTTP `422` on >3 slots, unknown models, or missing keys).
   - `failed_stage` recorded on failure and preserved in `job.json`.
   - `GET /api/jobs/{id}/files/qc_report.json` serves `qc/report.json` cleanly.
