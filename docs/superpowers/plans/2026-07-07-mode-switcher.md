# Mode Switcher (Transcription Only / Translation Only / Standard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-mode switcher so users can run Transcription Only (audio → Japanese transcript), Translation Only (transcript → Indonesian subtitle), or the existing Standard (full) pipeline.

**Architecture:** A new `mode` field in `job.params` controls which pipeline stages execute. The backend adds an `.ass`-file parser and a `source_job_id` input path for Translation Only. The frontend shows a top-of-page mode switcher that morphs the form; the session sidebar now also shows `done` jobs filtered and labelled by mode.

**Tech Stack:** Python / FastAPI backend (`app/`), vanilla HTML/CSS/JS frontend (`static/index.html`), no new dependencies.

---

## File Map

| File | Change |
|---|---|
| `app/subtitle.py` | Add `parse_ass_utterances()` to extract timecoded lines from a `.ass` file |
| `app/pipeline.py` | Add `mode` branching in `run_job()`, new `_stage_format_transcript()` for transcription-only output, plumb `source_job_id` / uploaded transcript into `_stage_load_for_translation()` |
| `app/main.py` | Accept new form fields (`mode`, `source_job_id`), relax validation per mode, expose `result_jp.ass` in `DOWNLOADABLE`, update `/api/sessions` to return done sessions with mode label |
| `static/index.html` | Mode switcher UI, conditional form sections, session sidebar updated |
| `tests/test_subtitle.py` | Tests for `parse_ass_utterances()` |
| `tests/test_pipeline_modes.py` | Tests for mode-aware pipeline branching |

---

## Task 1: Add `.ass` parser to `subtitle.py`

**Files:**
- Modify: `app/subtitle.py`
- Test: `tests/test_subtitle.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_subtitle.py  (add at the bottom; file already exists)
import pytest
from app.subtitle import parse_ass_utterances
from app.transcribe import Utterance

_SAMPLE_ASS = """[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,今日は楽しいですね
Dialogue: 0,0:00:04.00,0:00:06.00,Narrator,,0,0,0,,ナレーション
Dialogue: 0,0:00:07.00,0:00:09.00,Default,,0,0,0,,また来週
"""

def test_parse_ass_utterances_count():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert len(utts) == 3

def test_parse_ass_utterances_timing():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[0].start == pytest.approx(1.0)
    assert utts[0].end == pytest.approx(3.5)

def test_parse_ass_utterances_text():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[0].ja == "今日は楽しいですね"

def test_parse_ass_utterances_narration_type():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert utts[1].type == "narration"
    assert utts[0].type == "dialogue"

def test_parse_ass_utterances_ids_sequential():
    utts = parse_ass_utterances(_SAMPLE_ASS)
    assert [u.id for u in utts] == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/test_subtitle.py::test_parse_ass_utterances_count -v
```

Expected: `ERROR` or `ImportError` (function not yet defined).

- [ ] **Step 3: Implement `parse_ass_utterances` in `app/subtitle.py`**

Add after the `format_srt_time` function (after line 134):

```python
import re as _re

_ASS_DIALOGUE_RE = _re.compile(
    r"^Dialogue:\s*\d+,"          # layer
    r"(\d+:\d+:\d+\.\d+),"        # Start  group 1
    r"(\d+:\d+:\d+\.\d+),"        # End    group 2
    r"([^,]*),"                    # Style  group 3
    r"[^,]*,"                      # Actor
    r"\d+,\d+,\d+,"               # Margins
    r"[^,]*,"                      # Effect
    r"(.+)$",                      # Text   group 4
    _re.IGNORECASE,
)


def _parse_ass_time(t: str) -> float:
    """Convert ASS time '0:01:23.45' -> float seconds."""
    h, m, rest = t.split(":")
    s, cs = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100


def parse_ass_utterances(ass_text: str) -> list:
    """Parse a .ass file's Dialogue lines into Utterance objects with jp text."""
    from app.transcribe import Utterance  # local import to avoid circular
    utterances = []
    for line in ass_text.splitlines():
        m = _ASS_DIALOGUE_RE.match(line.strip())
        if not m:
            continue
        start = _parse_ass_time(m.group(1))
        end = _parse_ass_time(m.group(2))
        style = m.group(3).strip()
        text = m.group(4).strip().replace("\\N", "\n").replace("\\n", "\n")
        if not text:
            continue
        utype = "narration" if style.lower() == "narrator" else "dialogue"
        utterances.append(Utterance(
            id=len(utterances) + 1,
            start=start,
            end=end,
            type=utype,
            ja=text,
        ))
    return utterances
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/test_subtitle.py -v
```

Expected: all subtitle tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/subtitle.py tests/test_subtitle.py
git commit -m "feat: add parse_ass_utterances() for Translation Only input"
```

---

## Task 2: Add mode helpers to `pipeline.py`

**Files:**
- Modify: `app/pipeline.py`
- Test: `tests/test_pipeline_modes.py`

### 2a — Helper: `_find_source_by_ext` and `_load_utterances_from_source`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline_modes.py  (new file)
import json
import pytest
from pathlib import Path
from app.pipeline import _load_utterances_from_source


def _make_json_file(tmp_path, data):
    p = tmp_path / "transcript_jp.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_utterances_from_json(tmp_path):
    data = [{"id": 1, "start": 1.0, "end": 3.0, "type": "dialogue", "ja": "こんにちは"}]
    p = _make_json_file(tmp_path, data)
    utts = _load_utterances_from_source("json", json_path=p)
    assert len(utts) == 1
    assert utts[0].ja == "こんにちは"


def test_load_utterances_from_ass(tmp_path):
    ass = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,テスト\n"
    )
    p = tmp_path / "source.ass"
    p.write_text(ass, encoding="utf-8")
    utts = _load_utterances_from_source("ass", ass_path=p)
    assert len(utts) == 1
    assert utts[0].ja == "テスト"


def test_load_utterances_invalid_type():
    with pytest.raises(ValueError, match="source_type"):
        _load_utterances_from_source("unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/test_pipeline_modes.py -v
```

Expected: `ImportError` (function not yet defined).

- [ ] **Step 3: Add `_find_source_by_ext` to `app/pipeline.py`**

Add just after `_find_source` (after line 132):

```python
def _find_source_by_ext(job: "Job", ext: str) -> "Path | None":
    """Return the first file in job.dir matching the given extension."""
    matches = sorted(job.dir.glob(f"source{ext}"))
    return matches[0] if matches else None
```

- [ ] **Step 4: Add `_load_utterances_from_source` to `app/pipeline.py`**

Add just after `_find_source_by_ext`:

```python
def _load_utterances_from_source(
    source_type: str,
    *,
    json_path: "Path | None" = None,
    ass_path: "Path | None" = None,
) -> "list[transcribe.Utterance]":
    """Load utterances for Translation Only mode from a JSON or .ass file."""
    if source_type == "json":
        if json_path is None or not json_path.exists():
            raise PipelineError("transcript_jp.json tidak ditemukan.")
        return transcribe.load_transcript(json_path)
    if source_type == "ass":
        if ass_path is None or not ass_path.exists():
            raise PipelineError("File .ass tidak ditemukan.")
        from app.subtitle import parse_ass_utterances
        return parse_ass_utterances(ass_path.read_text(encoding="utf-8"))
    raise ValueError(f"source_type tidak dikenal: {source_type!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

```powershell
python -m pytest tests/test_pipeline_modes.py -v
```

Expected: all three pass.

### 2b — Transcription Only output stage: `_stage_format_transcript`

- [ ] **Step 6: Write failing tests**

```python
# tests/test_pipeline_modes.py  (continue)
import os
from unittest.mock import MagicMock
from app.pipeline import _stage_format_transcript
from app.transcribe import Utterance


def test_stage_format_transcript_json_only(tmp_path):
    job = MagicMock()
    job.dir = tmp_path
    job.params = {"output_format": "json", "save_mp4": False}
    job.events = []
    utterances = [Utterance(id=1, start=0.0, end=2.0, type="dialogue", ja="テスト")]
    (tmp_path / "transcript_jp.json").write_text(
        json.dumps([u.to_dict() for u in utterances]), encoding="utf-8"
    )
    files = _stage_format_transcript(job, utterances)
    assert "transcript_jp.json" in files
    assert not (tmp_path / "result_jp.ass").exists()


def test_stage_format_transcript_ass(tmp_path):
    job = MagicMock()
    job.dir = tmp_path
    job.params = {"output_format": "ass", "save_mp4": False}
    job.events = []
    utterances = [Utterance(id=1, start=0.0, end=2.0, type="dialogue", ja="テスト")]
    (tmp_path / "transcript_jp.json").write_text("[]", encoding="utf-8")
    # Create a minimal template
    os.makedirs("context", exist_ok=True)
    Path("context/template.ass").write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Style, Actor, "
        "MarginL, MarginR, MarginV, Effect, Text\n",
        encoding="utf-8",
    )
    files = _stage_format_transcript(job, utterances)
    assert "result_jp.ass" in files
    assert (tmp_path / "result_jp.ass").exists()
```

- [ ] **Step 7: Implement `_stage_format_transcript` in `app/pipeline.py`**

Add after `_stage_transcribe` (after line 265):

```python
def _stage_format_transcript(
    job: "Job", utterances: "list[transcribe.Utterance]"
) -> list[str]:
    """Transcription Only: write transcript_jp.json + optional result_jp.ass."""
    fmt = job.params.get("output_format", "json")
    _emit(job, "format", "start", f"menyimpan hasil transkripsi ({fmt})")
    files = ["transcript_jp.json"]
    if fmt in ("ass", "both"):
        events = [
            subtitle.SubEvent(start=u.start, end=u.end, text=u.ja, type=u.type)
            for u in utterances
        ]
        template = (CONTEXT_DIR / "template.ass").read_text(encoding="utf-8")
        (job.dir / "result_jp.ass").write_text(
            subtitle.render_ass(events, template), encoding="utf-8"
        )
        files.append("result_jp.ass")
    if (job.dir / "source.mp4").exists():
        files.append("source.mp4")
    _emit(job, "format", "done", ", ".join(files))
    return files
```

- [ ] **Step 8: Run all pipeline tests**

```powershell
python -m pytest tests/test_pipeline_modes.py -v
```

Expected: all pass.

- [ ] **Step 9: Commit**

```powershell
git add app/pipeline.py tests/test_pipeline_modes.py
git commit -m "feat: add _load_utterances_from_source and _stage_format_transcript"
```

---

## Task 3: Wire mode branching into `run_job()` and add `_stage_load_for_translation`

**Files:**
- Modify: `app/pipeline.py`

The three modes map to these stage sequences:

| Mode | Stages |
|---|---|
| `transcription_only` | download → normalize → chunk → transcribe → **format_transcript** |
| `translation_only` | **load_transcript** → translate → format |
| `full` (default) | download → normalize → chunk → transcribe → translate → format |

- [ ] **Step 1: Add `_stage_load_for_translation` to `app/pipeline.py`**

Add just before `_stage_translate`:

```python
def _stage_load_for_translation(job: "Job") -> "list[transcribe.Utterance]":
    """Translation Only: load utterances from uploaded file or a prior job."""
    # Checkpoint: already copied in a previous run
    final_path = job.dir / "transcript_jp.json"
    if final_path.exists():
        utts = transcribe.load_transcript(final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran (checkpoint)")
        return utts

    _emit(job, "load_transcript", "start", "memuat sumber transkripsi...")

    # Option A: reference a prior job by ID
    source_job_id = job.params.get("source_job_id")
    if source_job_id:
        src_path = OUTPUT_DIR / source_job_id / "transcript_jp.json"
        if not src_path.exists():
            raise PipelineError(
                f"transcript_jp.json tidak ditemukan di job {source_job_id}.")
        utts = transcribe.load_transcript(src_path)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari job {source_job_id}")
        return utts

    # Option B: uploaded JSON file
    json_src = _find_source_by_ext(job, ".json")
    if json_src:
        utts = _load_utterances_from_source("json", json_path=json_src)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari JSON")
        return utts

    # Option C: uploaded .ass file
    ass_src = _find_source_by_ext(job, ".ass")
    if ass_src:
        utts = _load_utterances_from_source("ass", ass_path=ass_src)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari .ass")
        return utts

    raise PipelineError(
        "Translation Only: tidak ada source_job_id, JSON, atau .ass yang tersedia.")
```

- [ ] **Step 2: Replace `run_job()` body in `app/pipeline.py`**

Replace lines 85–113 with:

```python
def run_job(job_id: str) -> None:
    job = JOBS.get(job_id) or load_job(job_id)
    if job is None:
        raise KeyError(f"Job tidak ditemukan: {job_id}")
    cfg = providers.Config.load()
    tracker = UsageTracker(job.dir / "usage.json")
    job.status, job.error = "running", None
    _save(job)
    stage = "init"
    mode = job.params.get("mode", "full")
    try:
        if mode == "transcription_only":
            stage = "download"
            source = _stage_download(job, cfg)
            stage = "normalize"
            audio_path = _stage_normalize(job, source)
            stage = "chunk"
            chunks = _stage_chunk(job, audio_path)
            stage = "transcribe"
            utterances = _stage_transcribe(job, cfg, tracker, chunks)
            stage = "format"
            files = _stage_format_transcript(job, utterances)
        elif mode == "translation_only":
            stage = "load_transcript"
            utterances = _stage_load_for_translation(job)
            stage = "translate"
            rows = _stage_translate(job, cfg, tracker, utterances)
            stage = "format"
            files = _stage_format(job, cfg, rows)
        else:  # "full"
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
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        _emit(job, stage, "error", message=str(exc))
```

- [ ] **Step 3: Write integration test for `_stage_load_for_translation` from session**

```python
# tests/test_pipeline_modes.py  (continue)
from unittest.mock import MagicMock, patch
import app.pipeline as _pipeline_mod


def test_stage_load_for_translation_from_job_id(tmp_path, monkeypatch):
    prior_job_dir = tmp_path / "prior_job"
    prior_job_dir.mkdir()
    utts_data = [{"id": 1, "start": 0.0, "end": 2.0,
                  "type": "dialogue", "ja": "Hello"}]
    (prior_job_dir / "transcript_jp.json").write_text(
        json.dumps(utts_data), encoding="utf-8"
    )
    monkeypatch.setattr(_pipeline_mod, "OUTPUT_DIR", tmp_path)

    job = MagicMock()
    job.dir = tmp_path / "new_job"
    job.dir.mkdir()
    job.params = {"source_job_id": "prior_job", "mode": "translation_only"}
    job.events = []

    utts = _pipeline_mod._stage_load_for_translation(job)
    assert len(utts) == 1
    assert utts[0].ja == "Hello"
    assert (job.dir / "transcript_jp.json").exists()
```

- [ ] **Step 4: Run full test suite**

```powershell
python -m pytest -q
```

Expected: no regressions, new tests pass.

- [ ] **Step 5: Commit**

```powershell
git add app/pipeline.py tests/test_pipeline_modes.py
git commit -m "feat: wire mode branching into run_job() for all three modes"
```

---

## Task 4: Update `main.py` — API, validation, session listing

**Files:**
- Modify: `app/main.py`

### 4a — Accept new form fields and per-mode validation

- [ ] **Step 1: Update `create_job` form parameters** (around line 152)

Add two new Form params:

```python
mode: str = Form("full"),
source_job_id: str = Form(""),
```

- [ ] **Step 2: Add mode validation and per-mode format check** (inside `create_job`, after existing `output_format` check)

```python
VALID_MODES = {"full", "transcription_only", "translation_only"}
if mode not in VALID_MODES:
    raise HTTPException(422, f"mode tidak dikenal: {mode!r}")

# Transcription Only allows output_format "json" in addition to "ass"/"both"
if mode == "transcription_only":
    if output_format not in {"ass", "json", "both"}:
        raise HTTPException(
            422,
            f"output_format tidak valid untuk Transcription Only: {output_format!r}"
        )
elif output_format not in VALID_FORMATS:
    raise HTTPException(422, f"output_format tidak dikenal: {output_format!r}")
```

- [ ] **Step 3: Skip translator validation for Transcription Only**

Wrap the existing translator/model validation block:

```python
if mode != "transcription_only":
    # --- existing translator / translate_model validation block ---
    translation_defaults = models_cfg["defaults"]["translation"]
    if not translate_model.strip() and translator not in translation_defaults:
        raise HTTPException(422, f"translator tidak dikenal: {translator!r}")
    resolved_translate_model = translate_model.strip() or translation_defaults[translator]
    provider_model_list = models_cfg["translation"].get(translator, [])
    if resolved_translate_model not in provider_model_list:
        raise HTTPException(422, f"translate_model tidak dikenal untuk provider {translator!r}: {resolved_translate_model!r}")
else:
    resolved_translate_model = ""
```

- [ ] **Step 4: Skip transcribe model validation for Translation Only**

Wrap the existing transcribe model validation:

```python
if mode != "translation_only":
    resolved_transcribe_model = transcribe_model.strip() or models_cfg["defaults"]["transcription"]
    if resolved_transcribe_model not in models_cfg["transcription"]:
        raise HTTPException(422, f"transcribe_model tidak dikenal: {resolved_transcribe_model!r}")
else:
    resolved_transcribe_model = ""
```

- [ ] **Step 5: Add Translation Only source handling** (after existing `source == "file"` block)

Add a new `elif` for Translation Only before the final `else`:

```python
elif mode == "translation_only" and source_job_id.strip():
    # Continuity from a prior session — no file upload needed
    upload_bytes = None
    upload_name = None
```

And adjust the `else` clause error to also mention Translation Only:

```python
else:
    if mode == "translation_only" and file is None and not source_job_id.strip():
        raise HTTPException(
            422, "Translation Only butuh file JSON/.ass atau source_job_id.")
    elif mode != "translation_only":
        raise HTTPException(422, f"source tidak dikenal: {source!r}")
```

- [ ] **Step 6: Add `mode` and `source_job_id` to the `params` dict**

```python
params = {
    "mode": mode,
    "source": source,
    "url": url.strip() or None,
    "translator": translator,
    "output_format": output_format,
    "save_mp4": save_mp4,
    "original_filename": upload_name,
    "group": group.strip() or "sakurazaka",
    "context_preset": context_preset.strip() or "sakurazaka_sokomagattara",
    "context_override": context_override.strip() or None,
    "additional_context": additional_context.strip() or None,
    "transcribe_model": resolved_transcribe_model,
    "translate_model": resolved_translate_model,
    "source_job_id": source_job_id.strip() or None,
}
```

### 4b — Add `result_jp.ass` to downloadable files

- [ ] **Step 7: Update `DOWNLOADABLE` set** (line 46)

```python
DOWNLOADABLE = {
    "result.ass", "result.srt", "transcript_jp.json",
    "translated_id.json", "flags.json", "usage.json",
    "source.mp4", "result_jp.ass",
}
```

### 4c — Update `/api/sessions` to return done sessions with mode labels

- [ ] **Step 8: Replace the `/api/sessions` endpoint** (around line 75)

```python
@app.get("/api/sessions")
def list_sessions(include_done: bool = Query(False)) -> dict:
    output_dir = pipeline.OUTPUT_DIR
    if not output_dir.exists():
        return {"sessions": []}
    sessions = []
    for job_dir in sorted(output_dir.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "job.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = meta.get("status", "")
        if status not in ("failed", "running") and not (include_done and status == "done"):
            continue
        job_id = meta["id"]
        params = meta.get("params", {})
        mode = params.get("mode", "full")
        display_name = (params.get("original_filename")
                        or params.get("url")
                        or job_id)
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        try:
            dt = datetime.strptime(job_id[:15], "%Y%m%d-%H%M%S")
            ts = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ts = job_id[:15]
        _MODE_LABELS = {
            "full": "Standard",
            "transcription_only": "Transcription Only",
            "translation_only": "Translation Only",
        }
        sessions.append({
            "id": job_id,
            "label": f"{ts} · {display_name} · {status}",
            "status": status,
            "mode": mode,
            "mode_label": _MODE_LABELS.get(mode, mode),
        })
    return {"sessions": sessions}
```

- [ ] **Step 9: Run all tests**

```powershell
python -m pytest -q
```

Expected: no regressions.

- [ ] **Step 10: Commit**

```powershell
git add app/main.py
git commit -m "feat: update main.py for mode-aware job creation and session listing"
```

---

## Task 5: Update the Frontend (`static/index.html`)

**Files:**
- Modify: `static/index.html`

Read `static/index.html` in full before editing to understand its current structure.

### 5a — Mode Switcher

- [ ] **Step 1: Add mode switcher HTML** at the very top of the main form, before the group/show section

```html
<!-- ── Mode Switcher ── -->
<div class="field" id="field-mode">
  <label class="label">Mode</label>
  <div class="tabs is-toggle is-fullwidth" id="mode-tabs">
    <ul>
      <li class="is-active" data-mode="full"><a>⚡ Standard</a></li>
      <li data-mode="transcription_only"><a>🎤 Transcription Only</a></li>
      <li data-mode="translation_only"><a>🌏 Translation Only</a></li>
    </ul>
  </div>
  <input type="hidden" id="mode-input" name="mode" value="full">
</div>
```

- [ ] **Step 2: Add mode switcher JS** in the `<script>` section

```javascript
// ── Mode Switcher ──────────────────────────────────────────────────
const modeTabs = document.querySelectorAll('#mode-tabs li');
const modeInput = document.getElementById('mode-input');

modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modeTabs.forEach(t => t.classList.remove('is-active'));
    tab.classList.add('is-active');
    modeInput.value = tab.dataset.mode;
    applyModeVisibility(tab.dataset.mode);
    refreshSessions();
  });
});

function applyModeVisibility(mode) {
  // Audio source section: hidden in Translation Only
  const srcSection = document.getElementById('section-source');
  if (srcSection) srcSection.style.display = mode === 'translation_only' ? 'none' : '';

  // Transcript source section: only in Translation Only
  const tsrcSection = document.getElementById('section-transcript-source');
  if (tsrcSection) tsrcSection.style.display = mode === 'translation_only' ? '' : 'none';

  // Translator / translate model: hidden in Transcription Only
  const transSection = document.getElementById('section-translator');
  if (transSection) transSection.style.display = mode === 'transcription_only' ? 'none' : '';

  // Transcription model: hidden in Translation Only
  const tscribeSection = document.getElementById('section-transcribe-model');
  if (tscribeSection) tscribeSection.style.display = mode === 'translation_only' ? 'none' : '';

  // Save MP4: only relevant when downloading from YouTube
  const mp4Section = document.getElementById('section-save-mp4');
  if (mp4Section) mp4Section.style.display = mode === 'translation_only' ? 'none' : '';

  // Output format options change per mode
  renderOutputFormatOptions(mode);
}

function renderOutputFormatOptions(mode) {
  const sel = document.getElementById('output-format-select');
  if (!sel) return;
  sel.innerHTML = '';
  const opts = mode === 'transcription_only'
    ? [['ass', '.ass (subtitle Jepang)'], ['json', 'JSON (transcript_jp.json)'], ['both', 'Keduanya']]
    : [['ass', '.ass'], ['srt', '.srt'], ['both', 'Keduanya (.ass + .srt)']];
  opts.forEach(([val, label]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = label;
    sel.appendChild(opt);
  });
}
```

### 5b — Transcript Source Section (Translation Only)

- [ ] **Step 3: Add transcript source HTML** after the existing audio source section

Wrap the existing audio source (`#section-source`) then add:

```html
<!-- ── Transcript Source (Translation Only only) ── -->
<div id="section-transcript-source" style="display:none" class="box">
  <div class="field">
    <label class="label">Sumber Transkripsi</label>
    <div class="tabs is-toggle is-small" id="tsrc-tabs">
      <ul>
        <li class="is-active" data-tsrc="session"><a>📂 Dari Sesi</a></li>
        <li data-tsrc="file_json"><a>📄 Upload JSON</a></li>
        <li data-tsrc="file_ass"><a>📝 Upload .ass</a></li>
      </ul>
    </div>
  </div>

  <!-- From prior session -->
  <div id="tsrc-session">
    <p class="help">Pilih sesi Transcription Only yang sudah selesai dari sidebar kanan, lalu klik "Gunakan". Job ID akan otomatis terisi di bawah.</p>
    <div class="field">
      <label class="label">Source Job ID</label>
      <input class="input" type="text" id="source-job-id-input"
             placeholder="(pilih dari sidebar atau tempel job ID)">
    </div>
  </div>

  <!-- Upload JSON -->
  <div id="tsrc-file-json" style="display:none">
    <div class="field">
      <label class="label">Upload transcript_jp.json</label>
      <div class="file has-name">
        <label class="file-label">
          <input class="file-input" type="file" id="file-input-json" accept=".json">
          <span class="file-cta"><span class="file-label">Pilih file…</span></span>
          <span class="file-name" id="file-name-json">Belum dipilih</span>
        </label>
      </div>
    </div>
  </div>

  <!-- Upload .ass -->
  <div id="tsrc-file-ass" style="display:none">
    <div class="field">
      <label class="label">Upload .ass (subtitle Jepang)</label>
      <div class="file has-name">
        <label class="file-label">
          <input class="file-input" type="file" id="file-input-ass" accept=".ass">
          <span class="file-cta"><span class="file-label">Pilih file…</span></span>
          <span class="file-name" id="file-name-ass">Belum dipilih</span>
        </label>
      </div>
    </div>
  </div>
</div>
```

- [ ] **Step 4: Add transcript source sub-tab JS**

```javascript
// ── Transcript source sub-tabs ─────────────────────────────────────
const tsrcTabs = document.querySelectorAll('#tsrc-tabs li');
tsrcTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tsrcTabs.forEach(t => t.classList.remove('is-active'));
    tab.classList.add('is-active');
    const tsrc = tab.dataset.tsrc;
    document.getElementById('tsrc-session').style.display  = tsrc === 'session'   ? '' : 'none';
    document.getElementById('tsrc-file-json').style.display = tsrc === 'file_json' ? '' : 'none';
    document.getElementById('tsrc-file-ass').style.display  = tsrc === 'file_ass'  ? '' : 'none';
  });
});

// File name display
['json', 'ass'].forEach(ext => {
  const inp = document.getElementById(`file-input-${ext}`);
  const lbl = document.getElementById(`file-name-${ext}`);
  if (inp && lbl) inp.addEventListener('change', () => {
    lbl.textContent = inp.files[0]?.name || 'Belum dipilih';
  });
});
```

### 5c — Update form submission to handle Translation Only

- [ ] **Step 5: Update the submit / FormData assembly section**

Find where `FormData` is built and submitted. Add mode-aware logic:

```javascript
// Inside submitJob() or the submit handler:
const mode = modeInput.value;
fd.set('mode', mode);

if (mode === 'translation_only') {
  fd.delete('source');
  fd.delete('url');
  const activeTsrc = document.querySelector('#tsrc-tabs li.is-active')?.dataset.tsrc;

  if (activeTsrc === 'session') {
    const jobId = document.getElementById('source-job-id-input').value.trim();
    if (!jobId) { showError('Pilih sesi atau isi Source Job ID.'); return; }
    fd.set('source_job_id', jobId);
    fd.set('source', 'session');  // backend ignores this for translation_only + source_job_id
  } else if (activeTsrc === 'file_json') {
    const f = document.getElementById('file-input-json').files[0];
    if (!f) { showError('Pilih file JSON.'); return; }
    fd.set('file', f);
    fd.set('source', 'file');
  } else {
    const f = document.getElementById('file-input-ass').files[0];
    if (!f) { showError('Pilih file .ass.'); return; }
    fd.set('file', f);
    fd.set('source', 'file');
  }
}
// For all other modes: existing source/file handling unchanged
```

### 5d — Update session sidebar to show done sessions in Translation Only mode

- [ ] **Step 6: Update `refreshSessions()` (or equivalent session load function)**

```javascript
async function refreshSessions() {
  const mode = modeInput ? modeInput.value : 'full';
  const includeDone = mode === 'translation_only';
  const res = await fetch(`/api/sessions?include_done=${includeDone}`);
  const { sessions } = await res.json();

  sessionList.innerHTML = '';
  if (!sessions.length) {
    sessionList.innerHTML = '<li class="has-text-grey-light" style="padding:8px">Tidak ada sesi</li>';
    return;
  }
  sessions.forEach(s => {
    const li = document.createElement('li');
    const isDone = s.status === 'done';

    if (isDone && mode === 'translation_only') {
      // Clickable "Use" button for Translation Only
      li.innerHTML = `
        <span style="display:block;font-size:0.75em;color:#48c774;margin-bottom:2px">
          ✅ ${s.mode_label}
        </span>
        <span style="font-size:0.85em">${s.label}</span>
        <button class="button is-small is-link is-light"
                style="margin-left:4px"
                data-job-id="${s.id}">Gunakan</button>`;
      li.querySelector('button').addEventListener('click', e => {
        e.stopPropagation();
        // Auto-fill source job ID and switch to "session" sub-tab
        document.getElementById('source-job-id-input').value = s.id;
        tsrcTabs.forEach(t => t.classList.remove('is-active'));
        document.querySelector('#tsrc-tabs li[data-tsrc="session"]').classList.add('is-active');
        document.getElementById('tsrc-session').style.display = '';
        document.getElementById('tsrc-file-json').style.display = 'none';
        document.getElementById('tsrc-file-ass').style.display = 'none';
      });
    } else {
      // Failed / running: existing retry behaviour
      li.style.cursor = 'pointer';
      li.innerHTML = `
        <span style="display:block;font-size:0.75em;color:${s.status === 'failed' ? '#f14668' : '#ffdd57'};margin-bottom:2px">
          ${s.status === 'failed' ? '❌ Gagal' : '⏳ Berjalan'}
        </span>
        <span style="font-size:0.85em">${s.label}</span>`;
      li.addEventListener('click', () => loadSession(s.id));
    }
    li.style.padding = '8px 4px';
    li.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
    sessionList.appendChild(li);
  });
}
```

- [ ] **Step 7: Manual smoke test**

```powershell
python -m uvicorn app.main:app --port 8000 --reload
```

Open http://localhost:8000 and verify:
1. Mode switcher renders at the top with three tabs.
2. **Standard** tab: everything identical to before.
3. **Transcription Only** tab: translator/model section hidden, output format shows ass/json/both.
4. **Translation Only** tab: audio source hidden, transcript source section shown with three sub-tabs.
5. Selecting a done session from sidebar auto-fills source job ID.
6. Submitting a Transcription Only job produces `transcript_jp.json` + `result_jp.ass` download links.

- [ ] **Step 8: Commit**

```powershell
git add static/index.html
git commit -m "feat: add mode switcher and translation-only transcript source UI"
```

---

## Task 6: Final verification

- [ ] **Step 1: Run full test suite**

```powershell
python -m pytest -q
```

Expected: all tests pass, no regressions.

- [ ] **Step 2: E2E — Transcription Only with .ass output**

1. Upload a short audio or paste a YouTube URL.
2. Select **Transcription Only** mode, output format **ass**.
3. Click **Mulai**. Wait for completion.
4. Download `result_jp.ass` — confirm Japanese Dialogue lines with timestamps.

- [ ] **Step 3: E2E — Translation Only from prior session**

1. The job from Step 2 now appears in the sidebar when Translation Only mode is selected.
2. Click **Gunakan** on that session.
3. Choose group/show, translator model.
4. Click **Mulai**. Wait for completion.
5. Download `result.ass` — confirm Indonesian subtitles.

- [ ] **Step 4: E2E — Translation Only from uploaded JSON**

1. Download `transcript_jp.json` from any completed job.
2. Switch to Translation Only → "Upload JSON" sub-tab.
3. Upload the file, configure, submit.
4. Confirm subtitle output is produced correctly.

- [ ] **Step 5: Final commit**

```powershell
git add .
git commit -m "feat: complete three-mode pipeline (Transcription Only, Translation Only, Standard)"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|---|---|
| Transcription Only mode | Tasks 2b, 3 |
| Translation Only mode | Tasks 2a, 3 |
| Standard (full) unchanged | Task 3 — existing `else` branch |
| Transcription Only input: local file or YouTube | Task 4 — same source handling as today |
| Transcription Only output: .ass (JP text) | Task 2b |
| Transcription Only output: JSON | Task 2b |
| Transcription Only output: mp4 (if save_mp4 checked) | Task 2b |
| Translation Only input: JSON upload | Tasks 2a, 4, 5 |
| Translation Only input: .ass upload | Tasks 1, 2a, 4, 5 |
| Translation Only input: pick from session (done jobs only) | Tasks 3, 4c, 5d |
| Session sidebar shows done sessions in Translation Only mode | Tasks 4c, 5d |
| Clicking "Gunakan" auto-fills source job ID | Task 5d |
| Session items labelled with mode (Transcription Only / Standard) | Task 4c |
| Retry works for all modes | Task 3 — `run_job()` always called; checkpoint files skip completed stages |
| No changes to Standard (full) mode | Task 3 — `else` branch is identical to original |

### Placeholder Scan

No TBDs, TODOs, or incomplete code blocks remain.

### Type Consistency

- `Utterance` from `app.transcribe` used consistently throughout all new helpers.
- `_stage_load_for_translation` returns `list[transcribe.Utterance]` — same type consumed by `_stage_translate`.
- `_stage_format_transcript` uses `subtitle.SubEvent` and `subtitle.render_ass` — identical pattern to existing `_stage_format`.
- `parse_ass_utterances` returns `list[Utterance]` — consistent with `transcribe.load_transcript`.
