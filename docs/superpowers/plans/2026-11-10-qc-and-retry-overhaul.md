# Phase 1: QC & Retry Stage Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add customizable 3-slot auto-fallback chains and stage-aware manual retry across all pipeline stages, prevent multi-speaker over-merging and translation term drift upstream, and replace the blind audio QC pass with a checkpointed Two-Pass Hybrid QC + UI report.

**Architecture:** `app/providers.py` introduces thread-safe key isolation (`make_audio_client` / `make_text_client`) and `FallbackExecutor` to step through 3 user-configured `(Key/Provider, Model)` fallback slots per failing chunk/batch before halting for stage-aware manual retry. Upstream stages enforce max-2-speaker `- ` turns (`transcribe.py` + `subtitle.py`) and propagate a rolling episode glossary across batches (`translate.py`), while `qc.py` runs a free local acoustic/rule audit + whole-episode text consistency pass + targeted (or optional Deep) audio repair pass with full JSON/UI reporting.

**Tech Stack:** Python 3.11+, FastAPI, `google-genai`, `openai`, `anthropic`, ffmpeg `silencedetect`, vanilla HTML/JS (`static/index.html`), `pytest`.

---

## Global Constraints & File Map

- **Spec reference:** `docs/superpowers/specs/2026-11-10-qc-and-retry-overhaul-design.md`
- **Test isolation:** `tests/conftest.py` patches `load_dotenv` to a no-op and resets `_active_gemini_label` between tests. Never hit real LLM APIs in tests.
- **Backward compatibility:** Existing `make_transcriber(cfg, model=None)`, `make_translator(cfg, provider, model=None)`, legacy `job.json` files (without new params), and legacy `chunks/translated_*.json` flat `{id: text}` checkpoints must continue to work.
- **Full suite command:** `python -m pytest -q` (currently 157 tests passing).

**Files touched:**
- Modify: `app/providers.py` (client factories, key isolation, `FallbackExecutor`, `execute_audio_with_fallback`)
- Modify: `app/transcribe.py` (max-2-speaker prompt rules, `audio_runner` hook)
- Modify: `app/translate.py` (rolling glossary, backward-compatible parser, `text_runner` hook, checkpoint snapshots)
- Modify: `app/subtitle.py` (`SubEvent.turns`, `_merge_short` max-2-turn cap)
- Modify: `app/timing.py` (per-chunk checkpoints, error propagation vs parse fallback, `audio_runner` hook)
- Modify: `app/qc.py` (Pass 1A `audit_local`, Pass 1B `run_text_qc`, Pass 2 `build_audio_qc_windows` + `run_audio_qc`, `run_qc_all`, structured `save_qc` / `load_qc`)
- Modify: `app/pipeline.py` (`failed_stage`, fallback runners, `episode_glossary.json`, `timing`/`qc` checkpoints, `translation_only` text QC, `qc_report.json` artifact)
- Modify: `app/main.py` (validation for fallbacks/`timing_model`/`qc_model`/`deep_audio_qc`, `RetryRequest` expansion, `qc_report.json` download mapping, `failed_stage` in API responses)
- Modify: `static/index.html` (Audio & Translation 3-slot fallback UI, `timing_model`/`qc_model`/`deep_audio_qc` controls, stage-aware `#retry-box`, `#qc-report-card`)
- Create: `tests/test_timing.py`, `tests/test_qc.py`
- Modify: `tests/test_providers.py`, `tests/test_transcribe.py`, `tests/test_translate.py`, `tests/test_subtitle.py`, `tests/test_pipeline.py`, `tests/test_pipeline_modes.py`, `tests/test_api.py`

---

### Task 1: Key-Isolated Client Factories (`app/providers.py`, `tests/test_providers.py`)

**Files:**
- Modify: `app/providers.py:50-70,370-388`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
from app.providers import (
    get_active_gemini_label,
    make_audio_client,
    make_text_client,
    resolve_gemini_key,
)


def test_resolve_gemini_key_uses_label_without_mutating_global(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_WORK", "g-work")
    assert get_active_gemini_label() == "Default"
    assert resolve_gemini_key("WORK") == "g-work"
    assert get_active_gemini_label() == "Default"


def test_resolve_gemini_key_unknown_label_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    with pytest.raises(ConfigError, match="NOPE"):
        resolve_gemini_key("NOPE")


def test_make_audio_client_uses_specific_key_label(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_audio_client(cfg, model="gemini-3.5-flash", key_label="ALT")
        MockClient.assert_called_once_with("g-alt", "gemini-3.5-flash")
    assert get_active_gemini_label() == "Default"


def test_make_text_client_uses_specific_gemini_key_label(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_text_client(cfg, "gemini", model="gemini-3.6-flash", key_label="ALT")
        MockClient.assert_called_once_with("g-alt", "gemini-3.6-flash")


def test_make_text_client_delegates_openai_and_anthropic():
    cfg = _fake_cfg()
    with patch("app.providers.OpenAIClient") as MockOAI:
        make_text_client(cfg, "openai", model="gpt-4.1")
        MockOAI.assert_called_once_with("okey", "gpt-4.1")
    with patch("app.providers.AnthropicClient") as MockAnt:
        make_text_client(cfg, "anthropic", model="claude-haiku-4-5")
        MockAnt.assert_called_once_with("akey", "claude-haiku-4-5")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -k "resolve_gemini_key or make_audio_client or make_text_client" -v`
Expected: `ImportError: cannot import name 'make_audio_client' from 'app.providers'`

- [ ] **Step 3: Implement `resolve_gemini_key`, `make_audio_client`, `make_text_client` in `app/providers.py`**

In `app/providers.py`, add `resolve_gemini_key` right after `set_active_gemini_key` (around line 66):

```python
def resolve_gemini_key(key_label: str | None = None, *, fallback_key: str | None = None) -> str:
    """Resolve a Gemini API key by label without mutating global active state."""
    if not key_label:
        if fallback_key is not None:
            return fallback_key
        all_keys = get_all_gemini_keys()
        return all_keys.get(_active_gemini_label, "")
    all_keys = get_all_gemini_keys()
    if key_label not in all_keys:
        raise ConfigError(
            f"Label Gemini key '{key_label}' tidak ditemukan. "
            f"Tambahkan GEMINI_API_KEY_{key_label}=... ke .env."
        )
    return all_keys[key_label]
```

Then replace `make_transcriber` and `make_translator` at the bottom of `app/providers.py` (lines 373-388):

```python
def make_transcriber(cfg: Config, model: str | None = None) -> GeminiClient:
    return GeminiClient(cfg.gemini_api_key, model or cfg.transcribe_model)


def make_audio_client(
    cfg: Config, model: str | None = None, key_label: str | None = None
) -> GeminiClient:
    if key_label is None:
        return make_transcriber(cfg, model=model)
    api_key = resolve_gemini_key(key_label, fallback_key=cfg.gemini_api_key)
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
):
    if provider == "gemini" and key_label is not None:
        api_key = resolve_gemini_key(key_label, fallback_key=cfg.gemini_api_key)
        return GeminiClient(api_key, model or cfg.translate_models["gemini"])
    return make_translator(cfg, provider, model=model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: All tests in `tests/test_providers.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/providers.py tests/test_providers.py
git commit -m "feat(providers): add key-isolated make_audio_client and make_text_client"
```

---

### Task 2: `FallbackExecutor` & Audio Re-Upload Handling (`app/providers.py`, `tests/test_providers.py`)

**Files:**
- Modify: `app/providers.py:130-175,388+`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
from pathlib import Path as _Path
from app.providers import (
    AudioFallbackSlot,
    TranslateFallbackSlot,
    execute_audio_with_fallback,
    execute_with_fallback,
)


def test_execute_with_fallback_steps_slots_on_503():
    attempts = []
    fallbacks_logged = []

    class FakeTextClient:
        def __init__(self, label, exc=None):
            self.model = label
            self._exc = exc

        def generate(self, system, user, audio=None):
            attempts.append(self.model)
            if self._exc:
                raise self._exc
            return LLMResponse(text='[{"id": 1, "id_text": "ok"}]', input_tokens=10, output_tokens=5)

    clients = {
        ("gemini", "Default", "primary-m"): FakeTextClient("primary-m", FakeHTTPError(503)),
        ("gemini", "ALT", "slot1-m"): FakeTextClient("slot1-m", FakeHTTPError(429)),
        ("openai", None, "gpt-4o"): FakeTextClient("gpt-4o", None),
    }

    primary = TranslateFallbackSlot(provider="gemini", key_label="Default", model="primary-m")
    slots = [
        TranslateFallbackSlot(provider="gemini", key_label="ALT", model="slot1-m"),
        TranslateFallbackSlot(provider="openai", key_label=None, model="gpt-4o"),
    ]

    resp, used_client = execute_with_fallback(
        primary=primary,
        fallback_slots=slots,
        build_client=lambda s: clients[(s.provider, s.key_label, s.model)],
        call_fn=lambda c: c.generate("sys", "usr"),
        on_fallback=lambda idx, slot, exc: fallbacks_logged.append((idx, slot.model)),
        sleep=lambda _: None,
    )
    assert used_client.model == "gpt-4o"
    assert "ok" in resp.text
    assert attempts == ["primary-m", "primary-m", "slot1-m", "slot1-m", "gpt-4o"]
    assert fallbacks_logged == [(1, "slot1-m"), (2, "gpt-4o")]


def test_execute_with_fallback_steps_on_403_but_stops_on_400():
    class FakeClient:
        def __init__(self, model, status):
            self.model = model
            self._status = status

        def generate(self, system, user, audio=None):
            raise FakeHTTPError(self._status)

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="ALT", model="m1")

    with pytest.raises(FakeHTTPError) as exc_info:
        execute_with_fallback(
            primary=primary,
            fallback_slots=[slot1],
            build_client=lambda s: FakeClient(s.model, 403 if s.model == "m0" else 400),
            call_fn=lambda c: c.generate("s", "u"),
            sleep=lambda _: None,
        )
    assert exc_info.value.status_code == 400


def test_execute_audio_with_fallback_reuploads_only_when_key_changes(tmp_path):
    uploads = []
    calls = []
    chunk = tmp_path / "chunk_001.mp3"
    chunk.write_bytes(b"mp3")

    class FakeAudioClient:
        def __init__(self, api_key, model):
            self.api_key = api_key
            self.model = model

        def upload_audio(self, path):
            uploads.append((self.api_key, path))
            return f"ref:{self.api_key}"

        def generate(self, system, user, audio=None):
            calls.append((self.api_key, self.model, audio))
            if self.model == "m-primary":
                raise FakeHTTPError(503)
            if self.model == "m-slot1":
                raise FakeHTTPError(503)
            return LLMResponse(text="ok", input_tokens=5, output_tokens=3)

    primary = AudioFallbackSlot(key_label="Default", model="m-primary")
    slots = [
        AudioFallbackSlot(key_label="Default", model="m-slot1"),  # same key -> reuse upload
        AudioFallbackSlot(key_label="ALT", model="m-slot2"),      # new key -> re-upload
    ]
    key_map = {"Default": "k1", "ALT": "k2"}

    resp, used_client = execute_audio_with_fallback(
        chunk_path=chunk,
        primary=primary,
        fallback_slots=slots,
        build_client=lambda s: FakeAudioClient(key_map[s.key_label], s.model),
        resolve_key=lambda s: key_map[s.key_label],
        call_fn=lambda c, ref: c.generate("sys", "usr", audio=ref),
        sleep=lambda _: None,
    )
    assert used_client.model == "m-slot2"
    assert resp.text == "ok"
    assert uploads == [("k1", chunk), ("k2", chunk)]
    assert calls[0][2] == "ref:k1"
    assert calls[2][2] == "ref:k1"  # slot 1 reused k1 upload
    assert calls[4][2] == "ref:k2"  # slot 2 uploaded under k2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_providers.py -k "execute_" -v`
Expected: `ImportError: cannot import name 'AudioFallbackSlot' from 'app.providers'`

- [ ] **Step 3: Implement `AudioFallbackSlot`, `TranslateFallbackSlot`, `execute_with_fallback`, `execute_audio_with_fallback` in `app/providers.py`**

In `app/providers.py`, right below `RETRYABLE_STATUS` (line 131), add `FALLBACK_TRIGGER_STATUS`:

```python
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FALLBACK_TRIGGER_STATUS = RETRYABLE_STATUS | {403}
NETWORK_ERRORS = (ConnectionError, TimeoutError, httpx.TransportError)
```

And add at the bottom of `app/providers.py`:

```python
@dataclass(frozen=True)
class AudioFallbackSlot:
    key_label: str
    model: str


@dataclass(frozen=True)
class TranslateFallbackSlot:
    provider: str
    model: str
    key_label: str | None = None


def _is_fallback_trigger(exc: Exception) -> bool:
    status = _status_of(exc)
    return status in FALLBACK_TRIGGER_STATUS or isinstance(exc, NETWORK_ERRORS)


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
    chain = [primary, *fallback_slots[:3]]
    last_exc: Exception | None = None
    for idx, slot in enumerate(chain):
        if idx > 0 and on_fallback is not None:
            on_fallback(idx, slot, last_exc)
        client = build_client(slot)
        try:
            result = call_with_retries(
                lambda c=client: call_fn(c),
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
            upload_cache[api_key] = client.upload_audio(chunk_path)
            if on_upload_done is not None:
                on_upload_done()
        return call_fn(client, upload_cache[api_key])

    chain = [primary, *fallback_slots[:3]]
    last_exc: Exception | None = None
    for idx, slot in enumerate(chain):
        if idx > 0 and on_fallback is not None:
            on_fallback(idx, slot, last_exc)
        client = build_client(slot)
        try:
            result = call_with_retries(
                lambda c=client, s=slot: _run_for_client(c, s),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_providers.py -v`
Expected: All tests in `tests/test_providers.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/providers.py tests/test_providers.py
git commit -m "feat(providers): add FallbackExecutor and per-key audio upload caching"
```

---

### Task 3: Transcription Speaker Cap & Fallback Hook (`app/transcribe.py`, `tests/test_transcribe.py`)

**Files:**
- Modify: `app/transcribe.py:94-118,160-184`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transcribe.py`:

```python
def test_build_transcribe_prompts_caps_speakers():
    _, user = build_transcribe_prompts("CTX", "MEMBERS")
    assert "1 pembicara" in user or "satu pembicara" in user.lower()
    assert "maksimal 2" in user.lower()
    assert "- " in user


def test_transcribe_chunk_uses_audio_runner_when_provided(tmp_path):
    chunk = tmp_path / "chunk_001.mp3"
    chunk.write_bytes(b"mp3")
    tracker = FakeTracker()
    runner_calls = []

    def fake_runner(path, system, user, on_upload_done):
        runner_calls.append(path)
        if on_upload_done:
            on_upload_done()
        return GOOD, "gemini-3.5-flash"

    utts = transcribe_chunk(
        None, chunk, "sys", "usr", tracker,
        audio_runner=fake_runner,
    )
    assert len(utts) == 2
    assert runner_calls == [chunk]
    assert tracker.calls == [("gemini-3.5-flash", 100, 50)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_transcribe.py -k "caps_speakers or audio_runner" -v`
Expected: FAIL (`AssertionError` on prompt text + `TypeError: unexpected keyword argument 'audio_runner'`).

- [ ] **Step 3: Update `build_transcribe_prompts` and `transcribe_chunk` in `app/transcribe.py`**

In `app/transcribe.py`, replace `build_transcribe_prompts` (lines 94-118) with:

```python
def build_transcribe_prompts(context_md: str, members_md: str,
                             additional_context: str | None = None) -> tuple[str, str]:
    system = (
        "Kamu adalah transcriber profesional untuk audio variety show Jepang "
        "(Soko Magattara, Sakurazaka?). Gunakan konteks berikut untuk menulis "
        "nama member dengan ejaan yang benar dan memahami istilah acara.\n\n"
        "=== KONTEKS ACARA ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )
    if additional_context:
        system += "\n\n=== KONTEKS VIDEO ===\n" + additional_context
    user = (
        "Transkripsikan audio terlampir per ujaran ke bahasa Jepang.\n"
        "Balas HANYA array JSON (tanpa teks lain) dengan skema per item:\n"
        '{"id": 1, "start": "MM:SS.mmm", "end": "MM:SS.mmm", '
        '"type": "dialogue" | "narration", "ja": "..."}\n'
        "Aturan:\n"
        "- id mulai dari 1, berurutan.\n"
        "- start/end relatif terhadap awal audio INI (format MM:SS.mmm).\n"
        '- type "narration" untuk voice-over narator; selain itu "dialogue".\n'
        "- Usahakan 1 pembicara per item. Jika ada sahut-sahutan sangat cepat atau tumpang-tindih, "
        'gabungkan maksimal 2 pembicara dalam satu item dengan format dua baris diawali "- " '
        '("- <pembicara 1>\\n- <pembicara 2>"). DILARANG menggabungkan 3 pembicara atau lebih dalam 1 item.\n'
        "- Abaikan backchannel pendek tanpa makna (うん/はい) kecuali jawaban penting.\n"
        "- Teks layar (telop) tidak ditranskripsikan kecuali dibacakan."
    )
    return system, user
```

And update `transcribe_chunk` (lines 160-184):

```python
def transcribe_chunk(gemini, chunk_path: Path, system: str, user: str,
                     tracker, depth: int = 0,
                     on_upload_done=None,
                     audio_runner=None) -> list[Utterance]:
    """Transcribe one audio chunk; timestamps relative to the chunk start."""
    if audio_runner is not None:
        response, model_used = audio_runner(chunk_path, system, user, on_upload_done)
    else:
        audio_ref = gemini.upload_audio(chunk_path)
        if on_upload_done is not None:
            on_upload_done()
        response = call_with_retries(
            lambda: gemini.generate(system=system, user=user, audio=audio_ref))
        model_used = gemini.model
    if tracker is not None:
        tracker.add(model_used, response.input_tokens, response.output_tokens)
    try:
        if response.truncated:
            raise TranscriptParseError("output terpotong (MAX_TOKENS)")
        return parse_transcript(response.text)
    except TranscriptParseError:
        if depth >= MAX_SPLIT_DEPTH:
            raise
        left, right, offset = audio.split_audio(chunk_path)
        first = transcribe_chunk(gemini, left, system, user, tracker, depth + 1,
                                 audio_runner=audio_runner)
        second = transcribe_chunk(gemini, right, system, user, tracker, depth + 1,
                                  audio_runner=audio_runner)
        shifted = [replace(u, start=u.start + offset, end=u.end + offset)
                   for u in second]
        combined = first + shifted
        return [replace(u, id=i) for i, u in enumerate(combined, start=1)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: All tests in `tests/test_transcribe.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/transcribe.py tests/test_transcribe.py
git commit -m "feat(transcribe): cap speaker count to 2 and add audio_runner fallback hook"
```

---

### Task 4: Rolling Episode Glossary in Translation (`app/translate.py`, `tests/test_translate.py`)

**Files:**
- Modify: `app/translate.py:1-154`
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_translate.py`:

```python
from app.translate import (
    MAX_GLOSSARY_TERMS,
    merge_glossary,
    parse_translation_response,
    translate_all,
)


def test_parse_translation_response_accepts_object_with_new_terms():
    raw = json.dumps({
        "translations": [{"id": 1, "id_text": "Pertandingan Ratu Serigala"}],
        "new_terms": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"},
    }, ensure_ascii=False)
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Pertandingan Ratu Serigala"}
    assert terms == {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"}


def test_parse_translation_response_tolerates_legacy_plain_array():
    raw = json.dumps([{"id": 1, "id_text": "Halo"}])
    mapping, terms = parse_translation_response(raw)
    assert mapping == {1: "Halo"}
    assert terms == {}


def test_merge_glossary_caps_at_max_terms():
    g = {f"jp{i}": f"id{i}" for i in range(MAX_GLOSSARY_TERMS)}
    merged = merge_glossary(g, {"baru": "terbaru"})
    assert len(merged) == MAX_GLOSSARY_TERMS
    assert merged["baru"] == "terbaru"


def test_translate_all_propagates_glossary_and_checkpoints(tmp_path):
    batch1_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": i, "id_text": f"ID-{i}"} for i in range(1, 101)],
            "new_terms": {"人狼クイーン決定戦": "Pertandingan Ratu Serigala"},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    batch2_resp = LLMResponse(
        text=json.dumps({
            "translations": [{"id": 101, "id_text": "Pertandingan Ratu Serigala!"}],
            "new_terms": {},
        }, ensure_ascii=False),
        input_tokens=10, output_tokens=10,
    )
    client = FakeClient([batch1_resp, batch2_resp])
    utts = [U(i, "台詞") for i in range(1, 102)]
    glossary = {}
    saved = {}

    def save_ckpt(idx, data):
        saved[idx] = data

    res = translate_all(
        client, utts, "SYS", FakeTracker(),
        save_checkpoint=save_ckpt,
        glossary=glossary,
    )
    assert res[101] == "Pertandingan Ratu Serigala!"
    assert "人狼クイーン決定戦" in client.prompts[1]
    assert glossary["人狼クイーン決定戦"] == "Pertandingan Ratu Serigala"
    assert saved[1]["glossary"]["人狼クイーン決定戦"] == "Pertandingan Ratu Serigala"


def test_translate_all_loads_legacy_flat_checkpoint():
    utts = [U(1, "こんにちは")]
    legacy_ckpt = {1: "Halo"}
    glossary = {}
    res = translate_all(
        FakeClient([]), utts, "SYS", FakeTracker(),
        load_checkpoint=lambda idx: legacy_ckpt,
        glossary=glossary,
    )
    assert res == {1: "Halo"}
    assert glossary == {}


def test_translate_batch_uses_text_runner_fallback():
    tracker = FakeTracker()
    runner_calls = []

    def fake_runner(system, user):
        runner_calls.append(user)
        return R(GOOD_ITEMS), "fallback-model"

    res = translate_batch(None, UTTS, "SYS", tracker, text_runner=fake_runner)
    assert res == {1: "Halo", 2: "Makasih"}
    assert len(runner_calls) == 1
    assert tracker.calls == [("fallback-model", 10, 10)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_translate.py -v`
Expected: `ImportError: cannot import name 'MAX_GLOSSARY_TERMS' from 'app.translate'`

- [ ] **Step 3: Implement rolling glossary + `text_runner` in `app/translate.py`**

Replace `app/translate.py` with:

```python
"""JP -> ID translation in numbered JSON batches (§6b)."""
from __future__ import annotations

import json
import re

from app.providers import call_with_retries
from app.transcribe import Utterance

BATCH_SIZE = 100
MAX_GLOSSARY_TERMS = 80


class TranslateError(Exception):
    """Translation failed after retry + per-line fallback."""


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def merge_glossary(base: dict[str, str], additions: dict[str, str],
                   limit: int = MAX_GLOSSARY_TERMS) -> dict[str, str]:
    """Merge new JP->ID terms into base, keeping at most `limit` entries."""
    merged = dict(base)
    for k, v in (additions or {}).items():
        jp = str(k).strip()
        id_val = str(v).strip()
        if jp and id_val:
            if jp in merged:
                del merged[jp]
            merged[jp] = id_val
    while len(merged) > limit:
        oldest = next(iter(merged))
        del merged[oldest]
    return merged


def parse_translation_response(text: str) -> tuple[dict[int, str], dict[str, str]]:
    """Parse model output -> ({id: id_text}, {jp_term: id_term}).

    Accepts both {"translations": [...], "new_terms": {...}} and plain JSON arrays.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    data = json.loads(cleaned)
    raw_terms: dict[str, str] = {}
    items = None

    if isinstance(data, dict):
        if isinstance(data.get("new_terms"), dict):
            for k, v in data["new_terms"].items():
                jp, id_val = str(k).strip(), str(v).strip()
                if jp and id_val:
                    raw_terms[jp] = id_val
        if isinstance(data.get("translations"), list):
            items = data["translations"]
        else:
            for value in data.values():
                if isinstance(value, list):
                    items = value
                    break
    elif isinstance(data, list):
        items = data

    if not isinstance(items, list):
        raise ValueError("output bukan array JSON")

    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item["id"])
            id_text = str(item["id_text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if id_text:
            result[item_id] = id_text
    return result, raw_terms


def parse_translation(text: str) -> dict[int, str]:
    """Backward-compatible wrapper returning only {id: id_text}."""
    mapping, _ = parse_translation_response(text)
    return mapping


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


def build_translate_system(context_md: str, members_md: str,
                           additional_context: str | None = None) -> str:
    result = (
        "Kamu adalah penerjemah subtitle JP->ID untuk variety show "
        "Soko Magattara, Sakurazaka?. Patuhi style guide, glossary, dan "
        "ejaan nama persis seperti referensi berikut.\n\n"
        "=== KONTEKS & STYLE GUIDE ===\n" + context_md +
        "\n\n=== ROSTER MEMBER ===\n" + members_md
    )
    if additional_context:
        result += "\n\n=== KONTEKS VIDEO ===\n" + additional_context
    return result


def _build_user_prompt(utterances: list[Utterance],
                       error_feedback: str | None = None,
                       glossary: dict[str, str] | None = None,
                       include_terms: bool = True) -> str:
    payload = json.dumps([{"id": u.id, "ja": u.ja} for u in utterances],
                         ensure_ascii=False)
    glossary_block = ""
    if glossary:
        lines = "\n".join(f"- {jp} -> {id_val}" for jp, id_val in glossary.items())
        glossary_block = (
            "=== GLOSARIUM EPISODE (WAJIB KONSISTEN) ===\n"
            + lines + "\n\n"
        )
    if include_terms:
        schema_instr = (
            "Balas HANYA JSON dengan format:\n"
            '{"translations": [{"id": <id sama>, "id_text": "<terjemahan>"}], '
            '"new_terms": {"<istilah/segmen JP berulang baru>": "<padanan ID konsisten>"}}\n'
            "Catatan:\n"
            "- Jumlah item di 'translations' HARUS SAMA PERSIS dengan id input.\n"
            "- Jika baris 'ja' memiliki format 2 pembicara diawali '- ', pertahankan format dua baris '- ' di id_text.\n"
            "- Isi 'new_terms' hanya untuk nama segmen, permainan, julukan, atau istilah berulang baru (boleh {}).\n"
        )
    else:
        schema_instr = (
            "Balas HANYA array JSON dengan id yang SAMA PERSIS dan jumlah item "
            "yang SAMA PERSIS, format per item: "
            '{"id": <id sama>, "id_text": "<terjemahan>"}.\n'
        )
    prompt = (
        glossary_block
        + "Terjemahkan tiap item berikut ke Bahasa Indonesia sesuai style guide.\n"
        + schema_instr
        + "Jangan menggabung, memecah, menambah, atau membuang item.\n\n" + payload
    )
    if error_feedback:
        prompt = (f"Output sebelumnya SALAH ({error_feedback}). Perbaiki dan "
                  f"ikuti instruksi dengan tepat.\n\n" + prompt)
    return prompt


def _request_mapping(client, utterances, system, tracker,
                     error_feedback=None, glossary=None,
                     text_runner=None) -> tuple[dict[int, str], dict[str, str]]:
    user_prompt = _build_user_prompt(
        utterances, error_feedback=error_feedback,
        glossary=glossary, include_terms=len(utterances) > 1,
    )
    if text_runner is not None:
        response, model_used = text_runner(system, user_prompt)
    else:
        response = call_with_retries(
            lambda: client.generate(system=system, user=user_prompt))
        model_used = client.model
    if tracker is not None:
        tracker.add(model_used, response.input_tokens, response.output_tokens)
    try:
        return parse_translation_response(response.text)
    except (json.JSONDecodeError, ValueError):
        return {}, {}


def translate_batch(client, utterances: list[Utterance], system: str,
                    tracker, glossary: dict[str, str] | None = None,
                    text_runner=None) -> dict[int, str]:
    expected_ids = [u.id for u in utterances]

    mapping, new_terms = _request_mapping(
        client, utterances, system, tracker,
        glossary=glossary, text_runner=text_runner,
    )
    error = validate_alignment(mapping, expected_ids)
    if error is None:
        if glossary is not None and new_terms:
            updated = merge_glossary(glossary, new_terms)
            glossary.clear()
            glossary.update(updated)
        return mapping

    retry, retry_terms = _request_mapping(
        client, utterances, system, tracker,
        error_feedback=error, glossary=glossary, text_runner=text_runner,
    )
    if validate_alignment(retry, expected_ids) is None:
        if glossary is not None and (new_terms or retry_terms):
            updated = merge_glossary(glossary, {**new_terms, **retry_terms})
            glossary.clear()
            glossary.update(updated)
        return retry

    mapping = {**retry, **{k: v for k, v in mapping.items() if k not in retry}}
    mapping = {k: v for k, v in mapping.items() if k in set(expected_ids)}

    by_id = {u.id: u for u in utterances}
    for missing_id in sorted(set(expected_ids) - set(mapping)):
        single, _ = _request_mapping(
            client, [by_id[missing_id]], system, tracker,
            glossary=glossary, text_runner=text_runner,
        )
        if missing_id not in single:
            raise TranslateError(
                f"Gagal menerjemahkan id={missing_id} "
                f"({by_id[missing_id].ja[:40]}...) setelah fallback per-baris.")
        mapping[missing_id] = single[missing_id]

    if glossary is not None and (new_terms or retry_terms):
        updated = merge_glossary(glossary, {**new_terms, **retry_terms})
        glossary.clear()
        glossary.update(updated)
    return mapping


def _unpack_checkpoint(raw) -> tuple[dict[int, str], dict[str, str]]:
    if isinstance(raw, dict) and "translations" in raw:
        trans = {int(k): v for k, v in raw["translations"].items()}
        gloss = {str(k): str(v) for k, v in (raw.get("glossary") or {}).items()}
        return trans, gloss
    if isinstance(raw, dict):
        return {int(k): v for k, v in raw.items()}, {}
    return {}, {}


def translate_all(client, utterances: list[Utterance], system: str, tracker,
                  on_batch=None, load_checkpoint=None,
                  save_checkpoint=None, glossary: dict[str, str] | None = None,
                  text_runner=None) -> dict[int, str]:
    """Translate in BATCH_SIZE groups with optional per-batch checkpoints & glossary."""
    batches = [utterances[i:i + BATCH_SIZE]
               for i in range(0, len(utterances), BATCH_SIZE)]
    result: dict[int, str] = {}
    active_glossary: dict[str, str] = glossary if glossary is not None else {}

    for idx, batch in enumerate(batches, start=1):
        ckpt_raw = load_checkpoint(idx) if load_checkpoint else None
        if ckpt_raw is not None:
            mapping, ckpt_glossary = _unpack_checkpoint(ckpt_raw)
            if ckpt_glossary:
                merged = merge_glossary(active_glossary, ckpt_glossary)
                active_glossary.clear()
                active_glossary.update(merged)
        else:
            mapping = translate_batch(
                client, batch, system, tracker,
                glossary=active_glossary, text_runner=text_runner,
            )
            if save_checkpoint:
                save_checkpoint(idx, {
                    "translations": mapping,
                    "glossary": dict(active_glossary),
                })
        result.update(mapping)
        if on_batch:
            on_batch(idx, len(batches))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_translate.py -v`
Expected: All tests in `tests/test_translate.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/translate.py tests/test_translate.py
git commit -m "feat(translate): add rolling episode glossary and text_runner fallback hook"
```

---

### Task 5: Post-Processing 2-Turn Merge Cap (`app/subtitle.py`, `tests/test_subtitle.py`)

**Files:**
- Modify: `app/subtitle.py:8-50`
- Test: `tests/test_subtitle.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_subtitle.py`:

```python
def test_merge_short_never_chains_past_two_blocks():
    # Four rapid <0.7s blocks should NOT all collapse into one giant 4-speaker line.
    events, _ = run([
        E(0.0, 0.4, "Eh!?"),
        E(0.5, 0.9, "Serius?!"),
        E(1.0, 1.4, "Masa sih!?"),
        E(1.5, 1.9, "Keren banget!"),
    ])
    assert len(events) == 2
    assert events[0].text == "Eh!? Serius?!"
    assert events[1].text == "Masa sih!? Keren banget!"


def test_merge_short_does_not_merge_two_speaker_dashed_block():
    two_speaker = "- Eh!?\n- Serius?!"
    events, _ = run([
        E(0.0, 0.5, two_speaker),
        E(0.6, 1.0, "Gila sih!"),
    ])
    assert len(events) == 2
    assert events[0].text == two_speaker
    assert events[1].text == "Gila sih!"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_subtitle.py -k "never_chains or two_speaker_dashed" -v`
Expected: FAIL (all 4 rapid blocks chained into 1 event, and `two_speaker` merged with `"Gila sih!"`).

- [ ] **Step 3: Update `SubEvent` and `_merge_short` in `app/subtitle.py`**

In `app/subtitle.py` (lines 8-48):

```python
def _count_speaker_turns(text: str) -> int:
    dashed_lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("- ")]
    return max(1, len(dashed_lines))


@dataclass
class SubEvent:
    start: float
    end: float
    text: str
    type: str = "dialogue"  # "dialogue" | "narration"
    turns: int = 0

    def __post_init__(self) -> None:
        if self.turns <= 0:
            self.turns = _count_speaker_turns(self.text)

    @property
    def duration(self) -> float:
        return self.end - self.start


def postprocess(events: list[SubEvent], *, min_duration: float,
                max_duration: float, merge_gap: float,
                cps_threshold: float) -> tuple[list[SubEvent], list[dict]]:
    evs = sorted((SubEvent(e.start, e.end, e.text.strip(), e.type, e.turns)
                  for e in events if e.text.strip()), key=lambda e: e.start)
    evs = _merge_short(evs, min_duration, merge_gap, max_duration)
    evs = _extend_short(evs, min_duration)
    evs = _split_long(evs, max_duration)
    evs = _fix_overlaps(evs)
    return evs, _cps_flags(evs, cps_threshold)


def _merge_short(events, min_duration, merge_gap, max_duration):
    """§7: short reaction + continuation, gap < merge_gap, same type, max 2 turns -> one block."""
    result: list[SubEvent] = []
    for ev in events:
        if result:
            prev = result[-1]
            gap = ev.start - prev.end
            short_involved = (prev.duration < min_duration
                              or ev.duration < min_duration)
            within_turn_cap = (prev.turns + ev.turns) <= 2
            has_dashed = ("\n" in prev.text and "- " in prev.text) or ("\n" in ev.text and "- " in ev.text)
            if (gap < merge_gap and ev.type == prev.type and short_involved
                    and within_turn_cap and not has_dashed
                    and (ev.end - prev.start) <= max_duration):
                result[-1] = SubEvent(prev.start, ev.end,
                                      f"{prev.text} {ev.text}", prev.type,
                                      turns=prev.turns + ev.turns)
                continue
        result.append(ev)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subtitle.py -v`
Expected: All tests in `tests/test_subtitle.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/subtitle.py tests/test_subtitle.py
git commit -m "feat(subtitle): cap _merge_short at 2 turns and protect multi-speaker cues"
```

---

### Task 6: Per-Chunk Checkpointing & Fallback Propagation in `timing.py` (`app/timing.py`, `tests/test_timing.py`)

**Files:**
- Modify: `app/timing.py:78-180`
- Create: `tests/test_timing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timing.py`:

```python
import pytest

from app.audio import PlannedChunk
from app.providers import LLMResponse
from app.timing import correct_chunk_timing, reanchor_all
from app.transcribe import Utterance


class FakeTracker:
    def __init__(self):
        self.calls = []

    def add(self, model, inp, out):
        self.calls.append((model, inp, out))


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_correct_chunk_timing_propagates_api_error_when_runner_fails(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]

    def failing_runner(path, system, user):
        raise FakeHTTPError(503)

    with pytest.raises(FakeHTTPError):
        correct_chunk_timing(
            None, chunk_path, utts, 0.0, FakeTracker(),
            audio_runner=failing_runner,
        )


def test_correct_chunk_timing_returns_fallback_on_parse_mismatch(tmp_path):
    chunk_path = tmp_path / "chunk_001.mp3"
    chunk_path.write_bytes(b"mp3")
    utts = [Utterance(1, 1.0, 2.0, "dialogue", "あ")]

    def bad_json_runner(path, system, user):
        return LLMResponse(text="not json", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    corrected, success = correct_chunk_timing(
        None, chunk_path, utts, 0.0, FakeTracker(),
        audio_runner=bad_json_runner,
    )
    assert success is False
    assert corrected == utts


def test_reanchor_all_uses_per_chunk_checkpoints(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "chunk_001.mp3").write_bytes(b"a")
    (chunks_dir / "chunk_002.mp3").write_bytes(b"b")
    chunks = [
        PlannedChunk(1, 0.0, 10.0, False),
        PlannedChunk(2, 10.0, 20.0, False),
    ]
    utts = [
        Utterance(1, 1.0, 2.0, "dialogue", "一"),
        Utterance(2, 11.0, 12.0, "dialogue", "二"),
    ]
    saved = {}
    runner_calls = []

    def load_ckpt(idx):
        if idx == 1:
            return {
                "utterances": [{"id": 1, "start": 1.2, "end": 2.2, "type": "dialogue", "ja": "一"}],
                "report_entry": {"chunk": 1, "status": "corrected", "count": 1},
            }
        return None

    def save_ckpt(idx, payload):
        saved[idx] = payload

    def fake_runner(path, system, user):
        runner_calls.append(path.name)
        return LLMResponse(
            text='[{"id": 2, "start": 1.5, "end": 2.5}]',
            input_tokens=10, output_tokens=5,
        ), "gemini-3.5-flash"

    result, report = reanchor_all(
        utts, chunks, chunks_dir, gemini=None, tracker=FakeTracker(),
        audio_runner=fake_runner, load_checkpoint=load_ckpt, save_checkpoint=save_ckpt,
    )
    assert runner_calls == ["chunk_002.mp3"]  # chunk 1 skipped via checkpoint
    assert result[0].start == pytest.approx(1.2)
    assert result[1].start == pytest.approx(11.5)
    assert 2 in saved and saved[2]["report_entry"]["status"] == "corrected"
    assert [r["chunk"] for r in report] == [1, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_timing.py -v`
Expected: FAIL (`TypeError: correct_chunk_timing() got an unexpected keyword argument 'audio_runner'`).

- [ ] **Step 3: Update `correct_chunk_timing` and `reanchor_all` in `app/timing.py`**

Replace `correct_chunk_timing` and `reanchor_all` (lines 78-180 of `app/timing.py`):

```python
def correct_chunk_timing(
    gemini,
    chunk_path: Path,
    utterances: list[Utterance],
    chunk_start: float,
    tracker,
    audio_runner=None,
) -> tuple[list[Utterance], bool]:
    """Send chunk audio + current timestamps to Gemini; get corrected timestamps back.

    Raises API/transport failures (after fallback exhaustion in audio_runner).
    Returns (utterances, False) only on malformed JSON / ID mismatch.
    """
    rel_utts = [replace(u, start=round(u.start - chunk_start, 3),
                        end=round(u.end - chunk_start, 3))
                for u in utterances]

    expected_ids = [u.id for u in rel_utts]
    user = build_timing_user_prompt(rel_utts)

    if audio_runner is not None:
        response, model_used = audio_runner(chunk_path, "", user)
    else:
        audio_ref = gemini.upload_audio(chunk_path)
        response = call_with_retries(
            lambda: gemini.generate(system="", user=user, audio=audio_ref)
        )
        model_used = gemini.model

    if tracker is not None:
        tracker.add(model_used, response.input_tokens, response.output_tokens)

    corrected_map = parse_timing_response(response.text, expected_ids)
    if corrected_map is None:
        return utterances, False

    result: list[Utterance] = []
    for u in utterances:
        rel_start, rel_end = corrected_map[u.id]
        result.append(replace(
            u,
            start=round(chunk_start + rel_start, 3),
            end=round(chunk_start + rel_end, 3),
        ))
    return result, True


def reanchor_all(
    utterances: list[Utterance],
    chunks: list,
    chunks_dir: Path,
    gemini,
    tracker,
    on_progress=None,
    audio_runner=None,
    load_checkpoint=None,
    save_checkpoint=None,
) -> tuple[list[Utterance], list[dict]]:
    """Re-anchor all utterances by re-sending each chunk to Gemini."""
    chunk_map: dict[int, list[Utterance]] = {c.index: [] for c in chunks}
    for u in utterances:
        for chunk in chunks:
            if chunk.start <= u.start < chunk.end or chunk == chunks[-1]:
                chunk_map[chunk.index].append(u)
                break

    result: list[Utterance] = []
    report: list[dict] = []
    n = len(chunks)

    for chunk in chunks:
        ckpt = load_checkpoint(chunk.index) if load_checkpoint else None
        if ckpt is not None:
            loaded_utts = [Utterance.from_dict(d) for d in ckpt.get("utterances", [])]
            result.extend(loaded_utts)
            if ckpt.get("report_entry"):
                report.append(ckpt["report_entry"])
            if on_progress:
                on_progress(chunk.index, n)
            continue

        chunk_utts = chunk_map[chunk.index]
        if not chunk_utts:
            entry = {"chunk": chunk.index, "status": "skipped", "reason": "no_utterances"}
            report.append(entry)
            if save_checkpoint:
                save_checkpoint(chunk.index, {"utterances": [], "report_entry": entry})
            if on_progress:
                on_progress(chunk.index, n)
            continue

        chunk_path = chunks_dir / f"chunk_{chunk.index:03d}.mp3"
        if not chunk_path.exists():
            entry = {"chunk": chunk.index, "status": "fallback", "reason": "chunk_file_missing"}
            report.append(entry)
            result.extend(chunk_utts)
            if save_checkpoint:
                save_checkpoint(chunk.index, {
                    "utterances": [u.to_dict() for u in chunk_utts],
                    "report_entry": entry,
                })
            if on_progress:
                on_progress(chunk.index, n)
            continue

        corrected, success = correct_chunk_timing(
            gemini, chunk_path, chunk_utts, chunk.start, tracker,
            audio_runner=audio_runner,
        )

        if success:
            entry = {"chunk": chunk.index, "status": "corrected", "count": len(corrected)}
        else:
            entry = {"chunk": chunk.index, "status": "fallback", "reason": "gemini_parse_error"}
        report.append(entry)
        result.extend(corrected)

        if save_checkpoint:
            save_checkpoint(chunk.index, {
                "utterances": [u.to_dict() for u in corrected],
                "report_entry": entry,
            })

        if on_progress:
            on_progress(chunk.index, n)

    result.sort(key=lambda u: u.start)
    result = [replace(u, id=i) for i, u in enumerate(result, start=1)]
    return result, report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_timing.py -v`
Expected: All 3 tests in `tests/test_timing.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/timing.py tests/test_timing.py
git commit -m "feat(timing): add per-chunk checkpoints and propagate API errors after fallback"
```

---

### Task 7: QC Pass 1A — Free Local Acoustic & Rule Audit (`app/qc.py`, `tests/test_qc.py`)

**Files:**
- Modify: `app/qc.py`
- Create: `tests/test_qc.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_qc.py`:

```python
import json
import pytest

from app.audio import Silence
from app.qc import audit_local


def _row(id, start, end, ja="こんにちは", id_text="Halo"):
    return {"id": id, "start": start, "end": end, "type": "dialogue", "ja": ja, "id_text": id_text}


def test_audit_local_detects_rule_violations():
    rows = [
        _row(1, 0.0, 2.0, ja="あ" * 40, id_text="Halo saya mau makan"),         # rate_outlier (>18 cps JP) + pronoun_violation
        _row(2, 1.8, 3.0, ja="あいうえ", id_text="Ini masih テスト ya"),          # overlap with row 1 + untranslated_jp
        _row(3, 5.0, 6.0, ja="あ", id_text="- A\n- B\n- C"),                     # speaker_overflow (>=3 turns)
        _row(4, 10.0, 11.0, ja="あいう", id_text="x" * 30),                      # high_cps (>25)
    ]
    flags = audit_local(rows, audio_path=None, timing_report=None, cps_threshold=25.0)
    assert "rate_outlier" in flags[1]
    assert "pronoun_violation" in flags[1]
    assert "overlap" in flags[2]
    assert "untranslated_jp" in flags[2]
    assert "speaker_overflow" in flags[3]
    assert "high_cps" in flags[4]


def test_audit_local_flags_timing_fallback_chunks_and_silence(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"mp3")
    monkeypatch.setattr("app.qc.audio.detect_silences", lambda p, db: [Silence(10.0, 15.0)])
    chunks_offsets = [
        {"index": 1, "start": 0.0, "end": 8.0, "overlap_prev": False},
        {"index": 2, "start": 8.0, "end": 20.0, "overlap_prev": False},
    ]
    timing_report = [
        {"chunk": 1, "status": "corrected"},
        {"chunk": 2, "status": "fallback", "reason": "gemini_parse_error"},
    ]
    rows = [
        _row(1, 1.0, 2.0, ja="おはよう", id_text="Pagi"),
        _row(2, 11.0, 12.5, ja="どうした", id_text="Kenapa?"),  # in chunk 2 fallback AND inside silence 10..15
    ]
    flags = audit_local(
        rows, audio_path=audio_path, timing_report=timing_report,
        chunk_offsets=chunks_offsets, cps_threshold=25.0,
    )
    assert 1 not in flags
    assert "timing_fallback" in flags[2]
    assert "in_silence" in flags[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qc.py -v`
Expected: `ImportError: cannot import name 'audit_local' from 'app.qc'`

- [ ] **Step 3: Implement `audit_local` in `app/qc.py`**

Add near the top of `app/qc.py` (below `AUDIO_BUFFER_SECONDS = 1.5`):

```python
_JP_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
_PRONOUN_RE = re.compile(r"\b(saya|anda|gue|lo)\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

ACOUSTIC_FLAG_REASONS = {"timing_fallback", "in_silence", "overlap", "rate_outlier", "speaker_overflow"}


def _is_in_silence(start: float, end: float, silences: list) -> bool:
    dur = end - start
    if dur < 0.7:
        return False
    for s in silences:
        if s.start <= start + 0.05 and end <= s.end + 0.05:
            return True
    return False


def audit_local(
    rows: list[dict],
    *,
    audio_path: Path | None = None,
    timing_report: list[dict] | None = None,
    chunk_offsets: list[dict] | None = None,
    cps_threshold: float = 25.0,
) -> dict[int, list[str]]:
    """Free deterministic QC audit. Returns {row_id: [reason_codes]}."""
    flags: dict[int, list[str]] = {}

    def _flag(row_id: int, code: str) -> None:
        lst = flags.setdefault(row_id, [])
        if code not in lst:
            lst.append(code)

    # 1. Identify utterances belonging to timing fallback chunks
    fallback_chunks = {
        e["chunk"] for e in (timing_report or []) if e.get("status") == "fallback"
    }
    if fallback_chunks and chunk_offsets:
        for row in rows:
            for ch in chunk_offsets:
                if ch["start"] <= row["start"] < ch["end"] or ch == chunk_offsets[-1]:
                    if ch["index"] in fallback_chunks:
                        _flag(row["id"], "timing_fallback")
                    break

    # 2. Acoustic silence check (only when audio_path is available)
    silences = []
    if audio_path is not None and audio_path.exists():
        try:
            silences = audio.detect_silences(audio_path, -30)
        except Exception:
            silences = []

    prev_end: float | None = None
    for row in rows:
        rid = int(row["id"])
        start = float(row["start"])
        end = float(row["end"])
        dur = max(0.01, end - start)
        ja = str(row.get("ja", ""))
        id_text = str(row.get("id_text", ""))

        if silences and _is_in_silence(start, end, silences):
            _flag(rid, "in_silence")

        if prev_end is not None and prev_end > start + 0.05:
            _flag(rid, "overlap")
        prev_end = end

        ja_compact = _SPACE_RE.sub("", ja)
        jp_rate = len(ja_compact) / dur
        if (dur >= 0.4 and jp_rate > 18.0) or (dur > 3.0 and jp_rate < 1.2):
            _flag(rid, "rate_outlier")

        ja_turns = sum(1 for ln in ja.splitlines() if ln.lstrip().startswith("- "))
        id_turns = sum(1 for ln in id_text.splitlines() if ln.lstrip().startswith("- "))
        if ja_turns >= 3 or id_turns >= 3:
            _flag(rid, "speaker_overflow")

        if _JP_KANA_RE.search(id_text):
            _flag(rid, "untranslated_jp")

        if len(id_text.strip()) / dur > cps_threshold:
            _flag(rid, "high_cps")

        if _PRONOUN_RE.search(id_text):
            _flag(rid, "pronoun_violation")

    return flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qc.py -v`
Expected: All tests in `tests/test_qc.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/qc.py tests/test_qc.py
git commit -m "feat(qc): add free deterministic Pass 1A local audit"
```

---

### Task 8: QC Pass 1B — Whole-Episode Text Consistency QC (`app/qc.py`, `tests/test_qc.py`)

**Files:**
- Modify: `app/qc.py`
- Test: `tests/test_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qc.py`:

```python
from app.providers import LLMResponse
from app.qc import apply_text_qc_results, run_text_qc


def test_apply_text_qc_results_updates_translations_and_collects_audio_flags():
    rows = [
        _row(1, 0.0, 2.0, ja="人狼クイーン決定戦", id_text="Pertandingan Wolf Queen"),
        _row(2, 2.5, 4.5, ja="変な日本語", id_text="Kalimat aneh"),
    ]
    qc_items = [
        {
            "id": 1,
            "fixed_id_text": "Pertandingan Ratu Serigala",
            "needs_audio_check": False,
            "reason": "Konsistensi glosarium",
        },
        {
            "id": 2,
            "fixed_id_text": "",
            "needs_audio_check": True,
            "reason": "ja mencurigakan",
        },
    ]
    updated, issues, audio_ids = apply_text_qc_results(rows, qc_items)
    assert updated[0]["id_text"] == "Pertandingan Ratu Serigala"
    assert updated[1]["id_text"] == "Kalimat aneh"
    assert audio_ids == {2}
    assert issues[0]["type"] == "translation"
    assert issues[0]["pass"] == "text_qc"


def test_run_text_qc_single_pass_and_windowed():
    rows = [_row(i, float(i), float(i) + 0.9, ja="あ", id_text=f"Teks {i}") for i in range(1, 5)]
    calls = []

    def fake_text_runner(system, user):
        calls.append(user)
        return LLMResponse(
            text=json.dumps([
                {"id": 1, "fixed_id_text": "Teks Konsisten", "needs_audio_check": False, "reason": "Glosarium"},
                {"id": 3, "fixed_id_text": "", "needs_audio_check": True, "reason": "Cek audio"},
            ]),
            input_tokens=20, output_tokens=10,
        ), "gemini-3.6-flash"

    tracker_calls = []

    class T:
        def add(self, m, i, o):
            tracker_calls.append((m, i, o))

    fixed, issues, audio_ids = run_text_qc(
        rows, system="SYS", glossary={"A": "B"}, audit_flags={3: ["rate_outlier"]},
        tracker=T(), text_runner=fake_text_runner,
    )
    assert len(calls) == 1
    assert "A -> B" in calls[0]
    assert fixed[0]["id_text"] == "Teks Konsisten"
    assert 3 in audio_ids
    assert tracker_calls == [("gemini-3.6-flash", 20, 10)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qc.py -k "text_qc" -v`
Expected: `ImportError: cannot import name 'apply_text_qc_results' from 'app.qc'`

- [ ] **Step 3: Implement Pass 1B functions in `app/qc.py`**

Add to `app/qc.py`:

```python
TEXT_QC_SINGLE_LIMIT = 250
TEXT_QC_WINDOW = 200
TEXT_QC_OVERLAP = 30


def build_text_qc_user_prompt(
    rows: list[dict],
    glossary: dict[str, str],
    audit_flags: dict[int, list[str]],
    *,
    check_ids: set[int] | None = None,
) -> str:
    gloss_block = (
        "\n".join(f"- {jp} -> {id_val}" for jp, id_val in glossary.items())
        if glossary else "(belum ada istilah khusus)"
    )
    items = []
    for r in rows:
        rid = int(r["id"])
        dur = round(float(r["end"]) - float(r["start"]), 2)
        entry = {
            "id": rid,
            "dur": dur,
            "ja": r["ja"],
            "id_text": r.get("id_text", ""),
        }
        if check_ids is not None and rid not in check_ids:
            entry["context_only"] = True
        if rid in audit_flags:
            entry["audit_flags"] = audit_flags[rid]
        items.append(entry)

    return (
        "=== GLOSARIUM EPISODE (WAJIB KONSISTEN) ===\n"
        + gloss_block
        + "\n\n=== TRANSKRIP & TERJEMAHAN ===\n"
        + json.dumps(items, ensure_ascii=False)
        + "\n\nTugasmu (HANYA kembalikan item yang perlu diperbaiki atau dicek ulang audionya):\n"
        "1. Seragamkan terjemahan istilah berulang/segmen/nama acara di seluruh episode.\n"
        "2. Perbaiki style guide (pakai aku/kamu, tanpa sisa kana/kanji Jepang, ringkas baris >25 CPS).\n"
        "3. Pastikan maksimal 2 pembicara per baris (format '- Baris 1\\n- Baris 2').\n"
        "4. Tandai needs_audio_check=true jika teks 'ja' tampak salah dengar, aneh secara konteks, atau bertumpuk 3+ pembicara.\n\n"
        "Balas HANYA array JSON berisi item bermasalah saja (boleh [] jika semua baik):\n"
        '[{"id": <id>, "fixed_id_text": "<terjemahan baru atau string kosong>", '
        '"needs_audio_check": true|false, "reason": "<alasan singkat>"}]'
    )


def apply_text_qc_results(
    rows: list[dict],
    qc_items: list[dict],
) -> tuple[list[dict], list[dict], set[int]]:
    by_id = {int(item["id"]): item for item in qc_items if isinstance(item, dict) and "id" in item}
    updated: list[dict] = []
    issues: list[dict] = []
    audio_ids: set[int] = set()

    for row in rows:
        out = dict(row)
        rid = int(row["id"])
        item = by_id.get(rid)
        if item is not None:
            fix = str(item.get("fixed_id_text") or "").strip()
            reason = str(item.get("reason") or "")
            if fix and fix != str(row.get("id_text", "")).strip():
                issues.append({
                    "id": rid,
                    "pass": "text_qc",
                    "type": "translation",
                    "original": row.get("id_text", ""),
                    "fixed": fix,
                    "reason": reason,
                })
                out["id_text"] = fix
            if item.get("needs_audio_check"):
                audio_ids.add(rid)
                issues.append({
                    "id": rid,
                    "pass": "text_qc",
                    "type": "audio_flag",
                    "reason": reason or "ja/timing dicurigai perlu dicek audio",
                })
        updated.append(out)
    return updated, issues, audio_ids


def run_text_qc(
    rows: list[dict],
    *,
    system: str,
    glossary: dict[str, str],
    audit_flags: dict[int, list[str]],
    tracker,
    text_runner,
    checkpoint_path: Path | None = None,
) -> tuple[list[dict], list[dict], set[int]]:
    """Run Pass 1B whole-episode text consistency QC."""
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return saved["rows"], saved["issues"], set(saved["audio_ids"])

    if not rows:
        return rows, [], set()

    if len(rows) <= TEXT_QC_SINGLE_LIMIT:
        windows = [(rows, None)]
    else:
        windows = []
        step = TEXT_QC_WINDOW - TEXT_QC_OVERLAP
        for start in range(0, len(rows), step):
            slice_rows = rows[start:start + TEXT_QC_WINDOW]
            if start == 0:
                check_ids = None
            else:
                check_ids = {int(r["id"]) for r in slice_rows[TEXT_QC_OVERLAP:]}
            windows.append((slice_rows, check_ids))
            if start + TEXT_QC_WINDOW >= len(rows):
                break

    current = [dict(r) for r in rows]
    all_issues: list[dict] = []
    all_audio_ids: set[int] = set()

    for win_rows, check_ids in windows:
        # Refresh window rows from `current` so prior window fixes are visible
        by_cur = {int(r["id"]): r for r in current}
        refreshed = [by_cur[int(r["id"])] for r in win_rows]
        prompt = build_text_qc_user_prompt(refreshed, glossary, audit_flags, check_ids=check_ids)
        response, model_used = text_runner(system, prompt)
        if tracker is not None:
            tracker.add(model_used, response.input_tokens, response.output_tokens)
        qc_items = parse_qc_response(response.text)
        if check_ids is not None:
            qc_items = [it for it in qc_items if isinstance(it, dict) and int(it.get("id", -1)) in check_ids]
        current, win_issues, win_audio = apply_text_qc_results(current, qc_items)
        all_issues.extend(win_issues)
        all_audio_ids |= win_audio

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps({
                "rows": current,
                "issues": all_issues,
                "audio_ids": sorted(all_audio_ids),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return current, all_issues, all_audio_ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qc.py -v`
Expected: All tests in `tests/test_qc.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/qc.py tests/test_qc.py
git commit -m "feat(qc): implement Pass 1B whole-episode text consistency QC"
```

---

### Task 9: QC Pass 2 — Targeted/Deep Audio QC, Splits & Structured Report (`app/qc.py`, `tests/test_qc.py`)

**Files:**
- Modify: `app/qc.py`
- Test: `tests/test_qc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_qc.py`:

```python
from app.qc import (
    apply_audio_qc_results,
    build_audio_qc_windows,
    load_qc,
    run_qc_all,
    save_qc,
)


def test_build_audio_qc_windows_groups_nearby_flagged_rows_and_skips_clean():
    rows = [
        _row(1, 1.0, 2.0),
        _row(2, 3.0, 4.0),    # flagged
        _row(3, 5.0, 6.0),    # close to row 2 (gap 1s) -> same slice
        _row(4, 120.0, 122.0),
        _row(5, 130.0, 132.0), # flagged, far from 2/3 -> separate slice
    ]
    assert build_audio_qc_windows(rows, flagged_ids=set(), deep_audio_qc=False) == []
    windows = build_audio_qc_windows(rows, flagged_ids={2, 3, 5}, deep_audio_qc=False)
    assert len(windows) == 2
    assert [r["id"] for r in windows[0]["batch"]] == [2, 3]
    assert [r["id"] for r in windows[1]["batch"]] == [5]


def test_apply_audio_qc_results_applies_ja_timing_and_split():
    batch = [
        _row(1, 10.0, 12.0, ja="間違えた", id_text="Salah"),
        _row(2, 13.0, 17.0, ja="A B C D", id_text="Empat orang"),
    ]
    qc_items = [
        {
            "id": 1,
            "fixed_ja": "正しい日本語",
            "fixed_id_text": "Jepang yang benar",
            "fixed_start": 2.0,   # relative to slice_start=8.5 -> 10.5
            "fixed_end": 4.0,     # -> 12.5
            "reason": "Perbaiki ja dan geser timing",
        },
        {
            "id": 2,
            "split": [
                {"start": 4.5, "end": 6.0, "ja": "- A\n- B", "id_text": "- A\n- B"},
                {"start": 6.2, "end": 8.2, "ja": "- C\n- D", "id_text": "- C\n- D"},
            ],
            "reason": "Pecah 4 pembicara",
        },
    ]
    fixed, issues = apply_audio_qc_results(batch, qc_items, slice_start=8.5, slice_duration=10.0)
    assert len(fixed) == 3
    assert fixed[0]["ja"] == "正しい日本語"
    assert fixed[0]["id_text"] == "Jepang yang benar"
    assert fixed[0]["start"] == pytest.approx(10.5)
    assert fixed[0]["end"] == pytest.approx(12.5)
    assert fixed[1]["ja"] == "- A\n- B"
    assert fixed[1]["start"] == pytest.approx(13.0)
    assert fixed[2]["start"] == pytest.approx(14.7)
    types = {i["type"] for i in issues}
    assert {"ja", "translation", "timing", "split"} <= types


def test_run_qc_all_hybrid_skips_audio_when_zero_flags(tmp_path):
    rows = [_row(1, 1.0, 3.0, ja="おはよう", id_text="Selamat pagi")]
    text_calls = []
    audio_calls = []

    def fake_text(system, user):
        text_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    def fake_audio(path, system, user):
        audio_calls.append(1)
        return LLMResponse(text="[]", input_tokens=5, output_tokens=2), "gemini-3.6-flash"

    class T:
        def add(self, *a):
            pass

    fixed, report = run_qc_all(
        None, rows, audio_path=None, tmp_dir=tmp_path / "qc_tmp",
        system="SYS", tracker=T(),
        text_runner=fake_text, audio_runner=fake_audio,
        deep_audio_qc=False,
    )
    assert len(text_calls) == 1
    assert len(audio_calls) == 0
    assert report["summary"]["audio_slices_checked"] == 0
    assert fixed[0]["id"] == 1


def test_save_and_load_qc_structured_report(tmp_path):
    rows = [_row(1, 0.0, 1.0)]
    report = {"mode": "hybrid", "deep_audio_qc": False, "summary": {"total_rows": 1}, "issues": []}
    save_qc(rows, report, tmp_path)
    assert load_qc(tmp_path) == rows
    disk_report = json.loads((tmp_path / "qc" / "report.json").read_text(encoding="utf-8"))
    assert disk_report["mode"] == "hybrid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_qc.py -k "audio_qc or hybrid or save_and_load" -v`
Expected: `ImportError: cannot import name 'build_audio_qc_windows' from 'app.qc'`

- [ ] **Step 3: Implement Pass 2, `run_qc_all`, and structured `save_qc` in `app/qc.py`**

Update `app/qc.py` by adding the Pass 2 + orchestrator functions and updating `save_qc`:

```python
MAX_SLICE_SPAN_SECONDS = 90.0
MAX_SLICE_GAP_SECONDS = 15.0


def build_audio_qc_windows(
    rows: list[dict],
    *,
    flagged_ids: set[int],
    deep_audio_qc: bool = False,
    batch_size: int = 20,
    window_size: int = 10,
) -> list[dict]:
    """Return list of {'batch': [...], 'context_window': [...]} descriptors."""
    if not rows:
        return []
    n = len(rows)
    if deep_audio_qc:
        windows = []
        for start_i in range(0, n, batch_size):
            end_i = min(start_i + batch_size, n)
            ctx_start = max(0, start_i - window_size)
            ctx_end = min(n, end_i + window_size)
            windows.append({
                "batch": rows[start_i:end_i],
                "context_window": rows[ctx_start:ctx_end],
            })
        return windows

    flagged_indices = [i for i, r in enumerate(rows) if int(r["id"]) in flagged_ids]
    if not flagged_indices:
        return []

    groups: list[list[int]] = [[flagged_indices[0]]]
    for idx in flagged_indices[1:]:
        prev_idx = groups[-1][-1]
        first_idx = groups[-1][0]
        gap = float(rows[idx]["start"]) - float(rows[prev_idx]["end"])
        span = float(rows[idx]["end"]) - float(rows[first_idx]["start"])
        if gap <= MAX_SLICE_GAP_SECONDS and span <= MAX_SLICE_SPAN_SECONDS:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    windows = []
    for grp in groups:
        lo, hi = grp[0], grp[-1] + 1
        ctx_start = max(0, lo - 5)
        ctx_end = min(n, hi + 5)
        windows.append({
            "batch": [rows[i] for i in grp],
            "context_window": rows[ctx_start:ctx_end],
        })
    return windows


def build_audio_qc_user_prompt(
    batch: list[dict],
    context_window: list[dict],
    slice_start: float,
    glossary: dict[str, str] | None = None,
) -> str:
    context_lines = "\n".join(
        f"[{i['id']}] {i['start']:.2f}-{i['end']:.2f}s | {i['ja']} | {i.get('id_text','')}"
        for i in context_window
    )
    batch_payload = json.dumps([
        {
            "id": i["id"],
            "start": round(float(i["start"]) - slice_start, 3),
            "end": round(float(i["end"]) - slice_start, 3),
            "ja": i["ja"],
            "id_text": i.get("id_text", ""),
        }
        for i in batch
    ], ensure_ascii=False)
    gloss_text = (
        "\n".join(f"- {jp} -> {id_val}" for jp, id_val in (glossary or {}).items())
        if glossary else "(kosong)"
    )

    return (
        "=== GLOSARIUM EPISODE ===\n"
        + gloss_text
        + "\n\n=== KONTEKS SUBTITLE SEKITAR (jangan ubah) ===\n"
        + context_lines
        + "\n\n=== ITEM UNTUK DIAUDIT DENGAN AUDIO ===\n"
        f"Audio terlampir dimulai dari detik {slice_start:.2f} video asli.\n"
        "Timestamp start/end di bawah relatif terhadap awal audio terlampir.\n\n"
        + batch_payload
        + "\n\nDengarkan audio dengan teliti. Untuk item yang bermasalah saja, perbaiki langsung:\n"
        "- fixed_ja: perbaiki jika Jepang salah dengar/nama member salah (kosongkan jika sudah benar)\n"
        "- fixed_id_text: perbaiki terjemahan agar akurat & sesuai glosarium (kosongkan jika sudah benar)\n"
        "- fixed_start / fixed_end: angka detik relatif terhadap audio terlampir jika timing meleset (atau null)\n"
        "- split: jika 1 item berisi 3+ pembicara, pecah menjadi 2-4 objek "
        '[{"start": <detik>, "end": <detik>, "ja": "...", "id_text": "..."}] (maks 2 pembicara per item)\n\n'
        "Balas HANYA array JSON (boleh [] jika semua sudah akurat):\n"
        '[{"id": <id>, "fixed_ja": "", "fixed_id_text": "", '
        '"fixed_start": null, "fixed_end": null, "split": null, "reason": "..."}]'
    )


def apply_audio_qc_results(
    batch_rows: list[dict],
    qc_results: list[dict],
    *,
    slice_start: float,
    slice_duration: float,
) -> tuple[list[dict], list[dict]]:
    """Apply audio QC fixes (ja, id_text, timing, splits) to batch_rows."""
    by_id = {int(r["id"]): r for r in qc_results if isinstance(r, dict) and "id" in r}
    fixed: list[dict] = []
    issues: list[dict] = []
    max_rel = slice_duration + 0.5

    for row in batch_rows:
        rid = int(row["id"])
        qc = by_id.get(rid)
        if qc is None:
            fixed.append(dict(row))
            continue

        reason = str(qc.get("reason") or "")
        split_items = qc.get("split")
        if isinstance(split_items, list) and len(split_items) >= 2:
            valid_parts: list[dict] = []
            for part in split_items[:4]:
                if not isinstance(part, dict):
                    continue
                try:
                    ps = float(part["start"])
                    pe = float(part["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                p_ja = str(part.get("ja") or "").strip()
                p_id = str(part.get("id_text") or "").strip()
                if 0.0 <= ps < pe <= max_rel and p_ja and p_id:
                    valid_parts.append({
                        "id": rid,
                        "start": round(slice_start + ps, 3),
                        "end": round(slice_start + pe, 3),
                        "type": row.get("type", "dialogue"),
                        "ja": p_ja,
                        "id_text": p_id,
                    })
            if len(valid_parts) >= 2:
                fixed.extend(valid_parts)
                issues.append({
                    "id": rid,
                    "pass": "audio_qc",
                    "type": "split",
                    "original": row.get("id_text", ""),
                    "fixed": f"Dipecah menjadi {len(valid_parts)} baris",
                    "reason": reason,
                })
                continue

        out = dict(row)
        f_ja = str(qc.get("fixed_ja") or "").strip()
        if f_ja and f_ja != str(row.get("ja", "")).strip():
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "ja",
                "original": row.get("ja", ""),
                "fixed": f_ja,
                "reason": reason,
            })
            out["ja"] = f_ja

        f_id = str(qc.get("fixed_id_text") or "").strip()
        if f_id and f_id != str(row.get("id_text", "")).strip():
            issues.append({
                "id": rid,
                "pass": "audio_qc",
                "type": "translation",
                "original": row.get("id_text", ""),
                "fixed": f_id,
                "reason": reason,
            })
            out["id_text"] = f_id

        fs = qc.get("fixed_start")
        fe = qc.get("fixed_end")
        if fs is not None and fe is not None:
            try:
                rel_s, rel_e = float(fs), float(fe)
                if 0.0 <= rel_s < rel_e <= max_rel:
                    abs_s = round(slice_start + rel_s, 3)
                    abs_e = round(slice_start + rel_e, 3)
                    if abs_s != row["start"] or abs_e != row["end"]:
                        issues.append({
                            "id": rid,
                            "pass": "audio_qc",
                            "type": "timing",
                            "original": f"{row['start']:.2f}-{row['end']:.2f}s",
                            "fixed": f"{abs_s:.2f}-{abs_e:.2f}s",
                            "reason": reason,
                        })
                        out["start"] = abs_s
                        out["end"] = abs_e
                else:
                    issues.append({
                        "id": rid,
                        "pass": "audio_qc",
                        "type": "warning",
                        "reason": f"Timestamp QC di luar batas slice diabaikan ({rel_s:.2f}-{rel_e:.2f}s)",
                    })
            except (TypeError, ValueError):
                pass

        fixed.append(out)
    return fixed, issues


def run_qc_all(
    gemini,
    rows: list[dict],
    audio_path: Path | None,
    tmp_dir: Path,
    system: str,
    tracker,
    batch_size: int = 20,
    window_size: int = 10,
    on_batch=None,
    *,
    glossary: dict[str, str] | None = None,
    timing_report: list[dict] | None = None,
    chunk_offsets: list[dict] | None = None,
    cps_threshold: float = 25.0,
    deep_audio_qc: bool = False,
    text_runner=None,
    audio_runner=None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Two-Pass Hybrid QC: Pass 1A local audit + Pass 1B text QC + Pass 2 targeted/deep audio QC."""
    if text_runner is None and gemini is not None:
        def text_runner(sys_p, usr_p):
            resp = call_with_retries(lambda: gemini.generate(system=sys_p, user=usr_p))
            return resp, gemini.model

    if audio_runner is None and gemini is not None:
        def audio_runner(slice_p, sys_p, usr_p):
            ref = gemini.upload_audio(slice_p)
            resp = call_with_retries(lambda: gemini.generate(system=sys_p, user=usr_p, audio=ref))
            return resp, gemini.model

    # Pass 1A: free local audit
    audit_flags = audit_local(
        rows,
        audio_path=audio_path,
        timing_report=timing_report,
        chunk_offsets=chunk_offsets,
        cps_threshold=cps_threshold,
    )

    # Pass 1B: episode-wide text consistency QC
    pass1_ckpt = (checkpoint_dir / "qc_pass1.json") if checkpoint_dir is not None else None
    if text_runner is not None:
        working_rows, pass1_issues, text_audio_ids = run_text_qc(
            rows,
            system=system,
            glossary=glossary or {},
            audit_flags=audit_flags,
            tracker=tracker,
            text_runner=text_runner,
            checkpoint_path=pass1_ckpt,
        )
    else:
        working_rows = [dict(r) for r in rows]
        pass1_issues = []
        text_audio_ids = set()

    # Pass 2: targeted or deep audio QC
    all_issues = list(pass1_issues)
    acoustic_flagged_ids = {
        rid for rid, reasons in audit_flags.items()
        if any(r in ACOUSTIC_FLAG_REASONS for r in reasons)
    }
    flagged_ids = acoustic_flagged_ids | text_audio_ids

    audio_slices_checked = 0
    if audio_path is not None and audio_runner is not None:
        windows = build_audio_qc_windows(
            working_rows,
            flagged_ids=flagged_ids,
            deep_audio_qc=deep_audio_qc,
            batch_size=batch_size,
            window_size=window_size,
        )
        tmp_dir.mkdir(parents=True, exist_ok=True)
        n_win = len(windows)
        by_id_replacements: dict[int, list[dict]] = {}

        for w_idx, win in enumerate(windows):
            ckpt_file = (
                checkpoint_dir / f"qc_audio_{w_idx:03d}.json"
                if checkpoint_dir is not None else None
            )
            if ckpt_file is not None and ckpt_file.exists():
                saved = json.loads(ckpt_file.read_text(encoding="utf-8"))
                slice_fixed = saved["fixed"]
                slice_issues = saved["issues"]
                batch_ids = saved["batch_ids"]
            else:
                batch = win["batch"]
                batch_ids = [int(r["id"]) for r in batch]
                slice_path = tmp_dir / f"qc_slice_{w_idx:03d}.mp3"
                slice_start = max(0.0, float(batch[0]["start"]) - AUDIO_BUFFER_SECONDS)
                slice_end = float(batch[-1]["end"]) + AUDIO_BUFFER_SECONDS
                slice_duration = max(0.5, slice_end - slice_start)
                audio.cut_chunk(audio_path, slice_path, slice_start, slice_end)
                try:
                    user = build_audio_qc_user_prompt(
                        batch, win["context_window"], slice_start, glossary=glossary,
                    )
                    response, model_used = audio_runner(slice_path, system, user)
                    if tracker is not None:
                        tracker.add(model_used, response.input_tokens, response.output_tokens)
                    qc_results = parse_qc_response(response.text)
                    slice_fixed, slice_issues = apply_audio_qc_results(
                        batch, qc_results,
                        slice_start=slice_start, slice_duration=slice_duration,
                    )
                finally:
                    slice_path.unlink(missing_ok=True)

                if ckpt_file is not None:
                    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_file.write_text(
                        json.dumps({
                            "batch_ids": batch_ids,
                            "fixed": slice_fixed,
                            "issues": slice_issues,
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

            audio_slices_checked += 1
            all_issues.extend(slice_issues)
            # Map original batch IDs to replacement items (supporting splits)
            if len(slice_fixed) == len(batch_ids) and all(
                int(sf["id"]) == bid for sf, bid in zip(slice_fixed, batch_ids)
            ):
                for sf, bid in zip(slice_fixed, batch_ids):
                    by_id_replacements[bid] = [sf]
            else:
                grouped: dict[int, list[dict]] = {bid: [] for bid in batch_ids}
                for sf in slice_fixed:
                    grouped.setdefault(int(sf["id"]), []).append(sf)
                by_id_replacements.update(grouped)

            if on_batch:
                on_batch(w_idx + 1, n_win)

        rebuilt: list[dict] = []
        for r in working_rows:
            rid = int(r["id"])
            if rid in by_id_replacements:
                rebuilt.extend(by_id_replacements[rid])
            else:
                rebuilt.append(r)
        working_rows = rebuilt

    # Sort and re-index sequentially
    working_rows.sort(key=lambda r: float(r["start"]))
    final_rows = [{**r, "id": i} for i, r in enumerate(working_rows, start=1)]

    report = {
        "mode": "deep" if deep_audio_qc else ("text_only" if audio_path is None else "hybrid"),
        "deep_audio_qc": deep_audio_qc,
        "summary": {
            "total_rows": len(final_rows),
            "local_flags": len(audit_flags),
            "audio_slices_checked": audio_slices_checked,
            "translation_fixes": sum(1 for i in all_issues if i.get("type") == "translation"),
            "ja_fixes": sum(1 for i in all_issues if i.get("type") == "ja"),
            "timing_fixes": sum(1 for i in all_issues if i.get("type") == "timing"),
            "splits": sum(1 for i in all_issues if i.get("type") == "split"),
            "unresolved_warnings": sum(1 for i in all_issues if i.get("type") in {"warning", "audio_flag"}),
        },
        "issues": all_issues,
    }
    return final_rows, report


def save_qc(rows: list[dict], report_or_issues: dict | list[dict], out_dir: Path) -> None:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(exist_ok=True)
    if isinstance(report_or_issues, list):
        report = {
            "mode": "hybrid",
            "deep_audio_qc": False,
            "summary": {
                "total_rows": len(rows),
                "local_flags": 0,
                "audio_slices_checked": 0,
                "translation_fixes": sum(1 for i in report_or_issues if i.get("type") == "translation"),
                "ja_fixes": sum(1 for i in report_or_issues if i.get("type") == "ja"),
                "timing_fixes": sum(1 for i in report_or_issues if i.get("type") == "timing"),
                "splits": sum(1 for i in report_or_issues if i.get("type") == "split"),
                "unresolved_warnings": sum(1 for i in report_or_issues if i.get("type") in {"warning", "audio_flag"}),
            },
            "issues": report_or_issues,
        }
    else:
        report = report_or_issues
    (qc_dir / "utterances.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (qc_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
```

Also check that the old `apply_qc_results` and `run_qc_all` imports in `app/qc.py` replace the previous definitions cleanly without duplicates, and keep `load_qc` intact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_qc.py -v`
Expected: All tests in `tests/test_qc.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/qc.py tests/test_qc.py
git commit -m "feat(qc): add Pass 2 targeted/deep audio repair, splits, and structured report"
```

---

### Task 10: Pipeline Orchestration Wiring (`app/pipeline.py`, `tests/test_pipeline.py`, `tests/test_pipeline_modes.py`)

**Files:**
- Modify: `app/pipeline.py:34-183,290-608`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_pipeline_modes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_failure_records_failed_stage_and_emits_it(env, monkeypatch):
    def boom(client, utterances, system, tracker, **kw):
        raise RuntimeError("translate 503")

    monkeypatch.setattr("app.translate.translate_batch", boom)
    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert job.failed_stage == "translate"
    assert job.events[-1]["stage"] == "translate"
    assert job.events[-1]["status"] == "error"
    assert job.events[-1]["failed_stage"] == "translate"
    loaded = pipeline.load_job(job.id)
    assert loaded.failed_stage == "translate"


def test_pipeline_runs_text_qc_in_translation_only_mode(env, monkeypatch):
    qc_calls = []

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        qc_calls.append({"audio_path": audio_path, "has_text_runner": kw.get("text_runner") is not None})
        return rows, {"mode": "text_only", "deep_audio_qc": False, "summary": {"translation_fixes": 0}, "issues": []}

    monkeypatch.setattr("app.qc.run_qc_all", spy_qc)
    job = pipeline.create_job(make_params(mode="translation_only", source="file", url=None))
    (job.dir / "transcript_jp.json").write_text(
        json.dumps([{"id": 1, "start": 0.0, "end": 1.5, "type": "dialogue", "ja": "こんにちは"}]),
        encoding="utf-8",
    )
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert len(qc_calls) == 1
    assert qc_calls[0]["audio_path"] is None
    assert qc_calls[0]["has_text_runner"] is True
```

Also update `tests/test_pipeline.py` mocks so existing mocked tests accept the new keyword arguments:
- Line 39 (`fake_translate_batch`):
```python
    def fake_translate_batch(client, utterances, system, tracker,
                             glossary=None, text_runner=None):
        counters["translate"] += 1
        return {u.id: f"ID-{u.id}" for u in utterances}
```
- Line 53 (`timing.reanchor_all` mock):
```python
    monkeypatch.setattr("app.timing.reanchor_all",
                        lambda utterances, chunks, chunks_dir, gemini, tracker, **kw: (utterances, []))
```
- Lines 125 and 137 inside `test_failure_marks_job_and_retry_resumes`:
```python
    def boom(client, utterances, system, tracker, **kw):
        raise RuntimeError("translasi meledak")
...
    monkeypatch.setattr("app.translate.translate_batch",
                        lambda c, u, s, t, **kw: {x.id: "OK" for x in u})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k "failed_stage or text_qc_in_translation_only" -v`
Expected: FAIL (`AttributeError: 'Job' object has no attribute 'failed_stage'`).

- [ ] **Step 3: Update `app/pipeline.py`**

1. Update `Job`, `load_job`, `_save`, `_emit`, and `run_job` error handling in `app/pipeline.py` (lines 34-183):

```python
@dataclass
class Job:
    id: str
    dir: Path
    params: dict
    status: str = "queued"  # queued | running | done | failed | cancelled
    error: str | None = None
    failed_stage: str | None = None
    events: list[dict] = field(default_factory=list)
    cancelled: bool = False
```

In `load_job`:

```python
    job = Job(id=meta["id"], dir=OUTPUT_DIR / job_id, params=meta["params"],
              status=meta["status"], error=meta.get("error"),
              failed_stage=meta.get("failed_stage"),
              events=meta.get("events", []))
```

In `_save`:

```python
    payload = {"id": job.id, "params": job.params, "status": job.status,
               "error": job.error, "failed_stage": job.failed_stage,
               "events": job.events}
```

In `_emit`:

```python
def _emit(job: Job, stage: str, status: str, message: str | None = None,
          result: dict | None = None, failed_stage: str | None = None) -> None:
    event: dict = {"stage": stage, "status": status}
    if message:
        event["message"] = message
    if result is not None:
        event["result"] = result
    if failed_stage is not None:
        event["failed_stage"] = failed_stage
    job.events.append(event)
    _save(job)
```

In `run_job`: reset `job.status, job.error, job.failed_stage = "running", None, None`, and in `except Exception as exc:`:

```python
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        job.failed_stage = stage
        _emit(job, stage, "error", message=str(exc), failed_stage=stage)
```

2. Add fallback helper builders in `app/pipeline.py`:

```python
def _parse_audio_fallback(params: dict) -> list[providers.AudioFallbackSlot]:
    return [
        providers.AudioFallbackSlot(key_label=s["key_label"], model=s["model"])
        for s in (params.get("audio_fallback") or [])[:3]
        if s.get("key_label") and s.get("model")
    ]


def _parse_translate_fallback(params: dict) -> list[providers.TranslateFallbackSlot]:
    return [
        providers.TranslateFallbackSlot(
            provider=s["provider"],
            model=s["model"],
            key_label=s.get("key_label"),
        )
        for s in (params.get("translate_fallback") or [])[:3]
        if s.get("provider") and s.get("model")
    ]


def _make_audio_runner(job: Job, cfg, stage: str, primary_model: str, primary_client=None):
    primary_slot = providers.AudioFallbackSlot(
        key_label=providers.get_active_gemini_label(),
        model=primary_model,
    )
    fallback_slots = _parse_audio_fallback(job.params)

    def _build(slot: providers.AudioFallbackSlot):
        if primary_client is not None and slot == primary_slot and hasattr(primary_client, "upload_audio"):
            return primary_client
        return providers.make_audio_client(cfg, model=slot.model, key_label=slot.key_label)

    def _resolve_key(slot: providers.AudioFallbackSlot) -> str:
        return providers.resolve_gemini_key(slot.key_label, fallback_key=cfg.gemini_api_key)

    def runner(chunk_path: Path, system: str, user: str, on_upload_done=None):
        def _on_fb(idx: int, slot: providers.AudioFallbackSlot, exc: Exception):
            _emit(job, stage, "progress",
                  f"{chunk_path.name} error ({exc}) -> mencoba Fallback {idx}: "
                  f"key {slot.key_label} / {slot.model}")

        resp, winning = providers.execute_audio_with_fallback(
            chunk_path=chunk_path,
            primary=primary_slot,
            fallback_slots=fallback_slots,
            build_client=_build,
            resolve_key=_resolve_key,
            call_fn=lambda c, ref: c.generate(system=system, user=user, audio=ref),
            on_upload_done=on_upload_done,
            on_fallback=_on_fb,
        )
        return resp, winning.model

    return runner


def _make_text_runner(job: Job, cfg, stage: str, primary_client=None):
    provider = job.params.get("translator", "gemini")
    primary_model = job.params.get("translate_model") or cfg.translate_models.get(provider, "")
    primary_slot = providers.TranslateFallbackSlot(
        provider=provider,
        model=primary_model,
        key_label=providers.get_active_gemini_label() if provider == "gemini" else None,
    )
    fallback_slots = _parse_translate_fallback(job.params)

    def _build(slot: providers.TranslateFallbackSlot):
        if primary_client is not None and slot == primary_slot and hasattr(primary_client, "generate"):
            return primary_client
        return providers.make_text_client(cfg, slot.provider, model=slot.model, key_label=slot.key_label)

    def runner(system: str, user: str):
        def _on_fb(idx: int, slot: providers.TranslateFallbackSlot, exc: Exception):
            key_info = f" ({slot.key_label})" if slot.key_label else ""
            _emit(job, stage, "progress",
                  f"error ({exc}) -> mencoba Fallback {idx}: {slot.provider}{key_info} / {slot.model}")

        resp, winning = providers.execute_with_fallback(
            primary=primary_slot,
            fallback_slots=fallback_slots,
            build_client=_build,
            call_fn=lambda c: c.generate(system=system, user=user),
            on_fallback=_on_fb,
        )
        return resp, winning.model

    return runner
```

3. Update `_stage_transcribe` (line 307 + line 348):

```python
    primary_model = job.params.get("transcribe_model") or cfg.transcribe_model
    gemini = providers.make_transcriber(cfg, model=primary_model)
    audio_runner = _make_audio_runner(job, cfg, "transcribe", primary_model, primary_client=gemini)
```
and inside the `try:` block:
```python
            try:
                utts = transcribe.transcribe_chunk(
                    gemini, chunk_path, system, user, tracker,
                    on_upload_done=_on_upload_done,
                    audio_runner=audio_runner,
                )
```

4. Update `_stage_translate` (lines 450-482):

```python
    client = providers.make_translator(
        cfg, job.params["translator"],
        model=job.params.get("translate_model") or None,
    )
    text_runner = _make_text_runner(job, cfg, "translate", primary_client=client)
    glossary: dict[str, str] = {}
    _emit(job, "translate", "start",
          f"terjemahkan {len(utterances)} ujaran via "
          f"{job.params['translator']}")

    def load_ckpt(batch_index: int):
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def save_ckpt(batch_index: int, payload) -> None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def on_batch(done: int, total: int) -> None:
        _emit(job, "translate", "progress", f"batch {done}/{total}")

    mapping = translate.translate_all(client, utterances, system, tracker,
                                      on_batch=on_batch,
                                      load_checkpoint=load_ckpt,
                                      save_checkpoint=save_ckpt,
                                      glossary=glossary,
                                      text_runner=text_runner)
    (job.dir / "episode_glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{**u.to_dict(), "id_text": mapping[u.id]} for u in utterances]
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    _emit(job, "translate", "done", f"{len(rows)} baris")
    return rows
```

5. Replace `_stage_timing`, `_stage_qc`, and update `_stage_format` (lines 484-608):

```python
def _stage_timing(job: Job, cfg, tracker, rows: list[dict],
                  chunks, audio_path: Path | None) -> list[dict]:
    timing_path = job.dir / "timing" / "utterances.json"
    if timing_path.exists():
        result = timing.load_timing(job.dir)
        _emit(job, "timing", "done", f"{len(result)} ujaran (checkpoint)")
        return result

    if chunks is None or audio_path is None:
        _emit(job, "timing", "done", "dilewati (translation_only mode)")
        return rows

    n = len(chunks)
    timing_model = job.params.get("timing_model") or cfg.timing_model
    _emit(job, "timing", "start",
          f"koreksi timing {len(rows)} ujaran via {timing_model} ({n} chunk)")

    gemini = providers.make_audio_client(cfg, model=timing_model)
    runner4 = _make_audio_runner(job, cfg, "timing", timing_model, primary_client=gemini)
    utterances = [transcribe.Utterance.from_dict(r) for r in rows]

    def on_progress(chunk_idx: int, total: int) -> None:
        _emit(job, "timing", "progress", f"chunk {chunk_idx}/{total}")

    def load_ckpt(chunk_idx: int):
        p = job.dir / "chunks" / f"timing_{chunk_idx:03d}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def save_ckpt(chunk_idx: int, payload: dict) -> None:
        p = job.dir / "chunks" / f"timing_{chunk_idx:03d}.json"
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    corrected, report = timing.reanchor_all(
        utterances, chunks, job.dir / "chunks",
        gemini=gemini, tracker=tracker, on_progress=on_progress,
        audio_runner=lambda p, s, u: runner4(p, s, u, None),
        load_checkpoint=load_ckpt, save_checkpoint=save_ckpt,
    )

    corrected_by_id = {u.id: u for u in corrected}
    result = []
    for r in rows:
        u = corrected_by_id.get(r["id"])
        if u:
            result.append({**r, "start": u.start, "end": u.end})
        else:
            result.append(r)

    timing.save_timing(result, report, job.dir)

    n_corrected = sum(1 for e in report if e.get("status") == "corrected")
    n_fallback = sum(1 for e in report if e.get("status") == "fallback")
    _emit(job, "timing", "done",
          f"{n_corrected} chunk dikoreksi, {n_fallback} fallback ke timestamp asli")
    return result


def _stage_qc(job: Job, cfg, tracker, rows: list[dict],
              audio_path: Path | None) -> list[dict]:
    qc_path = job.dir / "qc" / "utterances.json"
    if qc_path.exists():
        result = qc.load_qc(job.dir)
        _emit(job, "qc", "done", f"{len(result)} ujaran (checkpoint)")
        return result

    deep_audio_qc = bool(job.params.get("deep_audio_qc", False))
    qc_model = job.params.get("qc_model") or cfg.qc_model
    mode_label = "deep" if (deep_audio_qc and audio_path is not None) else ("hybrid" if audio_path is not None else "text-only")
    _emit(job, "qc", "start", f"QC {len(rows)} ujaran (mode={mode_label})")

    context_md, members_md = _read_context(
        job.params.get("group"), job.params.get("context_preset")
    )
    if job.params.get("context_override"):
        context_md = job.params["context_override"]
    additional_context = job.params.get("additional_context") or None

    system = qc.build_qc_system(context_md, members_md,
                                additional_context=additional_context)

    glossary_path = job.dir / "episode_glossary.json"
    glossary = json.loads(glossary_path.read_text(encoding="utf-8")) if glossary_path.exists() else {}

    timing_report_path = job.dir / "timing" / "report.json"
    timing_report = json.loads(timing_report_path.read_text(encoding="utf-8")) if timing_report_path.exists() else []

    offsets_path = job.dir / "offsets.json"
    chunk_offsets = json.loads(offsets_path.read_text(encoding="utf-8")) if offsets_path.exists() else []

    text_client = providers.make_text_client(
        cfg, job.params.get("translator", "gemini"),
        model=job.params.get("translate_model") or None,
    )
    text_runner = _make_text_runner(job, cfg, "qc", primary_client=text_client)

    audio_client = None
    audio_runner_3 = None
    if audio_path is not None:
        audio_client = providers.make_audio_client(cfg, model=qc_model)
        runner4 = _make_audio_runner(job, cfg, "qc", qc_model, primary_client=audio_client)
        audio_runner_3 = lambda p, s, u: runner4(p, s, u, None)

    def on_batch(done: int, total: int) -> None:
        _emit(job, "qc", "progress", f"audio slice {done}/{total}")

    fixed_rows, report_or_issues = qc.run_qc_all(
        audio_client, rows, audio_path, job.dir / "chunks" / "qc_tmp",
        system, tracker,
        batch_size=cfg.qc_batch_size,
        window_size=cfg.qc_window_size,
        on_batch=on_batch,
        glossary=glossary,
        timing_report=timing_report,
        chunk_offsets=chunk_offsets,
        cps_threshold=cfg.sub_cps_flag,
        deep_audio_qc=deep_audio_qc,
        text_runner=text_runner,
        audio_runner=audio_runner_3,
        checkpoint_dir=job.dir / "chunks",
    )

    qc.save_qc(fixed_rows, report_or_issues, job.dir)
    if isinstance(report_or_issues, dict):
        summary = report_or_issues.get("summary", {})
        report_payload = report_or_issues
    else:
        summary = {
            "translation_fixes": sum(1 for i in report_or_issues if i.get("type") == "translation"),
            "ja_fixes": sum(1 for i in report_or_issues if i.get("type") == "ja"),
            "timing_fixes": sum(1 for i in report_or_issues if i.get("type") == "timing"),
            "splits": sum(1 for i in report_or_issues if i.get("type") == "split"),
            "unresolved_warnings": sum(1 for i in report_or_issues if i.get("type") in {"warning", "audio_flag"}),
        }
        report_payload = {"mode": mode_label, "deep_audio_qc": deep_audio_qc, "summary": summary, "issues": report_or_issues}

    _emit(
        job, "qc", "done",
        f"{summary.get('translation_fixes', 0)} terjemahan, "
        f"{summary.get('ja_fixes', 0)} JP, "
        f"{summary.get('timing_fixes', 0)} timing, "
        f"{summary.get('splits', 0)} split diperbaiki",
        result={"report": report_payload},
    )
    return fixed_rows
```

And inside `_stage_format` (right before `if (job.dir / "source.mp4").exists():`):

```python
    if (job.dir / "qc" / "report.json").exists():
        files.append("qc_report.json")
```

- [ ] **Step 4: Run pipeline tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_pipeline_modes.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py tests/test_pipeline_modes.py
git commit -m "feat(pipeline): wire fallback runners, glossary, timing/qc checkpoints, and failed_stage"
```

---

### Task 11: FastAPI Endpoints, Validation & Download Mapping (`app/main.py`, `tests/test_api.py`)

**Files:**
- Modify: `app/main.py:49-56,120-266,267-366`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def test_download_qc_report_maps_to_qc_dir(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    (job.dir / "qc").mkdir(parents=True, exist_ok=True)
    (job.dir / "qc" / "report.json").write_text('{"summary": {"total_rows": 1}}', encoding="utf-8")
    resp = client.get(f"/api/jobs/{job_id}/files/qc_report.json")
    assert resp.status_code == 200
    assert "total_rows" in resp.text


def test_create_job_validates_and_stores_fallbacks_and_qc_options(client, fake_run, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_WORK", "g-work")
    audio_fb = json.dumps([{"key_label": "WORK", "model": "gemini-3.5-flash"}])
    tl_fb = json.dumps([{"provider": "gemini", "key_label": "WORK", "model": "gemini-2.5-flash"}])
    resp = post_url_job(
        client,
        audio_fallback=audio_fb,
        translate_fallback=tl_fb,
        timing_model="gemini-3.5-flash",
        qc_model="gemini-3.5-flash",
        deep_audio_qc="true",
    )
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["audio_fallback"] == [{"key_label": "WORK", "model": "gemini-3.5-flash"}]
    assert job.params["translate_fallback"] == [{"provider": "gemini", "key_label": "WORK", "model": "gemini-2.5-flash"}]
    assert job.params["timing_model"] == "gemini-3.5-flash"
    assert job.params["qc_model"] == "gemini-3.5-flash"
    assert job.params["deep_audio_qc"] is True


def test_create_job_rejects_more_than_three_fallback_slots(client):
    four = json.dumps([{"key_label": "Default", "model": "gemini-3.6-flash"}] * 4)
    resp = post_url_job(client, audio_fallback=four)
    assert resp.status_code == 422


def test_create_job_rejects_unknown_fallback_model_or_key(client):
    bad_key = json.dumps([{"key_label": "MISSING", "model": "gemini-3.6-flash"}])
    assert post_url_job(client, audio_fallback=bad_key).status_code == 422
    bad_model = json.dumps([{"key_label": "Default", "model": "not-a-model"}])
    assert post_url_job(client, audio_fallback=bad_model).status_code == 422


def test_retry_accepts_timing_qc_and_fallback_overrides(client, fake_run, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={
            "transcribe_model": "gemini-3.5-flash",
            "timing_model": "gemini-2.5-flash",
            "qc_model": "gemini-2.5-flash",
            "deep_audio_qc": True,
            "audio_fallback": [{"key_label": "ALT", "model": "gemini-2.5-flash"}],
        },
    )
    assert resp.status_code == 200
    params = pipeline.JOBS[job_id].params
    assert params["transcribe_model"] == "gemini-3.5-flash"
    assert params["timing_model"] == "gemini-2.5-flash"
    assert params["qc_model"] == "gemini-2.5-flash"
    assert params["deep_audio_qc"] is True
    assert params["audio_fallback"] == [{"key_label": "ALT", "model": "gemini-2.5-flash"}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py -k "qc_report or fallbacks or more_than_three or timing_qc" -v`
Expected: FAIL (`404` for `qc_report.json` and `audio_fallback` not stored/validated).

- [ ] **Step 3: Update `app/main.py`**

1. Add `"qc_report.json"` to `DOWNLOADABLE` (line 50).
2. Add validation helper functions in `app/main.py`:

```python
def _validate_audio_fallback(raw_slots: list[dict], models_cfg: dict) -> list[dict]:
    if len(raw_slots) > 3:
        raise HTTPException(422, "audio_fallback maksimal 3 slot.")
    keys = providers.get_all_gemini_keys()
    valid_models = set(models_cfg["transcription"])
    cleaned = []
    for slot in raw_slots:
        if not isinstance(slot, dict):
            raise HTTPException(422, "Format slot audio_fallback tidak valid.")
        k = (slot.get("key_label") or "").strip()
        m = (slot.get("model") or "").strip()
        if not k and not m:
            continue
        if k not in keys:
            raise HTTPException(422, f"Gemini key '{k}' pada audio_fallback tidak ditemukan.")
        if m not in valid_models:
            raise HTTPException(422, f"Model '{m}' pada audio_fallback tidak dikenal.")
        cleaned.append({"key_label": k, "model": m})
    return cleaned


def _validate_translate_fallback(raw_slots: list[dict], models_cfg: dict, cfg: providers.Config) -> list[dict]:
    if len(raw_slots) > 3:
        raise HTTPException(422, "translate_fallback maksimal 3 slot.")
    keys = providers.get_all_gemini_keys()
    avail = cfg.available_providers()
    cleaned = []
    for slot in raw_slots:
        if not isinstance(slot, dict):
            raise HTTPException(422, "Format slot translate_fallback tidak valid.")
        p = (slot.get("provider") or "").strip()
        m = (slot.get("model") or "").strip()
        k = (slot.get("key_label") or "").strip() or None
        if not p and not m:
            continue
        if not avail.get(p):
            raise HTTPException(422, f"Provider '{p}' pada translate_fallback belum aktif.")
        if m not in models_cfg["translation"].get(p, []):
            raise HTTPException(422, f"Model '{m}' tidak valid untuk provider '{p}'.")
        if p == "gemini":
            resolved_k = k or providers.get_active_gemini_label()
            if resolved_k not in keys:
                raise HTTPException(422, f"Gemini key '{resolved_k}' pada translate_fallback tidak ditemukan.")
            k = resolved_k
        else:
            k = None
        cleaned.append({"provider": p, "key_label": k, "model": m})
    return cleaned
```

3. Update `create_job`, `list_sessions`, `job_status`, `download`, `RetryRequest`, and `retry` in `app/main.py`:

In `list_sessions` (line 119):
```python
        sessions.append({
            "id": job_id,
            "label": f"{ts} · {display_name} · {status}",
            "status": status,
            "mode": mode,
            "mode_label": _MODE_LABELS.get(mode, mode),
            "failed_stage": meta.get("failed_stage"),
        })
```

In `create_job` signature & body (lines 166-265):
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
                     timing_model: str = Form(""),
                     qc_model: str = Form(""),
                     deep_audio_qc: bool = Form(False),
                     audio_fallback: str = Form("[]"),
                     translate_fallback: str = Form("[]"),
                     mode: str = Form("full"),
                     source_job_id: str = Form(""),
                     file: UploadFile | None = File(None)) -> dict:
```
After `resolved_translate_model`, validate the new parameters:
```python
    resolved_timing_model = timing_model.strip() or models_cfg["defaults"]["timing"]
    if resolved_timing_model not in models_cfg["transcription"]:
        raise HTTPException(422, f"timing_model tidak dikenal: {resolved_timing_model!r}")

    resolved_qc_model = qc_model.strip() or models_cfg["defaults"]["qc"]
    if resolved_qc_model not in models_cfg["transcription"]:
        raise HTTPException(422, f"qc_model tidak dikenal: {resolved_qc_model!r}")

    try:
        raw_audio_fb = json.loads(audio_fallback or "[]")
        raw_tl_fb = json.loads(translate_fallback or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Format JSON fallback tidak valid: {exc}") from exc
    if not isinstance(raw_audio_fb, list) or not isinstance(raw_tl_fb, list):
        raise HTTPException(422, "audio_fallback dan translate_fallback harus berupa array JSON.")
```
After `cfg = providers.Config.load()`:
```python
    validated_audio_fb = _validate_audio_fallback(raw_audio_fb, models_cfg)
    validated_tl_fb = _validate_translate_fallback(raw_tl_fb, models_cfg, cfg)
```
Include in `params`:
```python
        "timing_model": resolved_timing_model,
        "qc_model": resolved_qc_model,
        "deep_audio_qc": deep_audio_qc,
        "audio_fallback": validated_audio_fb,
        "translate_fallback": validated_tl_fb,
```

In `job_status` (line 270):
```python
    return {"id": job.id, "status": job.status, "error": job.error,
            "failed_stage": job.failed_stage,
            "params": job.params, "events": job.events}
```

In `download` (lines 294-303):
```python
@app.get("/api/jobs/{job_id}/files/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if filename not in DOWNLOADABLE:
        raise HTTPException(404, "File tidak tersedia untuk diunduh.")
    path = (job.dir / "qc" / "report.json") if filename == "qc_report.json" else (job.dir / filename)
    if not path.exists():
        raise HTTPException(404, f"{filename} belum ada.")
    return FileResponse(path, filename=f"{job.id}_{filename}")
```

In `RetryRequest` & `retry` (lines 317-365):
```python
class RetryRequest(BaseModel):
    transcribe_model: str | None = None
    timing_model: str | None = None
    qc_model: str | None = None
    translate_model: str | None = None
    translator: str | None = None
    gemini_key_label: str | None = None
    deep_audio_qc: bool | None = None
    audio_fallback: list[dict] | None = None
    translate_fallback: list[dict] | None = None
```
Inside `retry()`, after `if req.transcribe_model:`:
```python
        if req.timing_model:
            if req.timing_model not in models_cfg["transcription"]:
                raise HTTPException(422, f"timing_model tidak dikenal: {req.timing_model!r}")
            job.params["timing_model"] = req.timing_model

        if req.qc_model:
            if req.qc_model not in models_cfg["transcription"]:
                raise HTTPException(422, f"qc_model tidak dikenal: {req.qc_model!r}")
            job.params["qc_model"] = req.qc_model

        if req.deep_audio_qc is not None:
            job.params["deep_audio_qc"] = req.deep_audio_qc

        if req.audio_fallback is not None:
            job.params["audio_fallback"] = _validate_audio_fallback(req.audio_fallback, models_cfg)

        if req.translate_fallback is not None:
            cfg = providers.Config.load()
            job.params["translate_fallback"] = _validate_translate_fallback(req.translate_fallback, models_cfg, cfg)
```
And reset `job.failed_stage = None` alongside `job.error = None` before `pipeline._save(job)`.

- [ ] **Step 4: Run API tests**

Run: `python -m pytest tests/test_api.py -v`
Expected: All tests in `tests/test_api.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat(api): validate fallback slots, timing/qc models, deep_audio_qc, and qc_report download"
```

---

### Task 12: Web UI — Fallbacks, Stage-Aware Retry & QC Report Card (`static/index.html`)

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add form UI elements for `timing_model`, `qc_model`, `deep_audio_qc`, and two 3-slot fallback builders**

In `static/index.html`, right after the `translate_model` / `output_format` `.field-row` (around line 782), insert:

```html
      <div class="field-row mode-full">
        <div class="field">
          <label class="field-label" for="timing_model">Model timing</label>
          <select id="timing_model"></select>
        </div>
        <div class="field">
          <label class="field-label" for="qc_model">Model audio QC</label>
          <select id="qc_model"></select>
        </div>
      </div>

      <div class="field mode-full">
        <label class="check-row" style="margin-top:0;">
          <input type="checkbox" id="deep_audio_qc">
          Deep Audio QC (cek semua batch dengan audio — lebih lambat &amp; boros token, default hanya baris mencurigakan)
        </label>
      </div>

      <details class="field mode-full mode-trans" style="border:1px solid var(--border);border-radius:var(--r-sm);padding:10px 14px;background:rgba(0,0,0,.2);">
        <summary style="cursor:pointer;font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted2);">
          Fallback Transkripsi / Timing / Audio QC (maks 3 slot)
        </summary>
        <div id="audio-fallback-rows" style="display:flex;flex-direction:column;gap:8px;margin-top:10px;"></div>
      </details>

      <details class="field mode-full mode-tl" style="border:1px solid var(--border);border-radius:var(--r-sm);padding:10px 14px;background:rgba(0,0,0,.2);">
        <summary style="cursor:pointer;font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--muted2);">
          Fallback Translasi &amp; Text QC (maks 3 slot)
        </summary>
        <div id="tl-fallback-rows" style="display:flex;flex-direction:column;gap:8px;margin-top:10px;"></div>
      </details>
```

- [ ] **Step 2: Add `#qc-report-card` and upgrade `#retry-box` HTML**

Inside `.progress-left`, right below `#dl-card` (line 848), insert:

```html
        <div class="glass dl-card hidden" id="qc-report-card">
          <div class="dl-title" style="display:flex;justify-content:space-between;align-items:center;">
            <span>Laporan QC</span>
            <span id="qc-mode-badge" class="stage-chip">HYBRID</span>
          </div>
          <div id="qc-summary-pills" style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;"></div>
          <div id="qc-diff-list" style="max-height:240px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;font-size:.78rem;"></div>
        </div>
```

Replace `#retry-box` (lines 859-873) with:

```html
        <div class="retry-row hidden" id="retry-box">
          <div id="retry-stage-banner" style="font-size:.78rem;color:var(--muted2);margin-bottom:10px;">
            Gagal di tahap: <strong id="retry-stage-label" style="color:var(--c-error);">-</strong>
          </div>
          <div style="margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <div id="retry-key-wrap" style="display: flex; align-items: center; gap: 6px;">
              <label for="retry-gemini-key" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">API Key:</label>
              <select id="retry-gemini-key" style="max-width: 170px; font-size: .8rem; padding: 6px 10px;"></select>
            </div>
            <div id="retry-transcribe-wrap" class="hidden" style="display: flex; align-items: center; gap: 6px;">
              <label for="retry-transcribe-model" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">Model Transkripsi:</label>
              <select id="retry-transcribe-model" style="max-width: 210px; font-size: .8rem; padding: 6px 10px;"></select>
            </div>
            <div id="retry-timing-wrap" class="hidden" style="display: flex; align-items: center; gap: 6px;">
              <label for="retry-timing-model" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">Model Timing:</label>
              <select id="retry-timing-model" style="max-width: 210px; font-size: .8rem; padding: 6px 10px;"></select>
            </div>
            <div id="retry-translator-wrap" class="hidden" style="display: flex; align-items: center; gap: 6px;">
              <label for="retry-translator" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">Provider:</label>
              <select id="retry-translator" style="max-width: 160px; font-size: .8rem; padding: 6px 10px;"></select>
            </div>
            <div id="retry-translate-wrap" style="display: flex; align-items: center; gap: 6px;">
              <label for="retry-translate-model" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">Model Terjemahan:</label>
              <select id="retry-translate-model" style="max-width: 200px; font-size: .8rem; padding: 6px 10px;"></select>
            </div>
            <div id="retry-qc-wrap" class="hidden" style="display: flex; align-items: center; gap: 8px;">
              <label for="retry-qc-model" style="font-size: .8rem; color: var(--muted2); font-weight: 600;">Model Audio QC:</label>
              <select id="retry-qc-model" style="max-width: 200px; font-size: .8rem; padding: 6px 10px;"></select>
              <label class="check-row" style="margin-top:0;font-size:.78rem;">
                <input type="checkbox" id="retry-deep-audio-qc"> Deep Audio QC
              </label>
            </div>
          </div>
          <button class="btn-secondary" id="retry-btn" type="button">
            &#x21a9; Coba Lagi (lanjut dari checkpoint)
          </button>
        </div>
```

- [ ] **Step 3: Wire JS logic in `static/index.html`**

Add state tracking and helper functions in `<script>` inside `static/index.html`:

```javascript
let lastFailedStage = null;
let lastQcReport = null;

function buildFallbackRows(cfg) {
  const audioWrap = $("audio-fallback-rows");
  audioWrap.innerHTML = "";
  const keys = cfg.gemini_keys || ["Default"];
  const audioModels = (cfg.models && cfg.models.transcription) || [];
  for (let i = 1; i <= 3; i++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:56px 1fr 1.5fr;gap:8px;align-items:center;";
    row.innerHTML = `<span style="font-size:.75rem;color:var(--muted2);">Slot ${i}</span>
      <select id="afb-key-${i}"><option value="">— Tidak aktif —</option></select>
      <select id="afb-model-${i}"><option value="">— Pilih model —</option></select>`;
    audioWrap.appendChild(row);
    const kSel = $(`afb-key-${i}`);
    const mSel = $(`afb-model-${i}`);
    keys.forEach(k => kSel.add(new Option(k, k)));
    audioModels.forEach(m => mSel.add(new Option(m, m)));
  }

  const tlWrap = $("tl-fallback-rows");
  tlWrap.innerHTML = "";
  const activeProvs = Object.entries(cfg.providers || {}).filter(([, ok]) => ok).map(([p]) => p);
  for (let i = 1; i <= 3; i++) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:56px 1fr 1fr 1.4fr;gap:8px;align-items:center;";
    row.innerHTML = `<span style="font-size:.75rem;color:var(--muted2);">Slot ${i}</span>
      <select id="tfb-prov-${i}"><option value="">— Tidak aktif —</option></select>
      <select id="tfb-key-${i}"></select>
      <select id="tfb-model-${i}"><option value="">— Pilih model —</option></select>`;
    tlWrap.appendChild(row);
    const pSel = $(`tfb-prov-${i}`);
    const kSel = $(`tfb-key-${i}`);
    const mSel = $(`tfb-model-${i}`);
    activeProvs.forEach(p => pSel.add(new Option(PROVIDER_LABELS[p] || p, p)));
    keys.forEach(k => kSel.add(new Option(k, k)));
    const syncTlRow = () => {
      const prov = pSel.value;
      kSel.classList.toggle("hidden", prov !== "gemini");
      mSel.innerHTML = '<option value="">— Pilih model —</option>';
      const list = (cfg.models && cfg.models.translation && cfg.models.translation[prov]) || [];
      list.forEach(m => mSel.add(new Option(m, m)));
    };
    pSel.addEventListener("change", syncTlRow);
    syncTlRow();
  }
}

function collectAudioFallbacks() {
  const slots = [];
  for (let i = 1; i <= 3; i++) {
    const key_label = $(`afb-key-${i}`)?.value;
    const model = $(`afb-model-${i}`)?.value;
    if (key_label && model) slots.push({ key_label, model });
  }
  return slots;
}

function collectTranslateFallbacks() {
  const slots = [];
  for (let i = 1; i <= 3; i++) {
    const provider = $(`tfb-prov-${i}`)?.value;
    const model = $(`tfb-model-${i}`)?.value;
    const key_label = $(`tfb-key-${i}`)?.value;
    if (provider && model) {
      slots.push(provider === "gemini" ? { provider, key_label, model } : { provider, model });
    }
  }
  return slots;
}

function renderQcReport(report) {
  if (!report || !report.summary) return;
  const s = report.summary;
  $("qc-mode-badge").textContent = (report.mode || "hybrid").toUpperCase();
  const pills = $("qc-summary-pills");
  pills.innerHTML = "";
  [
    ["Terjemahan", s.translation_fixes || 0],
    ["JP Diperbaiki", s.ja_fixes || 0],
    ["Timing", s.timing_fixes || 0],
    ["Split", s.splits || 0],
    ["Warning", s.unresolved_warnings || 0],
  ].forEach(([lbl, val]) => {
    const span = document.createElement("span");
    span.className = "stage-chip";
    span.textContent = `${lbl}: ${val}`;
    pills.appendChild(span);
  });
  const list = $("qc-diff-list");
  list.innerHTML = "";
  const issues = report.issues || [];
  if (issues.length === 0) {
    list.innerHTML = '<div style="color:var(--muted2);">Tidak ada perbaikan atau peringatan QC.</div>';
  } else {
    issues.slice(0, 60).forEach(item => {
      const div = document.createElement("div");
      div.style.cssText = "padding:8px 10px;border-radius:8px;background:rgba(255,255,255,.03);border:1px solid var(--border);";
      const header = document.createElement("div");
      header.style.cssText = "font-weight:600;color:var(--violet);margin-bottom:2px;";
      header.textContent = `#${item.id ?? "-"} · ${(item.type || "").toUpperCase()} ${item.reason ? "— " + item.reason : ""}`;
      div.appendChild(header);
      if (item.original || item.fixed) {
        const diff = document.createElement("div");
        diff.style.color = "var(--muted2)";
        diff.textContent = `${item.original || ""} → ${item.fixed || ""}`;
        div.appendChild(diff);
      }
      list.appendChild(div);
    });
  }
  $("qc-report-card").classList.remove("hidden");
}
```

Update `loadConfig()` to populate `#timing_model`, `#qc_model`, and call `buildFallbackRows(cfg)`:

```javascript
    const fillAudioSelect = (selId, defVal) => {
      const el = $(selId);
      if (!el) return;
      el.innerHTML = "";
      for (const m of (cfg.models.transcription || [])) {
        const opt = new Option(m, m);
        if (m === defVal) opt.selected = true;
        el.add(opt);
      }
    };
    fillAudioSelect("timing_model", cfg.models.defaults && cfg.models.defaults.timing);
    fillAudioSelect("qc_model", cfg.models.defaults && cfg.models.defaults.qc);
    buildFallbackRows(cfg);
```

Update `showRetryBox(stage)` to show stage-relevant selects:

```javascript
function showRetryBox(stage) {
  const retryBox = $("retry-box");
  const effectiveStage = stage || lastFailedStage || "translate";
  $("retry-stage-label").textContent = effectiveStage.toUpperCase();

  const showTrans = effectiveStage === "transcribe";
  const showTiming = effectiveStage === "timing";
  const showTl = effectiveStage === "translate" || !["transcribe", "timing", "qc"].includes(effectiveStage);
  const showQc = effectiveStage === "qc";

  $("retry-transcribe-wrap").classList.toggle("hidden", !showTrans);
  $("retry-timing-wrap").classList.toggle("hidden", !showTiming);
  $("retry-translator-wrap").classList.toggle("hidden", !showTl);
  $("retry-translate-wrap").classList.toggle("hidden", !showTl);
  $("retry-qc-wrap").classList.toggle("hidden", !showQc);

  if (globalConfig) {
    const rKeySel = $("retry-gemini-key");
    rKeySel.innerHTML = "";
    (globalConfig.gemini_keys || ["Default"]).forEach(k => {
      const opt = new Option(k, k);
      if (k === ($("gemini-key-sel").value || globalConfig.active_gemini_key)) opt.selected = true;
      rKeySel.add(opt);
    });

    const fillList = (id, list, currentVal) => {
      const el = $(id);
      el.innerHTML = "";
      (list || []).forEach(m => {
        const opt = new Option(m, m);
        if (m === currentVal) opt.selected = true;
        el.add(opt);
      });
    };

    const audioList = (globalConfig.models && globalConfig.models.transcription) || [];
    fillList("retry-transcribe-model", audioList, $("transcribe_model").value);
    fillList("retry-timing-model", audioList, $("timing_model").value);
    fillList("retry-qc-model", audioList, $("qc_model").value);
    $("retry-deep-audio-qc").checked = $("deep_audio_qc").checked;

    const rProvSel = $("retry-translator");
    rProvSel.innerHTML = "";
    Object.entries(globalConfig.providers || {}).forEach(([p, ok]) => {
      if (ok) rProvSel.add(new Option(PROVIDER_LABELS[p] || p, p));
    });
    rProvSel.value = $("translator").value || "gemini";
    const syncRetryTl = () => {
      const prov = rProvSel.value;
      const list = (globalConfig.models && globalConfig.models.translation && globalConfig.models.translation[prov]) || [];
      fillList("retry-translate-model", list, $("translate_model").value);
      $("retry-key-wrap").classList.toggle("hidden", (showTl && prov !== "gemini"));
    };
    rProvSel.onchange = syncRetryTl;
    syncRetryTl();
  }
  retryBox.classList.remove("hidden");
}
```

In `resetProgress()`, reset the QC & failed-stage UI state right after `hideRetryBox();`:

```javascript
  lastFailedStage = null;
  lastQcReport = null;
  $("qc-report-card").classList.add("hidden");
  $("qc-summary-pills").innerHTML = "";
  $("qc-diff-list").innerHTML = "";
```

In `listen(id)` (`es.onmessage`), capture `lastQcReport` and `lastFailedStage`:

```javascript
    if (ev.stage === "qc" && ev.status === "done" && ev.result && ev.result.report) {
      lastQcReport = ev.result.report;
    }
    if (ev.status === "error") {
      logLine(ev.stage, "error", "GAGAL: " + (ev.message || ""));
      hideCancelBtn();
      lastFailedStage = ev.failed_stage || ev.stage;
      showRetryBox(lastFailedStage);
      return;
    }
```

In `finishJob(result)`, render the QC report card right after `buildPreview(result.files);`:

```javascript
  if (lastQcReport) {
    renderQcReport(lastQcReport);
  } else if (result.files && result.files.includes("qc_report.json")) {
    fetch("/api/jobs/" + jobId + "/files/qc_report.json")
      .then(r => r.ok ? r.json() : null)
      .then(rep => { if (rep) renderQcReport(rep); })
      .catch(() => {});
  }
```

In `$("#job-form")` submit handler, append the new fields alongside `translate_model`:

```javascript
  fd.append("timing_model", $("timing_model").value);
  fd.append("qc_model", $("qc_model").value);
  fd.append("deep_audio_qc", $("deep_audio_qc").checked ? "true" : "false");
  fd.append("audio_fallback", JSON.stringify(collectAudioFallbacks()));
  fd.append("translate_fallback", JSON.stringify(collectTranslateFallbacks()));
```

Replace the `$("retry-btn")` click handler (lines 1400-1424) with:

```javascript
$("retry-btn").addEventListener("click", async () => {
  const payload = {};
  if (!$("retry-key-wrap").classList.contains("hidden") && $("retry-gemini-key").value) {
    payload.gemini_key_label = $("retry-gemini-key").value;
  }
  if (!$("retry-transcribe-wrap").classList.contains("hidden") && $("retry-transcribe-model").value) {
    payload.transcribe_model = $("retry-transcribe-model").value;
    $("transcribe_model").value = payload.transcribe_model;
  }
  if (!$("retry-timing-wrap").classList.contains("hidden") && $("retry-timing-model").value) {
    payload.timing_model = $("retry-timing-model").value;
    $("timing_model").value = payload.timing_model;
  }
  if (!$("retry-translate-wrap").classList.contains("hidden")) {
    if ($("retry-translator").value) payload.translator = $("retry-translator").value;
    if ($("retry-translate-model").value) {
      payload.translate_model = $("retry-translate-model").value;
      $("translate_model").value = payload.translate_model;
    }
  }
  if (!$("retry-qc-wrap").classList.contains("hidden")) {
    if ($("retry-qc-model").value) {
      payload.qc_model = $("retry-qc-model").value;
      $("qc_model").value = payload.qc_model;
    }
    payload.deep_audio_qc = $("retry-deep-audio-qc").checked;
    $("deep_audio_qc").checked = payload.deep_audio_qc;
  }

  const resp = await fetch("/api/jobs/" + jobId + "/retry", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    alert(err.detail || "Gagal retry.");
    return;
  }
  if (payload.gemini_key_label && $("gemini-key-sel")) {
    $("gemini-key-sel").value = payload.gemini_key_label;
  }
  hideRetryBox();
  resetProgress();
  logLine("job", "start", "Job " + jobId + " diulang dari checkpoint...");
  listen(jobId);
});
```

- [ ] **Step 4: Run full test suite & verify UI startup**

Run: `python -m pytest -q`
Expected: All existing + new tests PASS.
Also start local server briefly or test `GET /` and `GET /api/config` via TestClient to verify HTML renders cleanly with no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat(ui): add 3-slot fallback selectors, stage-aware retry box, and QC report panel"
```

---

## Plan Self-Review Checklist

- **Spec coverage check:**
  - 3-slot audio & translation fallbacks + `FallbackExecutor` + key isolation + upload cache -> Tasks 1, 2, 10, 11, 12
  - Stage-aware retry UI (`transcribe_model`, `timing_model`, `qc_model`, `translate_model`, `translator`, `gemini_key_label`, `failed_stage`) -> Tasks 10, 11, 12
  - `timing` & `qc` per-chunk/batch checkpoints + no silent error swallowing -> Tasks 6, 8, 9, 10
  - `transcribe` max-2-speaker `- ` rules -> Task 3
  - `translate` rolling episode glossary + backward compatibility -> Task 4
  - `subtitle` `_merge_short` 2-turn cap -> Task 5
  - `qc` Pass 1A local audit + Pass 1B text consistency + Pass 2 targeted/deep audio repair & splits + `translation_only` text QC -> Tasks 7, 8, 9, 10
  - `qc_report.json` download allowlist & UI QC Report card -> Tasks 9, 10, 11, 12
- **No placeholders check:** All code blocks, signatures, tests, and commands are complete and consistent across tasks (`AudioFallbackSlot`, `TranslateFallbackSlot`, `make_audio_client`, `make_text_client`, `audit_local`, `run_text_qc`, `run_qc_all`, `qc_report.json`, `failed_stage`).
