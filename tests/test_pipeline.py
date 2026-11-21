import json
from dataclasses import replace

import pytest

from app import pipeline, providers, transcribe, translate
from app.providers import LLMResponse
from app.transcribe import Utterance

_REAL_TRANSCRIBE_CHUNK = transcribe.transcribe_chunk
_REAL_TRANSLATE_BATCH = translate.translate_batch

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
                              depth=0, on_upload_done=None, audio_runner=None):
        counters["transcribe"] += 1
        return FAKE_UTTS

    def fake_translate_batch(client, utterances, system, tracker,
                             glossary=None, text_runner=None):
        counters["translate"] += 1
        return {u.id: f"ID-{u.id}" for u in utterances}

    monkeypatch.setattr("app.audio.download_youtube", fake_download)
    monkeypatch.setattr("app.audio.normalize_audio", fake_normalize)
    monkeypatch.setattr("app.audio.get_duration", lambda p: 200.0)
    monkeypatch.setattr("app.audio.detect_silences", lambda p, t: [])
    monkeypatch.setattr("app.audio.cut_chunk", fake_cut)
    monkeypatch.setattr("app.transcribe.transcribe_chunk", fake_transcribe_chunk)
    monkeypatch.setattr("app.translate.translate_batch", fake_translate_batch)
    monkeypatch.setattr("app.providers.make_transcriber", lambda cfg, model=None: object())
    monkeypatch.setattr("app.providers.make_translator",
                        lambda cfg, p, model=None: object())
    monkeypatch.setattr("app.timing.reanchor_all",
                        lambda utterances, chunks, chunks_dir, gemini, tracker, **kw: (utterances, []))
    monkeypatch.setattr("app.qc.run_qc_all",
                        lambda gemini, rows, audio_path, tmp_dir, system, tracker, **kw: (rows, []))
    return counters


def make_params(**overrides):
    params = {"source": "url", "url": "https://youtu.be/x", "save_mp4": False,
              "translator": "gemini", "output_format": "both",
              "original_filename": None,
              "transcribe_model": "gemini-3.1-pro-preview",
              "translate_model": "gemini-3.5-flash"}
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
    def boom(client, utterances, system, tracker, **kw):
        raise RuntimeError("translasi meledak")

    monkeypatch.setattr("app.translate.translate_batch", boom)
    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert "translasi meledak" in job.error
    assert job.events[-1]["status"] == "error"

    # retry with translation fixed
    monkeypatch.setattr("app.translate.translate_batch",
                        lambda c, u, s, t, **kw: {x.id: "OK" for x in u})
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


def test_additional_context_passed_to_prompt_builders(env, monkeypatch):
    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["additional_context"] = additional_context
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(additional_context="Fishing vlog"))
    pipeline.run_job(job.id)
    assert received.get("additional_context") == "Fishing vlog"


def test_context_override_used_in_prompt_builders(env, monkeypatch):
    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(context_override="OVERRIDE-CTX"))
    pipeline.run_job(job.id)
    assert received.get("context_md") == "OVERRIDE-CTX"


def test_context_preset_param_loads_correct_file(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_chokosaku.md").write_text("CHOKOSAKU-CTX", encoding="utf-8")
    (ctx_dir / "members_sakurazaka.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="sakurazaka_chokosaku"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == "CHOKOSAKU-CTX"


def test_missing_context_preset_fails_job(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="nonexistent"
    ))
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert "nonexistent" in (job.error or "")


def test_read_context_loads_group_prefixed_members(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_sokomagattara.md").write_text("SOKO-CTX", encoding="utf-8")
    (ctx_dir / "members_sakurazaka.md").write_text("SAKURA-MEMBERS", encoding="utf-8")
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        received["members_md"] = members
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="sakurazaka_sokomagattara"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == "SOKO-CTX"
    assert received.get("members_md") == "SAKURA-MEMBERS"


def test_read_context_else_returns_empty_strings(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        received["members_md"] = members
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="else", context_preset="",
        additional_context="Custom context for a random vlog"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == ""
    assert received.get("members_md") == ""


def test_pipeline_passes_transcribe_model_to_maker(env, tmp_path, monkeypatch):
    models_called = []

    def fake_maker(cfg, model=None):
        models_called.append(model)
        return object()

    monkeypatch.setattr("app.providers.make_transcriber", fake_maker)
    params = make_params(transcribe_model="gemini-2.5-flash")
    job = pipeline.create_job(params)
    pipeline.run_job(job.id)
    assert models_called[0] == "gemini-2.5-flash"


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
    pipeline.JOBS.clear()
    loaded = pipeline.load_job(job.id)
    assert loaded is not None
    assert loaded.failed_stage == "translate"


def test_cancel_keeps_failed_stage_none(env, monkeypatch):
    job = pipeline.create_job(make_params())
    job.failed_stage = "translate"

    def cancel_during_download(url, job_dir, *, save_mp4, cookies_file):
        pipeline.cancel_job(job.id)
        p = job_dir / "source.m4a"
        p.write_bytes(b"av-data")
        return p

    monkeypatch.setattr("app.audio.download_youtube", cancel_during_download)
    pipeline.run_job(job.id)
    assert job.status == "cancelled"
    assert job.failed_stage is None
    assert "failed_stage" not in job.events[-1]


def test_pipeline_runs_text_qc_in_translation_only_mode(env, monkeypatch):
    qc_calls = []

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        qc_calls.append({
            "gemini": gemini,
            "audio_path": audio_path,
            "has_text_runner": kw.get("text_runner") is not None,
            "has_audio_runner": kw.get("audio_runner") is not None,
        })
        return rows, {
            "mode": "text_only",
            "deep_audio_qc": False,
            "summary": {
                "total_rows": len(rows),
                "local_flags": 0,
                "audio_slices_checked": 0,
                "translation_fixes": 0,
                "ja_fixes": 0,
                "timing_fixes": 0,
                "splits": 0,
                "unresolved_warnings": 0,
            },
            "issues": [],
        }

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
    assert qc_calls[0]["gemini"] is None
    assert qc_calls[0]["has_text_runner"] is True
    assert qc_calls[0]["has_audio_runner"] is False
    assert (job.dir / "qc" / "report.json").exists()
    assert "qc_report.json" in job.events[-1]["result"]["files"]


def test_pipeline_wires_audio_and_translate_fallbacks_end_to_end(env, monkeypatch):
    class HttpErr(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    audio_calls = []
    text_calls = []

    class FakeAudioClient:
        def __init__(self, api_key: str, model: str):
            self.api_key = api_key
            self.model = model

        def upload_audio(self, path):
            return f"ref:{self.api_key}"

        def generate(self, system: str, user: str, audio=None):
            audio_calls.append((self.api_key, self.model))
            if self.model == "gemini-3.1-pro-preview":
                raise HttpErr(503)
            return LLMResponse(
                text=json.dumps([
                    {"id": 1, "start": "00:01.000", "end": "00:02.000",
                     "type": "dialogue", "ja": "こんにちは"}
                ]),
                input_tokens=15,
                output_tokens=7,
            )

    class FakeTextClient:
        def __init__(self, provider: str, model: str, key_label: str | None = None):
            self.provider = provider
            self.model = model
            self.key_label = key_label

        def generate(self, system: str, user: str, audio=None):
            text_calls.append((self.provider, self.key_label, self.model))
            if self.provider == "gemini" and self.model == "gemini-3.5-flash":
                raise HttpErr(503)
            return LLMResponse(
                text=json.dumps({
                    "translations": [{"id": 1, "id_text": "Halo fallback"}],
                    "new_terms": {},
                }),
                input_tokens=11,
                output_tokens=5,
            )

    monkeypatch.setenv("GEMINI_API_KEY", "k-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "k-alt")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    monkeypatch.setattr("app.transcribe.transcribe_chunk", _REAL_TRANSCRIBE_CHUNK)
    monkeypatch.setattr("app.translate.translate_batch", _REAL_TRANSLATE_BATCH)

    orig_exec_audio = providers.execute_audio_with_fallback
    orig_exec_text = providers.execute_with_fallback
    monkeypatch.setattr(
        "app.providers.execute_audio_with_fallback",
        lambda **kw: orig_exec_audio(**{**kw, "sleep": lambda _: None}),
    )
    monkeypatch.setattr(
        "app.providers.execute_with_fallback",
        lambda **kw: orig_exec_text(**{**kw, "sleep": lambda _: None}),
    )

    monkeypatch.setattr(
        "app.providers.make_transcriber",
        lambda cfg, model=None: FakeAudioClient("k-default", model or cfg.transcribe_model),
    )
    monkeypatch.setattr(
        "app.providers.make_audio_client",
        lambda cfg, provider="gemini", model=None, key_label=None: FakeAudioClient(
            providers.resolve_gemini_key(key_label, fallback_key=cfg.gemini_api_key),
            model or cfg.transcribe_model,
        ),
    )
    monkeypatch.setattr(
        "app.providers.make_translator",
        lambda cfg, p, model=None: FakeTextClient(p, model or cfg.translate_models[p]),
    )
    monkeypatch.setattr(
        "app.providers.make_text_client",
        lambda cfg, p, model=None, key_label=None: FakeTextClient(
            p, model or cfg.translate_models[p], key_label=key_label
        ),
    )

    job = pipeline.create_job(make_params(
        audio_fallback=[{"key_label": "ALT", "model": "gemini-2.5-flash"}],
        translate_fallback=[{"provider": "openai", "key_label": None, "model": "gpt-4o"}],
    ))
    pipeline.run_job(job.id)

    assert job.status == "done"
    assert audio_calls == [
        ("k-default", "gemini-3.1-pro-preview"),
        ("k-default", "gemini-3.1-pro-preview"),
        ("k-alt", "gemini-2.5-flash"),
    ]
    assert text_calls == [
        ("gemini", None, "gemini-3.5-flash"),
        ("gemini", None, "gemini-3.5-flash"),
        ("openai", None, "gpt-4o"),
    ]
    assert any(
        e["stage"] == "transcribe" and "Fallback 1" in e.get("message", "")
        for e in job.events
    )
    assert any(
        e["stage"] == "translate" and "Fallback 1" in e.get("message", "")
        for e in job.events
    )
    translated = json.loads((job.dir / "translated_id.json").read_text(encoding="utf-8"))
    assert translated[0]["id_text"] == "Halo fallback"
    usage = json.loads((job.dir / "usage.json").read_text(encoding="utf-8"))
    assert "gemini-2.5-flash" in usage["models"]
    assert "gpt-4o" in usage["models"]


def test_pipeline_saves_episode_glossary_and_passes_to_qc(env, monkeypatch):
    class FakeGlossTranslator:
        model = "gemini-3.5-flash"

        def generate(self, system: str, user: str, audio=None):
            return LLMResponse(
                text=json.dumps({
                    "translations": [
                        {"id": 1, "id_text": "Halo"},
                        {"id": 2, "id_text": "Segmen berikutnya"},
                    ],
                    "new_terms": {"次のコーナー": "Segmen berikutnya"},
                }),
                input_tokens=10,
                output_tokens=5,
            )

    qc_glossaries = []

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        qc_glossaries.append(kw.get("glossary"))
        return rows, {"mode": "hybrid", "deep_audio_qc": False, "summary": {"translation_fixes": 0}, "issues": []}

    monkeypatch.setattr("app.translate.translate_batch", _REAL_TRANSLATE_BATCH)
    monkeypatch.setattr("app.providers.make_translator", lambda cfg, p, model=None: FakeGlossTranslator())
    monkeypatch.setattr("app.qc.run_qc_all", spy_qc)

    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)

    assert job.status == "done"
    glossary_path = job.dir / "episode_glossary.json"
    assert glossary_path.exists()
    expected = {"次のコーナー": "Segmen berikutnya"}
    assert json.loads(glossary_path.read_text(encoding="utf-8")) == expected
    assert qc_glossaries == [expected]


def test_pipeline_passes_timing_and_qc_models_and_deep_flag(env, monkeypatch):
    audio_client_calls = []
    timing_calls = []
    qc_calls = []

    def spy_make_audio_client(cfg, provider="gemini", model=None, key_label=None):
        audio_client_calls.append({"model": model, "key_label": key_label})
        return object()

    def spy_timing(utterances, chunks, chunks_dir, gemini, tracker, **kw):
        timing_calls.append({
            "has_audio_runner": kw.get("audio_runner") is not None,
            "has_load_ckpt": kw.get("load_checkpoint") is not None,
            "has_save_ckpt": kw.get("save_checkpoint") is not None,
        })
        return utterances, [{"chunk": 1, "status": "corrected", "count": len(utterances)}]

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        qc_calls.append({
            "deep_audio_qc": kw.get("deep_audio_qc"),
            "cps_threshold": kw.get("cps_threshold"),
            "checkpoint_dir": kw.get("checkpoint_dir"),
            "has_text_runner": kw.get("text_runner") is not None,
            "has_audio_runner": kw.get("audio_runner") is not None,
            "timing_report": kw.get("timing_report"),
            "chunk_offsets_len": len(kw.get("chunk_offsets") or []),
        })
        return rows, {"mode": "deep", "deep_audio_qc": True, "summary": {"translation_fixes": 0}, "issues": []}

    monkeypatch.setattr("app.providers.make_audio_client", spy_make_audio_client)
    monkeypatch.setattr("app.timing.reanchor_all", spy_timing)
    monkeypatch.setattr("app.qc.run_qc_all", spy_qc)

    job = pipeline.create_job(make_params(
        timing_model="gemini-3-flash-preview",
        qc_model="gemini-3.1-pro-preview",
        deep_audio_qc=True,
    ))
    pipeline.run_job(job.id)

    assert job.status == "done"
    assert {"model": "gemini-3-flash-preview", "key_label": None} in audio_client_calls
    assert {"model": "gemini-3.1-pro-preview", "key_label": None} in audio_client_calls
    assert timing_calls == [{"has_audio_runner": True, "has_load_ckpt": True, "has_save_ckpt": True}]
    assert len(qc_calls) == 1
    assert qc_calls[0]["deep_audio_qc"] is True
    assert qc_calls[0]["checkpoint_dir"] == job.dir / "chunks"
    assert qc_calls[0]["has_text_runner"] is True
    assert qc_calls[0]["has_audio_runner"] is True
    assert qc_calls[0]["timing_report"] == [{"chunk": 1, "status": "corrected", "count": 2}]
    assert qc_calls[0]["chunk_offsets_len"] == 1


def test_pipeline_timing_resort_merge_and_corrupt_checkpoint_recovery(env, monkeypatch):
    def resort_timing(utterances, chunks, chunks_dir, gemini, tracker, **kw):
        # Simulate corrupt checkpoint gracefully returning None
        load_ckpt = kw["load_checkpoint"]
        save_ckpt = kw["save_checkpoint"]
        (chunks_dir / "timing_001.json").write_text("{corrupt", encoding="utf-8")
        assert load_ckpt(1) is None

        # Simulate reanchor swapping order of utterances and re-indexing IDs 1..2
        u1, u2 = utterances[0], utterances[1]
        resorted = [
            replace(u2, id=1, start=0.5, end=1.2),
            replace(u1, id=2, start=2.5, end=3.5),
        ]
        report = [{"chunk": 1, "status": "corrected", "count": 2}]
        save_ckpt(1, {"utterances": [u.to_dict() for u in resorted], "report_entry": report[0]})
        assert load_ckpt(1)["report_entry"]["status"] == "corrected"
        return resorted, report

    monkeypatch.setattr("app.timing.reanchor_all", resort_timing)

    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)

    assert job.status == "done"
    saved_timing = json.loads((job.dir / "timing" / "utterances.json").read_text(encoding="utf-8"))
    # First row must be u2 ("次のコーナーです") with its own id_text "ID-2" and new timestamps 0.5..1.2
    assert saved_timing[0]["ja"] == "次のコーナーです"
    assert saved_timing[0]["id_text"] == "ID-2"
    assert saved_timing[0]["start"] == 0.5
    # Second row must be u1 ("こんにちは") with its own id_text "ID-1" and new timestamps 2.5..3.5
    assert saved_timing[1]["ja"] == "こんにちは"
    assert saved_timing[1]["id_text"] == "ID-1"
    assert saved_timing[1]["start"] == 2.5


def test_pipeline_qc_checkpoint_emits_saved_report_and_ignores_corrupt_sidecars(env, monkeypatch):
    seen = {}

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        seen["glossary"] = kw.get("glossary")
        seen["timing_report"] = kw.get("timing_report")
        seen["chunk_offsets"] = kw.get("chunk_offsets")
        return rows, {"mode": "text_only", "deep_audio_qc": False, "summary": {"translation_fixes": 1}, "issues": []}

    monkeypatch.setattr("app.qc.run_qc_all", spy_qc)

    job = pipeline.create_job(make_params(mode="translation_only", source="file", url=None))
    (job.dir / "transcript_jp.json").write_text(
        json.dumps([{"id": 1, "start": 0.0, "end": 1.5, "type": "dialogue", "ja": "こんにちは"}]),
        encoding="utf-8",
    )
    (job.dir / "episode_glossary.json").write_text("{bad-json", encoding="utf-8")
    (job.dir / "timing").mkdir(exist_ok=True)
    (job.dir / "timing" / "report.json").write_text("{bad-json", encoding="utf-8")
    (job.dir / "offsets.json").write_text("{bad-json", encoding="utf-8")

    # Wait: _stage_translate overwrites episode_glossary.json; let's checkpoint translate too so corrupt file stays
    (job.dir / "translated_id.json").write_text(
        json.dumps([{"id": 1, "start": 0.0, "end": 1.5, "type": "dialogue", "ja": "こんにちは", "id_text": "Halo"}]),
        encoding="utf-8",
    )
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert seen == {"glossary": {}, "timing_report": [], "chunk_offsets": []}

    # Now re-run hitting qc checkpoint: should emit saved report
    job.events.clear()
    pipeline.run_job(job.id)
    qc_events = [e for e in job.events if e["stage"] == "qc"]
    assert qc_events[-1]["status"] == "done"
    assert qc_events[-1]["result"]["report"]["summary"]["translation_fixes"] == 1

    # Corrupt qc/report.json on checkpoint hit must not crash
    (job.dir / "qc" / "report.json").write_text("{corrupt", encoding="utf-8")
    job.events.clear()
    pipeline.run_job(job.id)
    assert job.status == "done"


def test_primary_client_object_not_reused_as_audio_or_text_client(env, monkeypatch):
    built_audio = []
    built_text = []

    class RealFakeAudio:
        model = "m-audio"

        def upload_audio(self, p):
            return "ref"

        def generate(self, system, user, audio=None):
            return LLMResponse(text="[]", input_tokens=1, output_tokens=1)

    class RealFakeText:
        model = "m-text"

        def generate(self, system, user, audio=None):
            return LLMResponse(text="[]", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(
        "app.providers.make_audio_client",
        lambda cfg, provider="gemini", model=None, key_label=None: (built_audio.append(model), RealFakeAudio())[1],
    )
    monkeypatch.setattr(
        "app.providers.make_text_client",
        lambda cfg, p, model=None, key_label=None: (built_text.append(model), RealFakeText())[1],
    )
    monkeypatch.setenv("GEMINI_API_KEY", "k-default")
    cfg = providers.Config.load()
    job = pipeline.create_job(make_params())

    chunk = job.dir / "chunks" / "chunk_001.mp3"
    chunk.write_bytes(b"mp3")

    a_runner = pipeline._make_audio_runner(job, cfg, "transcribe", "m-audio", primary_client=object())
    resp_a, model_a = a_runner(chunk, "sys", "usr")
    assert model_a == "m-audio"
    assert built_audio == ["m-audio"]

    t_runner = pipeline._make_text_runner(job, cfg, "translate", primary_client=object())
    resp_t, model_t = t_runner("sys", "usr")
    assert model_t == "m-text"
    assert built_text == ["gemini-3.5-flash"]


def test_stage_timing_duplicate_ja_preserves_order_when_later_cue_shifts_earlier(env, monkeypatch):
    dup_utts = [
        Utterance(id=1, start=1.0, end=2.0, type="dialogue", ja="いや"),
        Utterance(id=2, start=2.0, end=3.0, type="dialogue", ja="いや"),
        Utterance(id=3, start=3.0, end=4.0, type="dialogue", ja="ちょっと"),
    ]
    id_by_uid = {1: "Iya-1", 2: "Iya-2", 3: "Sebentar"}

    monkeypatch.setattr(
        "app.transcribe.transcribe_chunk",
        lambda *a, **kw: dup_utts,
    )
    monkeypatch.setattr(
        "app.translate.translate_batch",
        lambda client, utterances, system, tracker, **kw: {u.id: id_by_uid[u.id] for u in utterances},
    )

    def shifted_timing(utterances, chunks, chunks_dir, gemini, tracker, **kw):
        # Cue 3 ("ちょっと") shifts earlier than the two duplicate "いや" cues,
        # and reanchor_all renumbers IDs 1..3 in chronological order.
        u1, u2, u3 = utterances[0], utterances[1], utterances[2]
        corrected = [
            replace(u3, id=1, start=0.5, end=0.9),
            replace(u1, id=2, start=1.1, end=1.9),
            replace(u2, id=3, start=2.2, end=2.8),
        ]
        return corrected, [{"chunk": 1, "status": "corrected", "count": 3}]

    monkeypatch.setattr("app.timing.reanchor_all", shifted_timing)

    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    assert job.status == "done"

    saved_timing = json.loads((job.dir / "timing" / "utterances.json").read_text(encoding="utf-8"))
    assert [(r["ja"], r["id_text"], r["start"], r["end"]) for r in saved_timing] == [
        ("ちょっと", "Sebentar", 0.5, 0.9),
        ("いや", "Iya-1", 1.1, 1.9),
        ("いや", "Iya-2", 2.2, 2.8),
    ]

    # Also verify direct _stage_timing with unsorted input rows + (ja) fallback on type mismatch
    cfg = providers.Config.load()
    tracker = providers.UsageTracker(job.dir / "usage2.json")
    (job.dir / "timing" / "utterances.json").unlink()
    unsorted_rows = [
        {"id": 2, "start": 2.0, "end": 3.0, "type": "dialogue", "ja": "いや", "id_text": "Iya-2"},
        {"id": 1, "start": 1.0, "end": 2.0, "type": "narration", "ja": "いや", "id_text": "Iya-1"},
    ]
    monkeypatch.setattr(
        "app.timing.reanchor_all",
        lambda utts, *a, **kw: (
            [
                Utterance(id=1, start=1.05, end=1.95, type="dialogue", ja="いや"),
                Utterance(id=2, start=2.05, end=2.95, type="dialogue", ja="いや"),
            ],
            [{"chunk": 1, "status": "corrected", "count": 2}],
        ),
    )
    res = pipeline._stage_timing(
        job, cfg, tracker, unsorted_rows, chunks=[object()], audio_path=job.dir / "audio.mp3"
    )
    assert [(r["id_text"], r["start"], r["end"]) for r in res] == [
        ("Iya-1", 1.05, 1.95),
        ("Iya-2", 2.05, 2.95),
    ]


def test_stage_translate_partial_resume_propagates_checkpoint_glossary(env, monkeypatch):
    utterances = [
        Utterance(id=i, start=float(i), end=float(i) + 0.5, type="dialogue", ja=f"台詞{i}")
        for i in range(1, 102)
    ]
    seen_prompts = []

    class FakeBatch2Translator:
        model = "gemini-3.5-flash"

        def generate(self, system: str, user: str, audio=None):
            seen_prompts.append(user)
            return LLMResponse(
                text=json.dumps({
                    "translations": [{"id": 101, "id_text": "Manusia Serigala terakhir"}],
                    "new_terms": {"鬼": "Oni"},
                }, ensure_ascii=False),
                input_tokens=10,
                output_tokens=5,
            )

    monkeypatch.setattr("app.translate.translate_batch", _REAL_TRANSLATE_BATCH)
    monkeypatch.setattr("app.providers.make_translator", lambda cfg, p, model=None: FakeBatch2Translator())

    job = pipeline.create_job(make_params(mode="translation_only", source="file", url=None))
    (job.dir / "chunks").mkdir(exist_ok=True)
    (job.dir / "chunks" / "translated_001.json").write_text(
        json.dumps({
            "translations": {str(i): f"ID-{i}" for i in range(1, 101)},
            "glossary": {"人狼": "Manusia Serigala"},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    cfg = providers.Config.load()
    tracker = providers.UsageTracker(job.dir / "usage.json")
    rows = pipeline._stage_translate(job, cfg, tracker, utterances)

    assert len(rows) == 101
    assert rows[0]["id_text"] == "ID-1"
    assert rows[100]["id_text"] == "Manusia Serigala terakhir"
    assert len(seen_prompts) == 1
    assert "人狼 -> Manusia Serigala" in seen_prompts[0]
    assert json.loads((job.dir / "episode_glossary.json").read_text(encoding="utf-8")) == {
        "人狼": "Manusia Serigala",
        "鬼": "Oni",
    }
    assert (job.dir / "chunks" / "translated_002.json").exists()


def test_stage_translate_corrupt_final_checkpoint_reruns_cleanly(env):
    job = pipeline.create_job(make_params(mode="translation_only", source="file", url=None))
    (job.dir / "transcript_jp.json").write_text(
        json.dumps([u.to_dict() for u in FAKE_UTTS], ensure_ascii=False),
        encoding="utf-8",
    )
    (job.dir / "translated_id.json").write_text("{corrupt", encoding="utf-8")

    pipeline.run_job(job.id)
    assert job.status == "done"
    assert env["translate"] == 1
    saved = json.loads((job.dir / "translated_id.json").read_text(encoding="utf-8"))
    assert [r["id_text"] for r in saved] == ["ID-1", "ID-2"]

    # Non-list JSON is also treated as invalid checkpoint and rerun cleanly
    (job.dir / "translated_id.json").write_text('{"unexpected": true}', encoding="utf-8")
    for f in (job.dir / "chunks").glob("translated_*.json"):
        f.unlink()
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert env["translate"] == 2


def test_stale_downstream_invalidation(env, monkeypatch):
    translate_version = {"v": 1}
    timing_calls = {"n": 0}
    qc_calls = []

    monkeypatch.setattr(
        "app.translate.translate_batch",
        lambda client, utterances, system, tracker, **kw: {
            u.id: f"V{translate_version['v']}-{u.id}" for u in utterances
        },
    )

    def spy_timing(utterances, chunks, chunks_dir, gemini, tracker, **kw):
        timing_calls["n"] += 1
        job_dir = chunks_dir.parent
        # On re-translate, _stage_translate must have wiped timing & qc artifacts before timing ran
        if timing_calls["n"] == 2:
            assert not (job_dir / "timing" / "report.json").exists()
            assert not (chunks_dir / "timing_001.json").exists()
            assert not (job_dir / "qc" / "utterances.json").exists()
            assert not (job_dir / "qc" / "report.json").exists()
            assert not (chunks_dir / "qc_pass1.json").exists()
            assert not (chunks_dir / "qc_audio_000.json").exists()
            assert not (chunks_dir / "qc_input.sha256").exists()
        return utterances, [{"chunk": 1, "status": "corrected", "count": len(utterances)}]

    def spy_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        ckpt_dir = kw["checkpoint_dir"]
        qc_calls.append({
            "rows": [r["id_text"] for r in rows],
            "pass1_exists": (ckpt_dir / "qc_pass1.json").exists(),
            "audio0_exists": (ckpt_dir / "qc_audio_000.json").exists(),
        })
        return rows, {"mode": "hybrid", "deep_audio_qc": False, "summary": {"translation_fixes": 0}, "issues": []}

    monkeypatch.setattr("app.timing.reanchor_all", spy_timing)
    monkeypatch.setattr("app.qc.run_qc_all", spy_qc)

    job = pipeline.create_job(make_params())
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert timing_calls["n"] == 1
    assert len(qc_calls) == 1
    assert (job.dir / "chunks" / "qc_input.sha256").exists()

    # Seed per-chunk timing & qc checkpoints to confirm invalidation cleans them up
    (job.dir / "chunks" / "timing_001.json").write_text("{}", encoding="utf-8")
    (job.dir / "chunks" / "qc_pass1.json").write_text("{}", encoding="utf-8")
    (job.dir / "chunks" / "qc_audio_000.json").write_text("{}", encoding="utf-8")

    # 1) Re-running _stage_translate invalidates pre-existing timing/ and qc/ artifacts
    translate_version["v"] = 2
    (job.dir / "translated_id.json").unlink()
    for f in (job.dir / "chunks").glob("translated_*.json"):
        f.unlink()
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert timing_calls["n"] == 2
    assert len(qc_calls) == 2
    assert qc_calls[1]["rows"] == ["V2-1", "V2-2"]

    # 2) Unchanged translated_id.json with missing qc/utterances.json preserves qc_pass1.json
    (job.dir / "qc" / "utterances.json").unlink()
    (job.dir / "chunks" / "qc_pass1.json").write_text("{}", encoding="utf-8")
    (job.dir / "chunks" / "qc_audio_000.json").write_text("{}", encoding="utf-8")
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert qc_calls[2]["pass1_exists"] is True
    assert qc_calls[2]["audio0_exists"] is True

    # 3) Editing translated_id.json invalidates stale chunks/qc_pass1.json and qc_audio_*.json
    (job.dir / "qc" / "utterances.json").unlink()
    edited = json.loads((job.dir / "translated_id.json").read_text(encoding="utf-8"))
    edited[0]["id_text"] = "Edited-manually"
    (job.dir / "translated_id.json").write_text(
        json.dumps(edited, ensure_ascii=False), encoding="utf-8"
    )
    # Keep timing aligned with translated_id or wipe timing so edited rows flow straight to QC
    (job.dir / "timing" / "utterances.json").write_text(
        json.dumps(edited, ensure_ascii=False), encoding="utf-8"
    )
    old_sha = (job.dir / "chunks" / "qc_input.sha256").read_text(encoding="utf-8")
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert qc_calls[3]["rows"][0] == "Edited-manually"
    assert qc_calls[3]["pass1_exists"] is False
    assert qc_calls[3]["audio0_exists"] is False
    assert (job.dir / "chunks" / "qc_input.sha256").read_text(encoding="utf-8") != old_sha

    # 4) Re-running _stage_timing invalidates pre-existing qc/ artifacts
    (job.dir / "chunks" / "qc_pass1.json").write_text("{}", encoding="utf-8")
    (job.dir / "timing" / "utterances.json").unlink()
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert timing_calls["n"] == 3
    assert len(qc_calls) == 5
    assert qc_calls[4]["pass1_exists"] is False


def test_parse_fallbacks_handles_malformed_payloads():
    for bad in (123, True, "bad", {"key_label": "ALT"}, None):
        assert pipeline._parse_audio_fallback({"audio_fallback": bad}) == []
        assert pipeline._parse_translate_fallback({"translate_fallback": bad}) == []

    audio_slots = pipeline._parse_audio_fallback({
        "audio_fallback": [
            "not-a-dict",
            {"key_label": 123, "model": "gemini-2.5-flash"},
            {"key_label": "ALT", "model": True},
        ]
    })
    assert audio_slots == []

    audio_valid = pipeline._parse_audio_fallback({
        "audio_fallback": [
            {"key_label": " K1 ", "model": " m1 "},
            {"key_label": "K2", "model": "m2"},
            {"key_label": "K3", "model": "m3"},
            {"key_label": "K4", "model": "m4"},
        ]
    })
    assert audio_valid == [
        providers.AudioFallbackSlot(key_label="K1", model="m1"),
        providers.AudioFallbackSlot(key_label="K2", model="m2"),
        providers.AudioFallbackSlot(key_label="K3", model="m3"),
    ]

    trans_slots = pipeline._parse_translate_fallback({
        "translate_fallback": [
            None,
            {"provider": "openai", "model": "gpt-4o", "key_label": "ALT"},
            {"provider": "gemini", "model": "gemini-2.5-flash", "key_label": " ALT "},
            {"provider": "gemini", "model": "gemini-2.5-pro", "key_label": 999},
        ]
    })
    assert trans_slots == [
        providers.TranslateFallbackSlot(provider="openai", model="gpt-4o", key_label=None),
        providers.TranslateFallbackSlot(provider="gemini", model="gemini-2.5-flash", key_label="ALT"),
    ]

    trans_non_str = pipeline._parse_translate_fallback({
        "translate_fallback": [
            {"provider": True, "model": "gpt-4o", "key_label": None},
            {"provider": "gemini", "model": 123, "key_label": "ALT"},
            {"provider": "gemini", "model": "gemini-2.5-flash", "key_label": 123},
        ]
    })
    assert trans_non_str == [
        providers.TranslateFallbackSlot(provider="gemini", model="gemini-2.5-flash", key_label=None),
    ]


def test_timing_and_qc_runners_step_through_fallbacks_and_clear_failed_stage_on_retry(env, monkeypatch):
    class HttpErr(Exception):
        def __init__(self, status_code: int):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    audio_calls = []
    text_calls = []
    qc_should_fail_all = {"flag": True}

    class FakeAudioClient:
        def __init__(self, api_key: str, model: str):
            self.api_key = api_key
            self.model = model

        def upload_audio(self, path):
            return f"ref:{self.api_key}"

        def generate(self, system: str, user: str, audio=None):
            audio_calls.append((self.api_key, self.model))
            if self.model in ("timing-primary", "qc-primary"):
                raise HttpErr(503)
            if self.model == "gemini-2.5-flash" and qc_should_fail_all["flag"] and "QC" in user:
                raise HttpErr(503)
            return LLMResponse(text="[]", input_tokens=7, output_tokens=3)

    class FakeTextClient:
        def __init__(self, provider: str, model: str, key_label: str | None = None):
            self.provider = provider
            self.model = model
            self.key_label = key_label

        def generate(self, system: str, user: str, audio=None):
            text_calls.append((self.provider, self.key_label, self.model))
            if self.provider == "gemini":
                raise HttpErr(503)
            return LLMResponse(text="[]", input_tokens=9, output_tokens=4)

    monkeypatch.setenv("GEMINI_API_KEY", "k-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "k-alt")
    monkeypatch.setenv("OPENAI_API_KEY", "o-key")

    orig_exec_audio = providers.execute_audio_with_fallback
    orig_exec_text = providers.execute_with_fallback
    monkeypatch.setattr(
        "app.providers.execute_audio_with_fallback",
        lambda **kw: orig_exec_audio(**{**kw, "sleep": lambda _: None}),
    )
    monkeypatch.setattr(
        "app.providers.execute_with_fallback",
        lambda **kw: orig_exec_text(**{**kw, "sleep": lambda _: None}),
    )
    monkeypatch.setattr(
        "app.providers.make_audio_client",
        lambda cfg, provider="gemini", model=None, key_label=None: FakeAudioClient(
            providers.resolve_gemini_key(key_label, fallback_key=cfg.gemini_api_key),
            model or cfg.transcribe_model,
        ),
    )
    monkeypatch.setattr(
        "app.providers.make_text_client",
        lambda cfg, p, model=None, key_label=None: FakeTextClient(
            p, model or cfg.translate_models[p], key_label=key_label
        ),
    )

    def fake_timing(utterances, chunks, chunks_dir, gemini, tracker, **kw):
        resp, model_used = kw["audio_runner"](chunks_dir / "chunk_001.mp3", "", "TIMING")
        tracker.add(model_used, resp.input_tokens, resp.output_tokens)
        return utterances, [{"chunk": 1, "status": "corrected", "count": len(utterances)}]

    def fake_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        resp_t, model_t = kw["text_runner"](system, "QC_TEXT")
        tracker.add(model_t, resp_t.input_tokens, resp_t.output_tokens)
        resp_a, model_a = kw["audio_runner"](audio_path, system, "QC_AUDIO")
        tracker.add(model_a, resp_a.input_tokens, resp_a.output_tokens)
        return rows, {"mode": "hybrid", "deep_audio_qc": False, "summary": {"translation_fixes": 0}, "issues": []}

    monkeypatch.setattr("app.timing.reanchor_all", fake_timing)
    monkeypatch.setattr("app.qc.run_qc_all", fake_qc)

    job = pipeline.create_job(make_params(
        timing_model="timing-primary",
        qc_model="qc-primary",
        audio_fallback=[{"key_label": "ALT", "model": "gemini-2.5-flash"}],
        translate_fallback=[{"provider": "openai", "key_label": None, "model": "gpt-4o"}],
    ))
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert job.failed_stage == "qc"
    assert any(
        e["stage"] == "timing" and "Fallback 1" in e.get("message", "")
        for e in job.events
    )

    # Retry with QC fallback fixed -> succeeds and clears job.failed_stage to None
    qc_should_fail_all["flag"] = False
    pipeline.run_job(job.id)
    assert job.status == "done"
    assert job.failed_stage is None
    pipeline.JOBS.clear()
    reloaded = pipeline.load_job(job.id)
    assert reloaded is not None
    assert reloaded.failed_stage is None
    assert ("k-alt", "gemini-2.5-flash") in audio_calls
    assert ("openai", None, "gpt-4o") in text_calls
    assert any(
        e["stage"] == "qc" and "Fallback 1" in e.get("message", "")
        for e in job.events
    )


def test_pipeline_handles_none_translator_active_key_race_model_defaults_and_qc_report_parity(env, monkeypatch):
    # 1) translator=None defaults safely to gemini across _make_text_runner, _stage_translate, _stage_qc
    # 2) _stage_qc returning a list in text_only mode saves identical report to disk & SSE
    monkeypatch.setattr(
        "app.qc.run_qc_all",
        lambda *a, **kw: (a[1], [{"id": 1, "type": "translation", "reason": "r"}]),
    )
    job = pipeline.create_job(make_params(
        mode="translation_only", source="file", url=None, translator=None,
    ))
    (job.dir / "transcript_jp.json").write_text(
        json.dumps([FAKE_UTTS[0].to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    pipeline.run_job(job.id)
    assert job.status == "done"
    qc_done = [e for e in job.events if e["stage"] == "qc" and e["status"] == "done"][-1]
    disk_report = json.loads((job.dir / "qc" / "report.json").read_text(encoding="utf-8"))
    assert disk_report == qc_done["result"]["report"]
    assert disk_report["mode"] == "text_only"

    # 3) Active-key race: _resolve_key(primary_slot) uses cfg.gemini_api_key even if active label flips
    # 4) primary_model fallback when primary_model="" and winning client has no .model
    monkeypatch.setenv("GEMINI_API_KEY", "k-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "k-alt")
    providers.set_active_gemini_key("Default")
    cfg = providers.Config.load()
    cfg.gemini_api_key = "k-cfg-locked"
    providers.set_active_gemini_key("ALT")
    try:
        uploaded_refs = []

        class NoModelAudioClient:
            def upload_audio(self, p):
                uploaded_refs.append(p)
                return "ref-locked"

            def generate(self, system, user, audio=None):
                return LLMResponse(text="[]", input_tokens=1, output_tokens=1)

        class NoModelTextClient:
            def generate(self, system, user, audio=None):
                return LLMResponse(text="[]", input_tokens=1, output_tokens=1)

        seen_audio_keys = []
        orig_exec_audio = providers.execute_audio_with_fallback

        def spy_exec_audio(**kw):
            seen_audio_keys.append(kw["resolve_key"](kw["primary"]))
            return orig_exec_audio(**kw)

        monkeypatch.setattr("app.providers.execute_audio_with_fallback", spy_exec_audio)

        chunk = job.dir / "chunks" / "chunk_001.mp3"
        chunk.write_bytes(b"mp3")
        a_runner = pipeline._make_audio_runner(
            job, cfg, "timing", "", primary_client=NoModelAudioClient()
        )
        _, model_a = a_runner(chunk, "sys", "usr")
        assert seen_audio_keys == ["k-cfg-locked"]
        assert model_a == cfg.timing_model

        t_runner = pipeline._make_text_runner(
            job, cfg, "translate", primary_client=NoModelTextClient()
        )
        _, model_t = t_runner("sys", "usr")
        assert model_t == "gemini-3.5-flash"
    finally:
        providers.set_active_gemini_key("Default")


def test_pipeline_inherits_audio_provider_and_transcribe_model(env, monkeypatch):
    import inspect

    timing_calls = []
    qc_calls = []

    def fake_make_audio_client(cfg, provider="gemini", model=None, key_label=None):
        timing_calls.append({
            "caller": inspect.stack()[1].function,
            "provider": provider,
            "model": model,
        })
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

    def _inherited(call):
        return (
            call["provider"] == "openrouter"
            and call["model"] == "qwen/qwen3.8-omni-flash"
        )

    assert any(c["caller"] == "_stage_transcribe" and _inherited(c) for c in timing_calls)
    assert any(c["caller"] == "_stage_timing" and _inherited(c) for c in timing_calls)
    assert any(c["caller"] == "_stage_qc" and _inherited(c) for c in timing_calls)
    assert qc_calls, "audio QC pass should have run"


def test_translate_stage_uses_text_client_for_9router_and_openrouter(env, monkeypatch):
    calls = []

    def fake_make_text_client(cfg, provider, model=None, key_label=None):
        calls.append({"provider": provider, "model": model})
        return object()

    monkeypatch.setattr("app.providers.make_text_client", fake_make_text_client)

    for provider, model in (
        ("9router", "ag/gemini-3.8-flash-high"),
        ("openrouter", "deepseek/deepseek-chat"),
    ):
        calls.clear()
        job = pipeline.create_job(make_params(
            mode="translation_only", source="file", url=None,
            translator=provider, translate_model=model,
        ))
        (job.dir / "transcript_jp.json").write_text(
            json.dumps([FAKE_UTTS[0].to_dict()], ensure_ascii=False),
            encoding="utf-8",
        )
        pipeline.run_job(job.id)

        assert job.status == "done", (provider, job.error)
        assert any(
            c["provider"] == provider and c["model"] == model for c in calls
        ), provider


def test_pipeline_inherits_translator_and_translate_model_for_text_qc(env, monkeypatch):
    text_calls = []
    runner_calls = []

    class FakeTextClient:
        def __init__(self, provider: str, model: str):
            self.provider = provider
            self.model = model

        def generate(self, system: str, user: str, audio=None):
            return LLMResponse(text="[]", input_tokens=1, output_tokens=1)

    def fake_make_text_client(cfg, provider, model=None, key_label=None):
        text_calls.append({"provider": provider, "model": model})
        return FakeTextClient(provider, model)

    def fake_qc(gemini, rows, audio_path, tmp_dir, system, tracker, **kw):
        runner_calls.append(kw["text_runner"]("sys", "usr"))
        return rows, {"mode": "text_only", "summary": {}, "issues": []}

    monkeypatch.setattr("app.providers.make_text_client", fake_make_text_client)
    monkeypatch.setattr("app.qc.run_qc_all", fake_qc)

    job = pipeline.create_job(make_params(
        mode="translation_only", source="file", url=None,
        translator="openrouter",
        translate_model="deepseek/deepseek-chat",
    ))
    (job.dir / "transcript_jp.json").write_text(
        json.dumps([FAKE_UTTS[0].to_dict()], ensure_ascii=False),
        encoding="utf-8",
    )
    pipeline.run_job(job.id)

    assert job.status == "done"
    assert any(
        c["provider"] == "openrouter" and c["model"] == "deepseek/deepseek-chat"
        for c in text_calls
    )
    assert runner_calls and runner_calls[0][1] == "deepseek/deepseek-chat"


def test_pipeline_whisper_transcription_does_not_leak_into_timing_or_qc(env, monkeypatch):
    job = pipeline.create_job(make_params(
        transcribe_model="openai/whisper-large-v3-turbo",
        timing_model="",
        qc_model="",
    ))
    cfg = providers.Config.load()
    runner_timing = pipeline._make_audio_runner(job, cfg, "timing", "")
    assert runner_timing is not None

    # Verify timing model resolution in pipeline
    trans_m = job.params.get("transcribe_model")
    if trans_m and providers._is_transcription_model(trans_m):
        trans_m = None
    resolved_timing = (
        job.params.get("timing_model")
        or trans_m
        or cfg.timing_model
    )
    assert resolved_timing == cfg.timing_model
    assert "whisper" not in resolved_timing.lower()


def test_pipeline_openrouter_audio_provider_maps_gemini_to_google_prefix(env, monkeypatch):
    from pathlib import Path
    from app import audio
    calls = []

    def fake_make_audio_client(cfg, provider="gemini", model=None, key_label=None):
        calls.append({"provider": provider, "model": model})
        return object()

    monkeypatch.setattr("app.providers.make_audio_client", fake_make_audio_client)
    monkeypatch.setattr("app.timing.reanchor_all", lambda *a, **kw: ([], []))

    job = pipeline.create_job(make_params(
        audio_provider="openrouter",
        transcribe_model="openai/whisper-large-v3-turbo",
        timing_model="",
        qc_model="",
    ))
    cfg = providers.Config.load()
    tracker = providers.UsageTracker(job.dir / "usage.json")
    rows = [{"id": 1, "start": 0.0, "end": 1.0, "ja": "test", "type": "dialogue"}]
    chunks = [audio.PlannedChunk(index=1, start=0.0, end=1.0, overlap_prev=False)]
    pipeline._stage_timing(job, cfg, tracker, rows, chunks, audio_path=Path("dummy.mp3"))

    assert any(
        c["provider"] == "openrouter" and c["model"].startswith("google/")
        for c in calls
    )


def test_pipeline_timing_provider_gemini_with_openrouter_audio(env, monkeypatch):
    from pathlib import Path
    from app import audio
    calls = []

    def fake_make_audio_client(cfg, provider="gemini", model=None, key_label=None):
        calls.append({"provider": provider, "model": model})
        return object()

    monkeypatch.setattr("app.providers.make_audio_client", fake_make_audio_client)
    monkeypatch.setattr("app.timing.reanchor_all", lambda *a, **kw: ([], []))

    job = pipeline.create_job(make_params(
        audio_provider="openrouter",
        timing_provider="gemini",
        transcribe_model="openai/whisper-large-v3-turbo",
        timing_model="gemini-2.5-flash",
    ))
    cfg = providers.Config.load()
    tracker = providers.UsageTracker(job.dir / "usage.json")
    rows = [{"id": 1, "start": 0.0, "end": 1.0, "ja": "test", "type": "dialogue"}]
    chunks = [audio.PlannedChunk(index=1, start=0.0, end=1.0, overlap_prev=False)]
    pipeline._stage_timing(job, cfg, tracker, rows, chunks, audio_path=Path("dummy.mp3"))

    assert any(
        c["provider"] == "gemini" and c["model"] == "gemini-2.5-flash"
        for c in calls
    )


