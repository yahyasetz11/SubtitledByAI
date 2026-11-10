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
        lambda cfg, model=None, key_label=None: FakeAudioClient(
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

    def spy_make_audio_client(cfg, model=None, key_label=None):
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
        lambda cfg, model=None, key_label=None: (built_audio.append(model), RealFakeAudio())[1],
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

