import json

import pytest

from app import pipeline, transcribe
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
                              depth=0, on_upload_done=None):
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
    monkeypatch.setattr("app.providers.make_transcriber", lambda cfg, model=None: object())
    monkeypatch.setattr("app.providers.make_translator",
                        lambda cfg, p, model=None: object())
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
