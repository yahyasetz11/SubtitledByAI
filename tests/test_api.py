import json
import pytest
from fastapi.testclient import TestClient

from app import main, pipeline, providers
from app.providers import load_models_config


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


def test_retry_with_model_override(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={"translate_model": "gemini-3.7-flash"}
    )
    assert resp.status_code == 200
    job = pipeline.JOBS[job_id]
    assert job.params["translate_model"] == "gemini-3.7-flash"
    # verify persisted to job.json on disk
    disk_data = json.loads((job.dir / "job.json").read_text(encoding="utf-8"))
    assert disk_data["params"]["translate_model"] == "gemini-3.7-flash"


def test_retry_reloads_disk_changes(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    # Simulate user modifying job.json on disk directly
    job_json_path = job.dir / "job.json"
    data = json.loads(job_json_path.read_text(encoding="utf-8"))
    data["params"]["translate_model"] = "gemini-3.7-flash"
    job_json_path.write_text(json.dumps(data), encoding="utf-8")

    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert pipeline.JOBS[job_id].params["translate_model"] == "gemini-3.7-flash"


def test_retry_invalid_model_returns_422(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={"translate_model": "invalid-model-name"}
    )
    assert resp.status_code == 422


def test_retry_with_gemini_key_override(client, fake_run, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_SECONDARY", "sec-key-123")
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={"gemini_key_label": "SECONDARY"}
    )
    assert resp.status_code == 200
    assert providers.get_active_gemini_label() == "SECONDARY"


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_get_context_with_preset_param(client, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_sokomagattara.md").write_text("# Soko Context", encoding="utf-8")
    monkeypatch.setattr(main, "CONTEXT_DIR", ctx_dir)
    resp = client.get("/api/context?preset=sakurazaka_sokomagattara")
    assert resp.status_code == 200
    assert resp.json()["context_md"] == "# Soko Context"


def test_get_context_unknown_preset_returns_404(client, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    monkeypatch.setattr(main, "CONTEXT_DIR", ctx_dir)
    resp = client.get("/api/context?preset=nonexistent")
    assert resp.status_code == 404


def test_get_contexts_returns_groups_structure(client):
    resp = client.get("/api/contexts")
    assert resp.status_code == 200
    body = resp.json()
    assert "groups" in body
    group_ids = {g["id"] for g in body["groups"]}
    assert group_ids == {"sakurazaka", "nogizaka", "hinatazaka", "else"}


def test_get_contexts_each_group_has_shows(client):
    body = client.get("/api/contexts").json()
    groups = {g["id"]: g for g in body["groups"]}
    assert len(groups["sakurazaka"]["shows"]) >= 3
    assert len(groups["nogizaka"]["shows"]) == 3
    assert len(groups["hinatazaka"]["shows"]) == 3
    assert groups["else"]["shows"] == []


def test_get_contexts_show_ids_are_group_prefixed(client):
    body = client.get("/api/contexts").json()
    groups = {g["id"]: g for g in body["groups"]}
    saka_ids = {s["id"] for s in groups["sakurazaka"]["shows"]}
    assert "sakurazaka_sokomagattara" in saka_ids
    assert "sakurazaka_chokosaku" in saka_ids
    assert "sakurazaka_channel" in saka_ids
    nogi_ids = {s["id"] for s in groups["nogizaka"]["shows"]}
    assert "nogizaka_kojichuu" in nogi_ids
    hina_ids = {s["id"] for s in groups["hinatazaka"]["shows"]}
    assert "hinatazaka_aimashou" in hina_ids


def test_create_job_stores_group_param(client, fake_run):
    resp = post_url_job(client, group="nogizaka",
                        context_preset="nogizaka_kojichuu")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params.get("group") == "nogizaka"
    assert job.params.get("context_preset") == "nogizaka_kojichuu"


def test_create_job_group_defaults_to_sakurazaka(client, fake_run):
    resp = post_url_job(client)
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params.get("group") == "sakurazaka"


def test_create_job_stores_additional_context_and_override(client, fake_run):
    resp = post_url_job(client, additional_context="Fishing vlog",
                        context_override="Custom context override")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params.get("additional_context") == "Fishing vlog"
    assert job.params.get("context_override") == "Custom context override"


# ---------------------------------------------------------------------------
# Multi-key / active-key feature
# ---------------------------------------------------------------------------

def test_config_includes_gemini_keys_and_active_key(client, monkeypatch):
    monkeypatch.setattr("app.audio.ensure_ffmpeg", lambda: True)
    monkeypatch.setenv("GEMINI_API_KEY_PERSONAL", "g-personal")
    body = client.get("/api/config").json()
    assert "Default" in body["gemini_keys"]
    assert "PERSONAL" in body["gemini_keys"]
    assert body["active_gemini_key"] == "Default"


def test_put_active_key_switches_and_reflects_in_config(client, monkeypatch):
    monkeypatch.setattr("app.audio.ensure_ffmpeg", lambda: True)
    monkeypatch.setenv("GEMINI_API_KEY_WORK", "g-work")
    resp = client.put("/api/config/active-key", json={"label": "WORK"})
    assert resp.status_code == 200
    body = client.get("/api/config").json()
    assert body["active_gemini_key"] == "WORK"


def test_put_active_key_invalid_label_returns_422(client, monkeypatch):
    resp = client.put("/api/config/active-key", json={"label": "DOESNOTEXIST"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Continue last session — GET /api/sessions
# ---------------------------------------------------------------------------

import json as _json


def _write_job(tmp_path, job_id, status, original_filename=None, url=None):
    job_dir = tmp_path / job_id
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    params = {"source": "file" if original_filename else "url",
              "url": url, "translator": "gemini", "output_format": "ass",
              "save_mp4": False, "original_filename": original_filename,
              "group": "sakurazaka", "context_preset": "sakurazaka_sokomagattara",
              "context_override": None, "additional_context": None}
    meta = {"id": job_id, "params": params, "status": status,
            "error": None, "events": []}
    (job_dir / "job.json").write_text(
        _json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def test_sessions_empty_when_no_output_dirs(client):
    body = client.get("/api/sessions").json()
    assert body["sessions"] == []


def test_sessions_excludes_done_jobs(client, tmp_path):
    _write_job(tmp_path, "20260618-100000-aaaaaa", "done", original_filename="ep1.mp4")
    body = client.get("/api/sessions").json()
    assert body["sessions"] == []


def test_sessions_returns_failed_and_running(client, tmp_path):
    _write_job(tmp_path, "20260618-100000-aaaaaa", "failed", original_filename="ep1.mp4")
    _write_job(tmp_path, "20260618-110000-bbbbbb", "running", original_filename="ep2.mp4")
    _write_job(tmp_path, "20260618-120000-cccccc", "done",   original_filename="ep3.mp4")
    sessions = client.get("/api/sessions").json()["sessions"]
    ids = [s["id"] for s in sessions]
    assert "20260618-100000-aaaaaa" in ids
    assert "20260618-110000-bbbbbb" in ids
    assert "20260618-120000-cccccc" not in ids


def test_sessions_sorted_newest_first(client, tmp_path):
    _write_job(tmp_path, "20260618-080000-aaaaaa", "failed", original_filename="old.mp4")
    _write_job(tmp_path, "20260618-120000-bbbbbb", "failed", original_filename="new.mp4")
    sessions = client.get("/api/sessions").json()["sessions"]
    assert sessions[0]["id"] == "20260618-120000-bbbbbb"
    assert sessions[1]["id"] == "20260618-080000-aaaaaa"


def test_sessions_label_uses_original_filename(client, tmp_path):
    _write_job(tmp_path, "20260618-121111-aaaaaa", "failed", original_filename="rabbit-island.mp3")
    session = client.get("/api/sessions").json()["sessions"][0]
    assert "2026-06-18 12:11" in session["label"]
    assert "rabbit-island.mp3" in session["label"]
    assert "failed" in session["label"]


def test_sessions_label_uses_url_when_no_filename(client, tmp_path):
    _write_job(tmp_path, "20260618-121111-aaaaaa", "failed",
               url="https://youtu.be/abc123")
    session = client.get("/api/sessions").json()["sessions"][0]
    assert "youtu.be" in session["label"]


def test_sessions_truncates_long_filename(client, tmp_path):
    long_name = "A" * 80 + ".mp4"
    _write_job(tmp_path, "20260618-121111-aaaaaa", "failed", original_filename=long_name)
    session = client.get("/api/sessions").json()["sessions"][0]
    assert len(session["label"]) < 120  # label stays reasonable length
    assert "..." in session["label"]


# ---------------------------------------------------------------------------
# Model selection — GET /api/config and POST /api/jobs
# ---------------------------------------------------------------------------

def test_config_includes_models_list(client, monkeypatch):
    monkeypatch.setattr("app.audio.ensure_ffmpeg", lambda: True)
    models_cfg = load_models_config()
    body = client.get("/api/config").json()
    assert "models" in body
    models = body["models"]
    assert "transcription" in models
    assert "translation" in models
    assert "defaults" in models
    assert models_cfg["defaults"]["transcription"] in models["transcription"]


def test_create_job_accepts_transcribe_model(client, fake_run):
    models_cfg = load_models_config()
    model = models_cfg["transcription"][0]
    resp = post_url_job(client, transcribe_model=model)
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["transcribe_model"] == model


def test_create_job_accepts_translate_model(client, fake_run):
    models_cfg = load_models_config()
    model = models_cfg["translation"]["gemini"][0]
    resp = post_url_job(client, translate_model=model)
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["translate_model"] == model


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


def test_create_job_empty_translate_model_uses_default(client, fake_run):
    models_cfg = load_models_config()
    resp = post_url_job(client, translator="gemini", translate_model="")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["translate_model"] == models_cfg["defaults"]["translation"]["gemini"]


# ---------------------------------------------------------------------------
# QC report download, timing/QC models, fallback validation, failed_stage, retry
# ---------------------------------------------------------------------------


def test_download_qc_report_maps_to_qc_subdir_and_404_when_absent(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    assert client.get(f"/api/jobs/{job_id}/files/qc_report.json").status_code == 404

    qc_dir = job.dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {"mode": "hybrid", "summary": {"translation_fixes": 2}, "issues": []}
    (qc_dir / "report.json").write_text(json.dumps(report_payload), encoding="utf-8")

    resp = client.get(f"/api/jobs/{job_id}/files/qc_report.json")
    assert resp.status_code == 200
    assert resp.json() == report_payload


def test_create_job_validates_and_stores_qc_timing_and_fallbacks(client, fake_run, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "alt-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    models_cfg = load_models_config()
    timing_model = models_cfg["transcription"][0]
    qc_model = models_cfg["transcription"][-1]
    audio_fb = [
        {},
        {"key_label": "ALT", "model": "gemini-2.5-flash"},
    ]
    translate_fb = [
        {"provider": "", "model": ""},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "openai", "model": "gpt-4o", "key_label": "ALT"},
    ]
    resp = post_url_job(
        client,
        timing_model=timing_model,
        qc_model=qc_model,
        deep_audio_qc="true",
        audio_fallback=json.dumps(audio_fb),
        translate_fallback=json.dumps(translate_fb),
    )
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["timing_model"] == timing_model
    assert job.params["qc_model"] == qc_model
    assert job.params["deep_audio_qc"] is True
    assert job.params["audio_fallback"] == [
        {"key_label": "ALT", "model": "gemini-2.5-flash"},
    ]
    assert job.params["translate_fallback"] == [
        {"provider": "gemini", "model": "gemini-2.5-flash", "key_label": "Default"},
        {"provider": "openai", "model": "gpt-4o", "key_label": None},
    ]


def test_create_job_defaults_timing_qc_deep_and_fallbacks(client, fake_run):
    models_cfg = load_models_config()
    resp = post_url_job(client)
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params["timing_model"] == models_cfg["defaults"]["timing"]
    assert job.params["qc_model"] == models_cfg["defaults"]["qc"]
    assert job.params["deep_audio_qc"] is False
    assert job.params["audio_fallback"] == []
    assert job.params["translate_fallback"] == []


@pytest.mark.parametrize("field,value", [
    ("timing_model", "nonexistent-timing-model"),
    ("qc_model", "nonexistent-qc-model"),
    ("audio_fallback", "{not-json"),
    ("translate_fallback", "{not-json"),
    ("audio_fallback", '{"key_label": "Default", "model": "gemini-2.5-flash"}'),
    ("translate_fallback", '{"provider": "gemini", "model": "gemini-2.5-flash"}'),
    ("audio_fallback", json.dumps([
        {"key_label": "Default", "model": "gemini-2.5-flash"},
        {"key_label": "Default", "model": "gemini-2.5-flash"},
        {"key_label": "Default", "model": "gemini-2.5-flash"},
        {"key_label": "Default", "model": "gemini-2.5-flash"},
    ])),
    ("translate_fallback", json.dumps([
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
        {"provider": "gemini", "model": "gemini-2.5-flash"},
    ])),
    ("audio_fallback", json.dumps([{"key_label": "Default", "model": ""}])),
    ("audio_fallback", json.dumps([{"key_label": "", "model": "gemini-2.5-flash"}])),
    ("audio_fallback", json.dumps([{"key_label": "NO_SUCH_KEY", "model": "gemini-2.5-flash"}])),
    ("audio_fallback", json.dumps([{"key_label": "Default", "model": "bogus-model"}])),
    ("translate_fallback", json.dumps([{"provider": "gemini", "model": ""}])),
    ("translate_fallback", json.dumps([{"provider": "", "model": "gemini-2.5-flash"}])),
    ("translate_fallback", json.dumps([{"provider": "gemini", "model": "bogus-model"}])),
    ("translate_fallback", json.dumps([{"provider": "gemini", "model": "gemini-2.5-flash", "key_label": "NO_SUCH_KEY"}])),
    ("translate_fallback", json.dumps([{"provider": "unknown-prov", "model": "gpt-4o"}])),
])
def test_create_job_invalid_qc_timing_or_fallback_returns_422(client, field, value):
    resp = post_url_job(client, **{field: value})
    assert resp.status_code == 422


def test_create_job_unconfigured_provider_in_translate_fallback_returns_422(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fb = [{"provider": "openai", "model": "gpt-4o"}]
    resp = post_url_job(client, translate_fallback=json.dumps(fb))
    assert resp.status_code == 422


def test_job_status_and_sessions_include_failed_stage(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    assert client.get(f"/api/jobs/{job_id}").json()["failed_stage"] is None

    job.status = "failed"
    job.failed_stage = "qc"
    pipeline._save(job)

    status_body = client.get(f"/api/jobs/{job_id}").json()
    assert status_body["failed_stage"] == "qc"

    sessions = client.get("/api/sessions").json()["sessions"]
    match = [s for s in sessions if s["id"] == job_id]
    assert len(match) == 1
    assert match[0]["failed_stage"] == "qc"


def test_retry_accepts_and_validates_all_overrides_and_clears_failed_stage(client, fake_run, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "alt-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    job.status = "failed"
    job.error = "qc failed"
    job.failed_stage = "qc"
    pipeline._save(job)

    resp = client.post(
        f"/api/jobs/{job_id}/retry",
        json={
            "transcribe_model": "gemini-2.5-flash",
            "timing_model": "gemini-2.5-pro",
            "qc_model": "gemini-3.7-flash",
            "deep_audio_qc": True,
            "audio_fallback": [{"key_label": "ALT", "model": "gemini-2.5-flash"}],
            "translate_fallback": [{"provider": "openai", "model": "gpt-4o"}],
        },
    )
    assert resp.status_code == 200
    reloaded = pipeline.JOBS[job_id]
    assert reloaded.failed_stage is None
    assert reloaded.params["transcribe_model"] == "gemini-2.5-flash"
    assert reloaded.params["timing_model"] == "gemini-2.5-pro"
    assert reloaded.params["qc_model"] == "gemini-3.7-flash"
    assert reloaded.params["deep_audio_qc"] is True
    assert reloaded.params["audio_fallback"] == [{"key_label": "ALT", "model": "gemini-2.5-flash"}]
    assert reloaded.params["translate_fallback"] == [
        {"provider": "openai", "model": "gpt-4o", "key_label": None}
    ]
    disk = json.loads((reloaded.dir / "job.json").read_text(encoding="utf-8"))
    assert disk["failed_stage"] is None

    resp2 = client.post(
        f"/api/jobs/{job_id}/retry",
        json={"deep_audio_qc": False, "audio_fallback": [], "translate_fallback": []},
    )
    assert resp2.status_code == 200
    assert pipeline.JOBS[job_id].params["deep_audio_qc"] is False
    assert pipeline.JOBS[job_id].params["audio_fallback"] == []
    assert pipeline.JOBS[job_id].params["translate_fallback"] == []


@pytest.mark.parametrize("payload", [
    {"transcribe_model": "bad-model"},
    {"timing_model": "bad-model"},
    {"qc_model": "bad-model"},
    {"translator": "unknown-provider"},
    {"audio_fallback": [{"key_label": "NOPE", "model": "gemini-2.5-flash"}]},
    {"audio_fallback": [{"key_label": "Default", "model": ""}]},
    {"translate_fallback": [{"provider": "gemini", "model": "bad-model"}]},
])
def test_retry_rejects_invalid_overrides_with_422(client, fake_run, payload):
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(f"/api/jobs/{job_id}/retry", json=payload)
    assert resp.status_code == 422


def test_retry_switches_translator_and_resets_default_translate_model(client, fake_run, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    models_cfg = load_models_config()
    job_id = post_url_job(client, translator="gemini", translate_model="gemini-3.7-flash").json()["job_id"]

    resp = client.post(f"/api/jobs/{job_id}/retry", json={"translator": "openai"})
    assert resp.status_code == 200
    job = pipeline.JOBS[job_id]
    assert job.params["translator"] == "openai"
    assert job.params["translate_model"] == models_cfg["defaults"]["translation"]["openai"]


def test_retry_rejects_unconfigured_translator_422(client, fake_run, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    job_id = post_url_job(client).json()["job_id"]
    resp = client.post(f"/api/jobs/{job_id}/retry", json={"translator": "openai"})
    assert resp.status_code == 422


def test_retry_returns_409_when_live_job_is_running_even_if_disk_says_failed(client, fake_run):
    job_id = post_url_job(client).json()["job_id"]
    job = pipeline.JOBS[job_id]
    job.status = "failed"
    pipeline._save(job)
    job.status = "running"

    resp = client.post(f"/api/jobs/{job_id}/retry")
    assert resp.status_code == 409
    assert pipeline.JOBS[job_id].status == "running"
    assert fake_run == [job_id]
