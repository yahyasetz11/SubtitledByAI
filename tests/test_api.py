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


def test_get_context_returns_context_md(client, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context.md").write_text("# Acara Context", encoding="utf-8")
    monkeypatch.setattr(main, "CONTEXT_DIR", ctx_dir)
    resp = client.get("/api/context")
    assert resp.status_code == 200
    assert resp.json()["context_md"] == "# Acara Context"


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
