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
