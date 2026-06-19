"""FastAPI app: UI, job API, SSE progress, artifact downloads (§1, §3)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app import audio, pipeline, providers

app = FastAPI(title="Sokomagattara SubGen")

STATIC_DIR = Path("static")
CONTEXT_DIR = Path("context")
# Allowlist: never serve job.json (contains params) or arbitrary paths.
DOWNLOADABLE = {"result.ass", "result.srt", "transcript_jp.json",
                "translated_id.json", "flags.json", "usage.json",
                "source.mp4"}
VALID_FORMATS = {"ass", "srt", "both"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/context")
def get_context() -> dict:
    return {"context_md": (CONTEXT_DIR / "context.md").read_text(encoding="utf-8")}


@app.get("/api/sessions")
def list_sessions() -> dict:
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
        if status not in ("failed", "running"):
            continue
        job_id = meta["id"]
        params = meta.get("params", {})
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
        sessions.append({
            "id": job_id,
            "label": f"{ts} · {display_name} · {status}",
            "status": status,
        })
    return {"sessions": sessions}


@app.get("/api/config")
def get_config() -> dict:
    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "providers": cfg.available_providers(),
        "translate_models": cfg.translate_models,
        "transcribe_model": cfg.transcribe_model,
        "ffmpeg": audio.ensure_ffmpeg(),
        "gemini_keys": list(providers.get_all_gemini_keys().keys()),
        "active_gemini_key": providers.get_active_gemini_label(),
    }


@app.put("/api/config/active-key")
def set_active_key(body: dict) -> dict:
    label = (body.get("label") or "").strip()
    if not label:
        raise HTTPException(422, "label wajib diisi.")
    try:
        providers.set_active_gemini_key(label)
    except providers.ConfigError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"active_gemini_key": providers.get_active_gemini_label()}


def _get_job_or_404(job_id: str) -> pipeline.Job:
    job = pipeline.JOBS.get(job_id) or pipeline.load_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan.")
    return job


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     source: str = Form(...),
                     url: str = Form(""),
                     translator: str = Form("gemini"),
                     output_format: str = Form("both"),
                     save_mp4: bool = Form(False),
                     context_override: str = Form(""),
                     additional_context: str = Form(""),
                     file: UploadFile | None = File(None)) -> dict:
    if output_format not in VALID_FORMATS:
        raise HTTPException(422, f"output_format tidak dikenal: {output_format!r}")
    upload_bytes: bytes | None = None
    upload_name: str | None = None
    if source == "url":
        if not url.strip():
            raise HTTPException(422, "URL YouTube belum diisi.")
    elif source == "file":
        if file is None:
            raise HTTPException(422, "File belum dipilih.")
        upload_bytes = await file.read()
        upload_name = file.filename
    else:
        raise HTTPException(422, f"source tidak dikenal: {source!r}")

    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        raise HTTPException(500, str(exc)) from exc
    if not cfg.available_providers().get(translator):
        raise HTTPException(
            422, f"API key untuk provider {translator!r} belum diisi di .env.")

    params = {"source": source, "url": url.strip() or None,
              "translator": translator, "output_format": output_format,
              "save_mp4": save_mp4, "original_filename": upload_name,
              "context_override": context_override.strip() or None,
              "additional_context": additional_context.strip() or None}
    job = pipeline.create_job(params, upload_bytes=upload_bytes,
                              upload_filename=upload_name)
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    return {"id": job.id, "status": job.status, "error": job.error,
            "params": job.params, "events": job.events}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = _get_job_or_404(job_id)

    async def stream():
        idx = 0
        while True:
            while idx < len(job.events):
                payload = json.dumps(job.events[idx], ensure_ascii=False)
                yield f"data: {payload}\n\n"
                idx += 1
            if job.status in ("done", "failed"):
                end = json.dumps({"status": job.status, "error": job.error})
                yield f"event: end\ndata: {end}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/files/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if filename not in DOWNLOADABLE:
        raise HTTPException(404, "File tidak tersedia untuk diunduh.")
    path = job.dir / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} belum ada.")
    return FileResponse(path, filename=f"{job.id}_{filename}")


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str, background_tasks: BackgroundTasks) -> dict:
    job = _get_job_or_404(job_id)
    if job.status == "running":
        raise HTTPException(409, "Job masih berjalan.")
    job.status = "queued"
    job.error = None
    job.events.clear()
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id, "status": "queued"}
