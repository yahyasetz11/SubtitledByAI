"""FastAPI app: UI, job API, SSE progress, artifact downloads (§1, §3)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app import audio, pipeline, providers

app = FastAPI(title="SubtitledByAI")

GROUPS: dict[str, dict] = {
    "sakurazaka": {
        "label": "Sakurazaka46",
        "shows": [
            {"id": "sakurazaka_sokomagattara", "label": "Sokomagattara Sakurazaka"},
            {"id": "sakurazaka_chokosaku", "label": "Choko-Saku"},
            {"id": "sakurazaka_channel", "label": "Sakurazaka Channel"},
        ],
    },
    "nogizaka": {
        "label": "Nogizaka46",
        "shows": [
            {"id": "nogizaka_kojichuu", "label": "Nogizaka Kojichuu"},
            {"id": "nogizaka_enchouchuu", "label": "Nogizaka Kouji Enchouchuu"},
            {"id": "nogizaka_haishinchuu", "label": "Nogizaka Haishinchuu"},
        ],
    },
    "hinatazaka": {
        "label": "Hinatazaka46",
        "shows": [
            {"id": "hinatazaka_aimashou", "label": "Hinatazaka de Aimashou"},
            {"id": "hinatazaka_narimashou", "label": "Hinatazaka de Narimashou"},
            {"id": "hinatazaka_channel", "label": "Hinatazaka Channel"},
        ],
    },
}

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


@app.get("/api/contexts")
def list_contexts() -> dict:
    groups = [
        {"id": gid, "label": gdata["label"], "shows": gdata["shows"]}
        for gid, gdata in GROUPS.items()
    ]
    groups.append({"id": "else", "label": "Else / Other", "shows": []})
    return {"groups": groups}


@app.get("/api/context")
def get_context(preset: str = Query("sakurazaka_sokomagattara")) -> dict:
    path = CONTEXT_DIR / f"context_{preset}.md"
    if not path.exists():
        raise HTTPException(404, f"Preset '{preset}' tidak ditemukan.")
    return {"context_md": path.read_text(encoding="utf-8")}


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
        "models": providers.load_models_config(),
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
                     group: str = Form("sakurazaka"),
                     context_preset: str = Form("sakurazaka_sokomagattara"),
                     context_override: str = Form(""),
                     additional_context: str = Form(""),
                     transcribe_model: str = Form(""),
                     translate_model: str = Form(""),
                     file: UploadFile | None = File(None)) -> dict:
    if output_format not in VALID_FORMATS:
        raise HTTPException(422, f"output_format tidak dikenal: {output_format!r}")

    models_cfg = providers.load_models_config()

    resolved_transcribe_model = transcribe_model.strip() or models_cfg["defaults"]["transcription"]
    if resolved_transcribe_model not in models_cfg["transcription"]:
        raise HTTPException(422, f"transcribe_model tidak dikenal: {resolved_transcribe_model!r}")

    resolved_translate_model = translate_model.strip() or models_cfg["defaults"]["translation"].get(translator, "")
    provider_model_list = models_cfg["translation"].get(translator, [])
    if resolved_translate_model and resolved_translate_model not in provider_model_list:
        raise HTTPException(422, f"translate_model tidak dikenal untuk provider {translator!r}: {resolved_translate_model!r}")

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
              "group": group.strip() or "sakurazaka",
              "context_preset": context_preset.strip() or "sakurazaka_sokomagattara",
              "context_override": context_override.strip() or None,
              "additional_context": additional_context.strip() or None,
              "transcribe_model": resolved_transcribe_model,
              "translate_model": resolved_translate_model}
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
