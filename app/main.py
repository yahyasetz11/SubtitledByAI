"""FastAPI app: UI, job API, SSE progress, artifact downloads (§1, §3)."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import audio, pipeline, providers

app = FastAPI(title="SubtitledByAI")
app.mount("/static", StaticFiles(directory="static"), name="static")

GROUPS: dict[str, dict] = {
    "sakurazaka": {
        "label": "Sakurazaka46",
        "shows": [
            {"id": "sakurazaka_sokomagattara", "label": "Sokomagattara Sakurazaka"},
            {"id": "sakurazaka_chokosaku", "label": "Choko-Saku"},
            {"id": "sakurazaka_channel", "label": "Sakurazaka Channel"},
            {"id": "sakurazaka_sakura_meets", "label": "Sakura Meets"},
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
                "source.mp4", "result_jp.ass", "qc_report.json"}
# Files allowed to be streamed inline (for in-browser preview player)
STREAMABLE = {"source.mp4", "audio.mp3", "result.ass", "result_jp.ass"}
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
            "failed_stage": meta.get("failed_stage"),
            "mode": mode,
            "mode_label": _MODE_LABELS.get(mode, mode),
        })
    return {"sessions": sessions}


def fetch_9router_models(base_url: str, api_key: str | None = None) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return [m["id"] for m in data if isinstance(m, dict) and "id" in m]
    except Exception:
        pass
    return []


@app.get("/api/config")
def get_config() -> dict:
    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        return {"ok": False, "error": str(exc)}
    models_cfg = providers.load_models_config()
    if cfg.ninerouter_api_key and cfg.ninerouter_base_url:
        live_models = fetch_9router_models(cfg.ninerouter_base_url, cfg.ninerouter_api_key)
        if live_models:
            models_cfg["translation"]["9router"] = live_models
    return {
        "ok": True,
        "providers": cfg.available_providers(),
        "audio_providers": cfg.available_audio_providers(),
        "translate_models": cfg.translate_models,
        "transcribe_model": cfg.transcribe_model,
        "ffmpeg": audio.ensure_ffmpeg(),
        "gemini_keys": list(providers.get_all_gemini_keys().keys()),
        "active_gemini_key": providers.get_active_gemini_label(),
        "models": models_cfg,
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


def _parse_fallback_json(raw: str, field_name: str) -> list[dict]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"{field_name} bukan JSON valid.") from exc
    if not isinstance(parsed, list):
        raise HTTPException(422, f"{field_name} harus berupa JSON list.")
    return parsed


def _opt_str(slot: dict, field: str, prefix: str) -> str:
    raw = slot.get(field)
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(422, f"{prefix}.{field} harus berupa string.")
    return (raw or "").strip()


def _validate_audio_fallback(raw_slots: list[dict], models_cfg: dict) -> list[dict]:
    if not isinstance(raw_slots, list):
        raise HTTPException(422, "audio_fallback harus berupa list.")
    if len(raw_slots) > 3:
        raise HTTPException(422, "audio_fallback maksimal 3 slot.")
    keys = providers.get_all_gemini_keys()
    trans_models = models_cfg["transcription"]
    if isinstance(trans_models, dict):
        valid_models = set(m for sub in trans_models.values() for m in sub)
    else:
        valid_models = set(trans_models)
    cleaned: list[dict] = []
    for slot in raw_slots:
        if not isinstance(slot, dict):
            raise HTTPException(422, "audio_fallback slot harus berupa object.")
        k = _opt_str(slot, "key_label", "audio_fallback")
        m = _opt_str(slot, "model", "audio_fallback")
        if not k and not m:
            continue
        if not k or not m:
            raise HTTPException(422, "audio_fallback slot harus mengisi key_label dan model.")
        if k not in keys:
            raise HTTPException(422, f"audio_fallback key_label tidak dikenal: {k!r}")
        if m not in valid_models:
            raise HTTPException(422, f"audio_fallback model tidak dikenal: {m!r}")
        cleaned.append({"key_label": k, "model": m})
    return cleaned


def _validate_translate_fallback(
    raw_slots: list[dict],
    models_cfg: dict,
    cfg: providers.Config,
    active_key_label: str | None = None,
) -> list[dict]:
    if not isinstance(raw_slots, list):
        raise HTTPException(422, "translate_fallback harus berupa list.")
    if len(raw_slots) > 3:
        raise HTTPException(422, "translate_fallback maksimal 3 slot.")
    avail = cfg.available_providers()
    keys = providers.get_all_gemini_keys()
    trans_models = models_cfg["translation"]
    default_key_label = (active_key_label or "").strip() or providers.get_active_gemini_label()
    cleaned: list[dict] = []
    for slot in raw_slots:
        if not isinstance(slot, dict):
            raise HTTPException(422, "translate_fallback slot harus berupa object.")
        p = _opt_str(slot, "provider", "translate_fallback")
        m = _opt_str(slot, "model", "translate_fallback")
        k = _opt_str(slot, "key_label", "translate_fallback")
        if not p and not m:
            continue
        if not p or not m:
            raise HTTPException(422, "translate_fallback slot harus mengisi provider dan model.")
        if not avail.get(p):
            raise HTTPException(422, f"translate_fallback provider tidak tersedia atau belum dikonfigurasi: {p!r}")
        if m not in trans_models.get(p, []):
            raise HTTPException(422, f"translate_fallback model tidak dikenal untuk provider {p!r}: {m!r}")
        if p == "gemini":
            k = k or default_key_label
            if k not in keys:
                raise HTTPException(422, f"translate_fallback key_label tidak dikenal: {k!r}")
        else:
            k = None
        cleaned.append({"provider": p, "model": m, "key_label": k})
    return cleaned


@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     source: str = Form(...),
                     url: str = Form(""),
                     translator: str = Form("gemini"),
                     audio_provider: str = Form("gemini"),
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

    translator = translator.strip()
    audio_provider = audio_provider.strip()
    if audio_provider not in ("gemini", "openrouter"):
        raise HTTPException(422, f"audio_provider tidak dikenal: {audio_provider!r}")

    models_cfg = providers.load_models_config()
    trans_models = models_cfg["transcription"]
    if isinstance(trans_models, dict):
        valid_audio_models = set(m for sub in trans_models.values() for m in sub)
    else:
        valid_audio_models = set(trans_models)

    defaults = models_cfg.get("defaults", {})
    default_trans_cfg = defaults.get("transcription", "gemini-3.6-flash")
    default_trans = (
        default_trans_cfg.get("gemini", "gemini-3.6-flash")
        if isinstance(default_trans_cfg, dict)
        else default_trans_cfg
    )
    default_trans_openrouter = (
        default_trans_cfg.get("openrouter", default_trans)
        if isinstance(default_trans_cfg, dict)
        else default_trans
    )

    if mode != "translation_only":
        if audio_provider == "openrouter":
            resolved_transcribe_model = transcribe_model.strip() or default_trans_openrouter
            if not resolved_transcribe_model:
                raise HTTPException(
                    422, "transcribe_model wajib diisi untuk audio provider 'openrouter'.")
        else:
            resolved_transcribe_model = transcribe_model.strip() or default_trans
            if resolved_transcribe_model not in models_cfg["transcription"]["gemini"]:
                raise HTTPException(422, f"transcribe_model tidak dikenal: {resolved_transcribe_model!r}")
    else:
        resolved_transcribe_model = ""

    if mode == "full":
        resolved_timing_model = timing_model.strip() or defaults.get("timing", "gemini-3.6-flash")
        if resolved_timing_model not in valid_audio_models:
            raise HTTPException(422, f"timing_model tidak dikenal: {resolved_timing_model!r}")

        resolved_qc_model = qc_model.strip() or defaults.get("qc", "gemini-3.6-flash")
        if resolved_qc_model not in valid_audio_models:
            raise HTTPException(422, f"qc_model tidak dikenal: {resolved_qc_model!r}")
    else:
        resolved_timing_model = ""
        resolved_qc_model = ""

    translation_defaults = models_cfg["defaults"]["translation"]
    if mode != "transcription_only":
        if not translate_model.strip() and translator not in translation_defaults:
            raise HTTPException(422, f"translator tidak dikenal: {translator!r}")
        resolved_translate_model = translate_model.strip() or translation_defaults[translator]
        if translator in ("openrouter", "9router"):
            if not resolved_translate_model:
                raise HTTPException(
                    422, f"translate_model wajib diisi untuk provider {translator!r}.")
        else:
            provider_model_list = models_cfg["translation"].get(translator, [])
            if resolved_translate_model not in provider_model_list:
                raise HTTPException(422, f"translate_model tidak dikenal untuk provider {translator!r}: {resolved_translate_model!r}")
    else:
        resolved_translate_model = ""

    raw_audio_fallback = _parse_fallback_json(audio_fallback, "audio_fallback")
    raw_translate_fallback = _parse_fallback_json(translate_fallback, "translate_fallback")

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
    elif mode == "translation_only" and source_job_id.strip():
        # Continuity from a prior session — no file upload needed
        upload_bytes = None
        upload_name = None
    else:
        if mode == "translation_only" and file is None and not source_job_id.strip():
            raise HTTPException(
                422, "Translation Only butuh file JSON/.ass atau source_job_id.")
        elif mode != "translation_only":
            raise HTTPException(422, f"source tidak dikenal: {source!r}")

    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        raise HTTPException(500, str(exc)) from exc
    if mode != "translation_only" and not cfg.available_audio_providers().get(audio_provider):
        raise HTTPException(
            422, f"API key untuk audio provider {audio_provider!r} belum diisi di .env.")
    if (
        mode != "transcription_only"
        and translator != "9router"
        and not cfg.available_providers().get(translator)
    ):
        raise HTTPException(
            422, f"API key untuk provider {translator!r} belum diisi di .env.")

    validated_audio_fallback = _validate_audio_fallback(raw_audio_fallback, models_cfg)
    validated_translate_fallback = _validate_translate_fallback(
        raw_translate_fallback, models_cfg, cfg
    )

    params = {
        "mode": mode,
        "source": source,
        "url": url.strip() or None,
        "translator": translator,
        "audio_provider": audio_provider,
        "output_format": output_format,
        "save_mp4": save_mp4,
        "original_filename": upload_name,
        "group": group.strip() or "sakurazaka",
        "context_preset": context_preset.strip() or "sakurazaka_sokomagattara",
        "context_override": context_override.strip() or None,
        "additional_context": additional_context.strip() or None,
        "transcribe_model": resolved_transcribe_model,
        "translate_model": resolved_translate_model,
        "timing_model": resolved_timing_model,
        "qc_model": resolved_qc_model,
        "deep_audio_qc": bool(deep_audio_qc),
        "audio_fallback": validated_audio_fallback,
        "translate_fallback": validated_translate_fallback,
        "source_job_id": source_job_id.strip() or None,
    }
    job = pipeline.create_job(params, upload_bytes=upload_bytes,
                              upload_filename=upload_name)
    background_tasks.add_task(pipeline.run_job, job.id)
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    return {"id": job.id, "status": job.status, "error": job.error,
            "failed_stage": job.failed_stage,
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
            if job.status in ("done", "failed", "cancelled"):
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
    path = (job.dir / "qc" / "report.json") if filename == "qc_report.json" else (job.dir / filename)
    if not path.exists():
        raise HTTPException(404, f"{filename} belum ada.")
    return FileResponse(path, filename=f"{job.id}_{filename}")


@app.get("/api/jobs/{job_id}/stream/{filename}")
def stream_media(job_id: str, filename: str) -> FileResponse:
    """Serve a job file inline (no attachment header) for browser media playback."""
    job = _get_job_or_404(job_id)
    if filename not in STREAMABLE:
        raise HTTPException(404, "File tidak tersedia untuk di-stream.")
    path = job.dir / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} belum ada.")
    return FileResponse(path, headers={"Content-Disposition": f'inline; filename="{filename}"'})


class RetryRequest(BaseModel):
    transcribe_model: str | None = None
    timing_model: str | None = None
    qc_model: str | None = None
    translate_model: str | None = None
    translator: str | None = None
    audio_provider: str | None = None
    gemini_key_label: str | None = None
    deep_audio_qc: bool | None = None
    audio_fallback: list[dict] | None = None
    translate_fallback: list[dict] | None = None


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str, background_tasks: BackgroundTasks,
          req: RetryRequest | None = None) -> dict:
    live = pipeline.JOBS.get(job_id)
    if live is not None and (
        live.status in ("running", "queued")
        or (live.cancelled and live.status == "queued")
    ):
        raise HTTPException(409, "Job masih berjalan.")

    meta_path = pipeline.OUTPUT_DIR / job_id / "job.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        candidate = pipeline.Job(
            id=meta["id"],
            dir=pipeline.OUTPUT_DIR / job_id,
            params=meta["params"],
            status=meta["status"],
            error=meta.get("error"),
            failed_stage=meta.get("failed_stage"),
            events=meta.get("events", []),
        )
    elif live is not None:
        candidate = live
    else:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan.")

    if candidate.status in ("running", "queued"):
        raise HTTPException(409, "Job masih berjalan.")

    models_cfg = providers.load_models_config()
    updated_params = dict(candidate.params)

    key_label = ((req.gemini_key_label if req else None) or "").strip()
    resolved_key = ""
    if key_label:
        try:
            resolved_key = providers.resolve_gemini_key(key_label)
        except providers.ConfigError as exc:
            raise HTTPException(422, str(exc)) from exc

    try:
        cfg = providers.Config.load()
    except providers.ConfigError as exc:
        if not key_label:
            raise HTTPException(500, str(exc)) from exc
        defaults = models_cfg["defaults"]
        default_trans_cfg = defaults.get("transcription", "gemini-3.6-flash")
        default_trans = (
            default_trans_cfg.get("gemini", "gemini-3.6-flash")
            if isinstance(default_trans_cfg, dict)
            else default_trans_cfg
        )
        cfg = providers.Config(
            gemini_api_key=resolved_key,
            openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None,
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip() or None,
            ninerouter_api_key=(os.getenv("NINEROUTER_API_KEY") or "").strip() or None,
            ninerouter_base_url=(os.getenv("NINEROUTER_BASE_URL") or "http://localhost:20128/v1").strip(),
            openrouter_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip() or None,
            openrouter_base_url=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
            transcribe_model=default_trans,
            translate_models=dict(defaults["translation"]),
            ytdlp_cookies_file=None,
            sub_min_duration=0.7,
            sub_max_duration=7.5,
            sub_merge_gap=0.5,
            sub_cps_flag=25.0,
            chunk_min_seconds=240.0,
            chunk_max_seconds=360.0,
            timing_model=defaults.get("timing", "gemini-2.5-flash"),
            qc_window_size=10,
            qc_batch_size=20,
            qc_model=defaults.get("qc", "gemini-2.5-flash"),
        )

    if resolved_key and not cfg.gemini_api_key:
        import dataclasses
        cfg = dataclasses.replace(cfg, gemini_api_key=resolved_key)

    if req:
        avail = cfg.available_providers()
        prev_translator = updated_params.get("translator") or "gemini"
        audio_provider_val = (req.audio_provider or "").strip()
        if audio_provider_val:
            if audio_provider_val not in ("gemini", "openrouter"):
                raise HTTPException(
                    422, f"audio_provider tidak dikenal: {audio_provider_val!r}")
            if not cfg.available_audio_providers().get(audio_provider_val):
                raise HTTPException(
                    422,
                    f"API key untuk audio provider {audio_provider_val!r} belum diisi di .env.",
                )
            updated_params["audio_provider"] = audio_provider_val

        translator_val = (req.translator or "").strip()
        if translator_val:
            if translator_val not in models_cfg["translation"]:
                raise HTTPException(422, f"translator tidak dikenal: {translator_val!r}")
            if not avail.get(translator_val):
                raise HTTPException(
                    422, f"API key untuk provider {translator_val!r} belum diisi di .env."
                )
            updated_params["translator"] = translator_val

        current_translator = updated_params.get("translator") or "gemini"
        valid_translate_models = models_cfg["translation"].get(current_translator, [])
        translate_model_val = (req.translate_model or "").strip()
        if translate_model_val:
            if translate_model_val not in valid_translate_models:
                raise HTTPException(
                    422,
                    f"translate_model tidak dikenal untuk provider {current_translator!r}: {translate_model_val!r}",
                )
            updated_params["translate_model"] = translate_model_val
        elif translator_val and translator_val != prev_translator:
            if updated_params.get("translate_model") not in valid_translate_models:
                updated_params["translate_model"] = models_cfg["defaults"]["translation"][current_translator]

        trans_models = models_cfg["transcription"]
        if isinstance(trans_models, dict):
            valid_audio_models = set(m for sub in trans_models.values() for m in sub)
        else:
            valid_audio_models = set(trans_models)
        for field_name, raw_val in (
            ("transcribe_model", req.transcribe_model),
            ("timing_model", req.timing_model),
            ("qc_model", req.qc_model),
        ):
            model_val = (raw_val or "").strip()
            if model_val:
                if model_val not in valid_audio_models:
                    raise HTTPException(422, f"{field_name} tidak dikenal: {model_val!r}")
                updated_params[field_name] = model_val

        if req.deep_audio_qc is not None:
            updated_params["deep_audio_qc"] = req.deep_audio_qc

        if req.audio_fallback is not None:
            updated_params["audio_fallback"] = _validate_audio_fallback(
                req.audio_fallback, models_cfg
            )

        if req.translate_fallback is not None:
            updated_params["translate_fallback"] = _validate_translate_fallback(
                req.translate_fallback,
                models_cfg,
                cfg,
                active_key_label=key_label or providers.get_active_gemini_label(),
            )

    prev_key_label = providers.get_active_gemini_label()
    switched_key = False
    if key_label:
        providers.set_active_gemini_key(key_label)
        switched_key = True
    try:
        candidate.params = updated_params
        candidate.status = "queued"
        candidate.error = None
        candidate.failed_stage = None
        candidate.cancelled = False
        candidate.events.clear()
        pipeline._save(candidate)
        pipeline.JOBS[job_id] = candidate
    except Exception:
        if switched_key:
            providers._active_gemini_label = prev_key_label
        raise

    background_tasks.add_task(pipeline.run_job, candidate.id)
    return {"job_id": candidate.id, "status": "queued"}


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    if job.status not in ("running", "queued"):
        raise HTTPException(409, f"Job tidak bisa dibatalkan (status: {job.status}).")
    found = pipeline.cancel_job(job_id)
    if not found:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan.")
    return {"job_id": job_id, "status": "cancelling"}
