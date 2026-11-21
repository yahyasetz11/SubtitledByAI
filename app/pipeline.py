"""Job orchestration: stages, checkpoints, SSE-ready events (§1, §9 layer 3).

Every stage skips itself when its artifact already exists in output/{job_id}/,
so run_job() after a failure resumes instead of redoing work. Transcription
additionally checkpoints per chunk and translation per batch.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app import audio, providers, qc, subtitle, timing, transcribe, translate
from app.providers import UsageTracker
from app.subtitle import SubEvent

OUTPUT_DIR = Path("output")
CONTEXT_DIR = Path("context")
JOBS: dict[str, "Job"] = {}


class PipelineError(Exception):
    """Job cannot proceed (bad params, missing source file, ...)."""


class JobCancelledError(Exception):
    """Raised when a job is cancelled mid-run."""


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


def create_job(params: dict, upload_bytes: bytes | None = None,
               upload_filename: str | None = None) -> Job:
    job_id = (datetime.now().strftime("%Y%m%d-%H%M%S")
              + "-" + uuid.uuid4().hex[:6])
    job_dir = OUTPUT_DIR / job_id
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, dir=job_dir, params=params)
    if upload_bytes is not None:
        suffix = Path(upload_filename or "source.bin").suffix or ".bin"
        (job_dir / f"source{suffix}").write_bytes(upload_bytes)
    JOBS[job_id] = job
    _save(job)
    return job


def load_job(job_id: str) -> Job | None:
    meta_path = OUTPUT_DIR / job_id / "job.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job = Job(id=meta["id"], dir=OUTPUT_DIR / job_id, params=meta["params"],
              status=meta["status"], error=meta.get("error"),
              failed_stage=meta.get("failed_stage"),
              events=meta.get("events", []))
    JOBS[job.id] = job
    return job


def cancel_job(job_id: str) -> bool:
    """Signal a running job to stop. Returns True if the job was found."""
    job = JOBS.get(job_id) or load_job(job_id)
    if job is None:
        return False
    job.cancelled = True
    return True


def _check_cancelled(job: Job) -> None:
    """Raise JobCancelledError if the cancel flag has been set."""
    if job.cancelled:
        raise JobCancelledError("Job dibatalkan oleh pengguna.")


def _save(job: Job) -> None:
    payload = {"id": job.id, "params": job.params, "status": job.status,
               "error": job.error, "failed_stage": job.failed_stage,
               "events": job.events}
    (job.dir / "job.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def run_job(job_id: str) -> None:
    job = JOBS.get(job_id) or load_job(job_id)
    if job is None:
        raise KeyError(f"Job tidak ditemukan: {job_id}")
    cfg = providers.Config.load()
    active_key_label = providers.get_active_gemini_label()
    tracker = UsageTracker(job.dir / "usage.json")
    job.status, job.error, job.failed_stage = "running", None, None
    _save(job)
    stage = "init"
    mode = job.params.get("mode", "full")
    try:
        if mode == "transcription_only":
            stage = "download"
            source = _stage_download(job, cfg)
            _check_cancelled(job)
            stage = "normalize"
            audio_path = _stage_normalize(job, source)
            _check_cancelled(job)
            stage = "chunk"
            chunks = _stage_chunk(job, audio_path, cfg)
            _check_cancelled(job)
            stage = "transcribe"
            utterances = _stage_transcribe(
                job, cfg, tracker, chunks, active_key_label=active_key_label
            )
            _check_cancelled(job)
            stage = "format"
            files = _stage_format_transcript(job, utterances)
        elif mode == "translation_only":
            stage = "load_transcript"
            utterances = _stage_load_for_translation(job)
            _check_cancelled(job)
            stage = "translate"
            rows = _stage_translate(
                job, cfg, tracker, utterances, active_key_label=active_key_label
            )
            _check_cancelled(job)
            stage = "timing"
            rows = _stage_timing(
                job, cfg, tracker, rows, chunks=None, audio_path=None,
                active_key_label=active_key_label,
            )
            _check_cancelled(job)
            stage = "qc"
            rows = _stage_qc(
                job, cfg, tracker, rows, audio_path=None,
                active_key_label=active_key_label,
            )
            _check_cancelled(job)
            stage = "format"
            files = _stage_format(job, cfg, rows)
        else:  # "full"
            stage = "download"
            source = _stage_download(job, cfg)
            _check_cancelled(job)
            stage = "normalize"
            audio_path = _stage_normalize(job, source)
            _check_cancelled(job)
            stage = "chunk"
            chunks = _stage_chunk(job, audio_path, cfg)
            _check_cancelled(job)
            stage = "transcribe"
            utterances = _stage_transcribe(
                job, cfg, tracker, chunks, active_key_label=active_key_label
            )
            _check_cancelled(job)
            stage = "translate"
            rows = _stage_translate(
                job, cfg, tracker, utterances, active_key_label=active_key_label
            )
            _check_cancelled(job)
            stage = "timing"
            rows = _stage_timing(
                job, cfg, tracker, rows, chunks=chunks, audio_path=audio_path,
                active_key_label=active_key_label,
            )
            _check_cancelled(job)
            stage = "qc"
            rows = _stage_qc(
                job, cfg, tracker, rows, audio_path=audio_path,
                active_key_label=active_key_label,
            )
            _check_cancelled(job)
            stage = "format"
            files = _stage_format(job, cfg, rows)

        job.status = "done"
        _emit(job, "done", "done",
              result={"usage": tracker.summary(), "files": files})
    except JobCancelledError:
        job.status = "cancelled"
        job.error = "Dibatalkan oleh pengguna."
        job.failed_stage = None
        _emit(job, stage, "cancelled", message="Job dibatalkan oleh pengguna.")
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        job.failed_stage = stage
        _emit(job, stage, "error", message=str(exc), failed_stage=stage)


def _load_json_safe(path: Path, default, expected_type=None):
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default
    if expected_type is not None and not isinstance(data, expected_type):
        return default
    return data


def _invalidate_downstream(
    job: Job,
    *,
    clear_timing: bool = False,
    clear_qc: bool = False,
) -> None:
    if clear_timing:
        (job.dir / "timing" / "utterances.json").unlink(missing_ok=True)
        (job.dir / "timing" / "report.json").unlink(missing_ok=True)
        for p in (job.dir / "chunks").glob("timing_*.json"):
            p.unlink(missing_ok=True)
    if clear_qc:
        (job.dir / "qc" / "utterances.json").unlink(missing_ok=True)
        (job.dir / "qc" / "report.json").unlink(missing_ok=True)
        (job.dir / "chunks" / "qc_pass1.json").unlink(missing_ok=True)
        for p in (job.dir / "chunks").glob("qc_audio_*.json"):
            p.unlink(missing_ok=True)
        (job.dir / "chunks" / "qc_input.sha256").unlink(missing_ok=True)


def _qc_input_digest(
    rows: list[dict],
    glossary: dict,
    timing_report: list,
    chunk_offsets: list,
    cps_threshold: float,
    has_audio: bool,
) -> str:
    payload = json.dumps(
        [rows, glossary, timing_report, chunk_offsets, float(cps_threshold), bool(has_audio)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_audio_fallback(params: dict) -> list[providers.AudioFallbackSlot]:
    raw = params.get("audio_fallback")
    if not isinstance(raw, (list, tuple)):
        return []
    slots: list[providers.AudioFallbackSlot] = []
    for s in raw[:3]:
        if not isinstance(s, dict):
            continue
        raw_label = s.get("key_label")
        raw_model = s.get("model")
        key_label = raw_label.strip() if isinstance(raw_label, str) else ""
        model = raw_model.strip() if isinstance(raw_model, str) else ""
        if key_label and model:
            slots.append(providers.AudioFallbackSlot(key_label=key_label, model=model))
    return slots


def _parse_translate_fallback(params: dict) -> list[providers.TranslateFallbackSlot]:
    raw = params.get("translate_fallback")
    if not isinstance(raw, (list, tuple)):
        return []
    slots: list[providers.TranslateFallbackSlot] = []
    for s in raw[:3]:
        if not isinstance(s, dict):
            continue
        raw_provider = s.get("provider")
        raw_model = s.get("model")
        raw_label = s.get("key_label")
        provider = raw_provider.strip() if isinstance(raw_provider, str) else ""
        model = raw_model.strip() if isinstance(raw_model, str) else ""
        key_label = (
            (raw_label.strip() or None)
            if provider == "gemini" and isinstance(raw_label, str)
            else None
        )
        if provider and model:
            slots.append(providers.TranslateFallbackSlot(
                provider=provider,
                model=model,
                key_label=key_label,
            ))
    return slots


def _make_audio_runner(
    job: Job,
    cfg,
    stage: str,
    primary_model: str,
    primary_client=None,
    *,
    primary_provider: str = "gemini",
    active_key_label: str | None = None,
):
    client_model = getattr(primary_client, "model", None)
    if not isinstance(client_model, str):
        client_model = ""
    stage_transcribe_default = (
        "" if providers._is_transcription_model(cfg.transcribe_model or "")
        else cfg.transcribe_model
    )
    stage_default = {
        "timing": cfg.timing_model,
        "qc": cfg.qc_model,
    }.get(stage, stage_transcribe_default or cfg.transcribe_model)
    primary_model = (
        primary_model
        or client_model
        or stage_default
        or stage_transcribe_default
        or "gemini-2.5-flash"
    )
    if primary_provider == "openrouter" and primary_model.startswith("gemini-"):
        primary_model = f"google/{primary_model}"
    primary_label = active_key_label or providers.get_active_gemini_label()
    primary_slot = providers.AudioFallbackSlot(
        key_label=primary_label,
        model=primary_model,
        provider=primary_provider,
    )
    fallback_slots = _parse_audio_fallback(job.params)

    def _build(slot: providers.AudioFallbackSlot):
        if (
            primary_client is not None
            and slot == primary_slot
            and hasattr(primary_client, "upload_audio")
        ):
            return primary_client
        key_label = None if slot == primary_slot else slot.key_label
        provider = getattr(slot, "provider", None) or primary_provider
        model = slot.model
        if provider == "openrouter" and model.startswith("gemini-"):
            model = f"google/{model}"
        return providers.make_audio_client(
            cfg, provider=provider, model=model, key_label=key_label
        )

    def _resolve_key(slot: providers.AudioFallbackSlot) -> str:
        if slot.provider != "gemini":
            return f"{slot.provider}:{slot.key_label or ''}"
        if slot == primary_slot and cfg.gemini_api_key:
            return cfg.gemini_api_key
        return providers.resolve_gemini_key(slot.key_label, fallback_key=cfg.gemini_api_key)

    def runner(chunk_path: Path, system: str, user: str, on_upload_done=None):
        def _on_fb(idx: int, slot: providers.AudioFallbackSlot, exc: Exception):
            _emit(
                job, stage, "progress",
                f"{chunk_path.name} error ({exc}) -> mencoba Fallback {idx}: "
                f"key {slot.key_label} / {slot.model}",
            )

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
        winning_model = getattr(winning, "model", None)
        return resp, (winning_model if isinstance(winning_model, str) and winning_model else primary_model)

    return runner


def _make_text_runner(
    job: Job,
    cfg,
    stage: str,
    primary_client=None,
    *,
    active_key_label: str | None = None,
):
    provider = job.params.get("translator") or "gemini"
    client_model = getattr(primary_client, "model", None)
    if not isinstance(client_model, str):
        client_model = ""
    primary_model = (
        job.params.get("translate_model")
        or client_model
        or cfg.translate_models.get(provider, "")
        or cfg.translate_models.get("gemini", "")
        or "gemini-2.5-flash"
    )
    primary_label = active_key_label or providers.get_active_gemini_label()
    primary_slot = providers.TranslateFallbackSlot(
        provider=provider,
        model=primary_model,
        key_label=primary_label if provider == "gemini" else None,
    )
    fallback_slots = _parse_translate_fallback(job.params)

    def _build(slot: providers.TranslateFallbackSlot):
        if (
            primary_client is not None
            and slot == primary_slot
            and hasattr(primary_client, "generate")
        ):
            return primary_client
        key_label = None if (slot == primary_slot and slot.provider == "gemini") else slot.key_label
        return providers.make_text_client(
            cfg, slot.provider, model=slot.model, key_label=key_label
        )

    def runner(system: str, user: str):
        def _on_fb(idx: int, slot: providers.TranslateFallbackSlot, exc: Exception):
            key_info = f" ({slot.key_label})" if slot.key_label else ""
            _emit(
                job, stage, "progress",
                f"error ({exc}) -> mencoba Fallback {idx}: "
                f"{slot.provider}{key_info} / {slot.model}",
            )

        resp, winning = providers.execute_with_fallback(
            primary=primary_slot,
            fallback_slots=fallback_slots,
            build_client=_build,
            call_fn=lambda c: c.generate(system=system, user=user),
            on_fallback=_on_fb,
        )
        winning_model = getattr(winning, "model", None)
        return resp, (winning_model if isinstance(winning_model, str) and winning_model else primary_model)

    return runner


def _read_context(group: str | None, preset: str | None) -> tuple[str, str]:
    if not group or group == "else":
        return "", ""
    members_path = CONTEXT_DIR / f"members_{group}.md"
    members_md = members_path.read_text(encoding="utf-8") if members_path.exists() else ""
    if not preset:
        return "", members_md
    path = CONTEXT_DIR / f"context_{preset}.md"
    if not path.exists():
        raise FileNotFoundError(f"Context preset '{preset}' tidak ditemukan: {path}")
    context_md = path.read_text(encoding="utf-8")
    return context_md, members_md


def _find_source(job: Job) -> Path | None:
    matches = sorted(job.dir.glob("source.*"))
    return matches[0] if matches else None


def _find_source_by_ext(job: "Job", ext: str) -> "Path | None":
    """Return the first file in job.dir matching the given extension."""
    matches = sorted(job.dir.glob(f"source{ext}"))
    return matches[0] if matches else None


def _load_utterances_from_source(
    source_type: str,
    *,
    json_path: "Path | None" = None,
    ass_path: "Path | None" = None,
) -> "list[transcribe.Utterance]":
    """Load utterances for Translation Only mode from a JSON or .ass file."""
    if source_type == "json":
        if json_path is None or not json_path.exists():
            raise PipelineError("transcript_jp.json tidak ditemukan.")
        return transcribe.load_transcript(json_path)
    if source_type == "ass":
        if ass_path is None or not ass_path.exists():
            raise PipelineError("File .ass tidak ditemukan.")
        from app.subtitle import parse_ass_utterances
        return parse_ass_utterances(ass_path.read_text(encoding="utf-8"))
    raise ValueError(f"source_type tidak dikenal: {source_type!r}")


def _stage_download(job: Job, cfg) -> Path:
    existing = _find_source(job)
    if existing is not None:
        _emit(job, "download", "done",
              f"{existing.name} sudah ada (checkpoint)")
        return existing
    if job.params.get("source") != "url" or not job.params.get("url"):
        raise PipelineError("File sumber tidak ditemukan dan tidak ada URL.")
    _emit(job, "download", "start", "mengunduh dari YouTube...")
    path = audio.download_youtube(
        job.params["url"], job.dir,
        save_mp4=bool(job.params.get("save_mp4", False)),
        cookies_file=cfg.ytdlp_cookies_file)
    _emit(job, "download", "done", path.name)
    return path


def _stage_normalize(job: Job, source: Path) -> Path:
    dst = job.dir / "audio.mp3"
    if dst.exists():
        _emit(job, "normalize", "done", "audio.mp3 sudah ada (checkpoint)")
        return dst
    _emit(job, "normalize", "start",
          "normalisasi audio (mp3 mono 16kHz 64kbps)")
    audio.normalize_audio(source, dst)
    _emit(job, "normalize", "done", "audio.mp3")
    return dst


def _stage_chunk(job: Job, audio_path: Path, cfg) -> list[audio.PlannedChunk]:
    offsets_path = job.dir / "offsets.json"
    chunks_dir = job.dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    if offsets_path.exists():
        chunks = audio.load_offsets(offsets_path)
        _emit(job, "chunk", "done", f"{len(chunks)} chunk (checkpoint)")
    else:
        _emit(job, "chunk", "start",
              "deteksi keheningan & perencanaan chunk")
        duration = audio.get_duration(audio_path)
        silence_cache: dict[int, list] = {}

        def get_silences(threshold_db: int):
            if threshold_db not in silence_cache:
                silence_cache[threshold_db] = audio.detect_silences(
                    audio_path, threshold_db)
            return silence_cache[threshold_db]

        chunks = audio.plan_chunks(
            duration, get_silences,
            min_seconds=cfg.chunk_min_seconds,
            max_seconds=cfg.chunk_max_seconds,
        )
        audio.write_offsets(chunks, offsets_path)
        _emit(job, "chunk", "done", f"{len(chunks)} chunk")
    for c in chunks:
        dst = chunks_dir / f"chunk_{c.index:03d}.mp3"
        if not dst.exists():
            audio.cut_chunk(audio_path, dst, c.start, c.end)
    return chunks


def _stage_transcribe(job: Job, cfg, tracker,
                      chunks: list[audio.PlannedChunk],
                      *, active_key_label: str | None = None
                      ) -> list[transcribe.Utterance]:
    final_path = job.dir / "transcript_jp.json"
    if final_path.exists():
        utterances = transcribe.load_transcript(final_path)
        _emit(job, "transcribe", "done",
              f"{len(utterances)} ujaran (checkpoint)")
        return utterances
    context_md, members_md = _read_context(
        job.params.get("group"), job.params.get("context_preset")
    )
    if job.params.get("context_override"):
        context_md = job.params["context_override"]
    additional_context = job.params.get("additional_context") or None
    system, user = transcribe.build_transcribe_prompts(context_md, members_md,
                                                       additional_context=additional_context)
    primary_model = job.params.get("transcribe_model") or cfg.transcribe_model
    audio_provider = job.params.get("audio_provider") or "gemini"
    gemini = providers.make_audio_client(
        cfg, provider=audio_provider, model=primary_model
    )
    audio_runner = _make_audio_runner(
        job, cfg, "transcribe", primary_model, primary_client=gemini,
        primary_provider=audio_provider,
        active_key_label=active_key_label,
    )
    n = len(chunks)
    _emit(job, "transcribe", "start", f"transkripsi {n} chunk")
    per_chunk: list[list[transcribe.Utterance]] = []
    for c in chunks:
        ckpt = job.dir / "chunks" / f"transcript_{c.index:03d}.json"
        if ckpt.exists():
            utts = transcribe.load_transcript(ckpt)
        else:
            idx = c.index
            chunk_path = job.dir / "chunks" / f"chunk_{c.index:03d}.mp3"

            _emit(job, "transcribe", "progress",
                  f"mengunggah audio chunk {idx}/{n}...")

            ctx: dict = {
                "stop": threading.Event(),
                "generate_started": threading.Event(),
                "t0": time.monotonic(),
                "idx": idx,
                "n": n,
            }

            def _heartbeat(ctx: dict = ctx) -> None:
                while not ctx["stop"].wait(30):
                    if ctx["generate_started"].is_set():
                        elapsed = int(time.monotonic() - ctx["t0"])
                        _emit(job, "transcribe", "progress",
                              f"masih menunggu Gemini chunk "
                              f"{ctx['idx']}/{ctx['n']}... ({elapsed}s elapsed)")

            def _on_upload_done(ctx: dict = ctx) -> None:
                elapsed = int(time.monotonic() - ctx["t0"])
                ctx["generate_started"].set()
                _emit(job, "transcribe", "progress",
                      f"upload selesai ({elapsed}s), "
                      f"menunggu respons Gemini chunk {ctx['idx']}/{ctx['n']}...")

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()
            try:
                utts = transcribe.transcribe_chunk(
                    gemini, chunk_path, system, user, tracker,
                    on_upload_done=_on_upload_done,
                    audio_runner=audio_runner,
                )
            finally:
                ctx["stop"].set()
                hb.join(timeout=2)

            transcribe.save_transcript(utts, ckpt)
        per_chunk.append(utts)
        _emit(job, "transcribe", "progress",
              f"chunk {c.index}/{n} selesai")
    merged = transcribe.merge_transcripts(per_chunk, chunks)
    transcribe.save_transcript(merged, final_path)
    _emit(job, "transcribe", "done", f"{len(merged)} ujaran")
    return merged


def _stage_format_transcript(
    job: "Job", utterances: "list[transcribe.Utterance]"
) -> list[str]:
    """Transcription Only: write transcript_jp.json + optional result_jp.ass."""
    fmt = job.params.get("output_format", "json")
    _emit(job, "format", "start", f"menyimpan hasil transkripsi ({fmt})")
    files = ["transcript_jp.json"]
    if fmt in ("ass", "both"):
        events = [
            subtitle.SubEvent(start=u.start, end=u.end, text=u.ja, type=u.type)
            for u in utterances
        ]
        template = (CONTEXT_DIR / "template.ass").read_text(encoding="utf-8")
        (job.dir / "result_jp.ass").write_text(
            subtitle.render_ass(events, template), encoding="utf-8"
        )
        files.append("result_jp.ass")
    if (job.dir / "source.mp4").exists():
        files.append("source.mp4")
    _emit(job, "format", "done", ", ".join(files))
    return files


def _stage_load_for_translation(job: "Job") -> "list[transcribe.Utterance]":
    """Translation Only: load utterances from uploaded file or a prior job."""
    # Checkpoint: already copied in a previous run
    final_path = job.dir / "transcript_jp.json"
    if final_path.exists():
        utts = transcribe.load_transcript(final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran (checkpoint)")
        return utts

    _emit(job, "load_transcript", "start", "memuat sumber transkripsi...")

    # Option A: reference a prior job by ID
    source_job_id = job.params.get("source_job_id")
    if source_job_id:
        src_path = OUTPUT_DIR / source_job_id / "transcript_jp.json"
        if not src_path.exists():
            raise PipelineError(
                f"transcript_jp.json tidak ditemukan di job {source_job_id}.")
        utts = transcribe.load_transcript(src_path)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari job {source_job_id}")
        return utts

    # Option B: uploaded JSON file
    json_src = _find_source_by_ext(job, ".json")
    if json_src:
        utts = _load_utterances_from_source("json", json_path=json_src)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari JSON")
        return utts

    # Option C: uploaded .ass file
    ass_src = _find_source_by_ext(job, ".ass")
    if ass_src:
        utts = _load_utterances_from_source("ass", ass_path=ass_src)
        transcribe.save_transcript(utts, final_path)
        _emit(job, "load_transcript", "done",
              f"{len(utts)} ujaran dari .ass")
        return utts

    raise PipelineError(
        "Translation Only: tidak ada source_job_id, JSON, atau .ass yang tersedia.")


def _stage_translate(job: Job, cfg, tracker,
                     utterances: list[transcribe.Utterance],
                     *, active_key_label: str | None = None) -> list[dict]:
    out_path = job.dir / "translated_id.json"
    if out_path.exists():
        rows = _load_json_safe(out_path, None, list)
        if rows is not None:
            _emit(job, "translate", "done", f"{len(rows)} baris (checkpoint)")
            return rows
    _invalidate_downstream(job, clear_timing=True, clear_qc=True)
    context_md, members_md = _read_context(
        job.params.get("group"), job.params.get("context_preset")
    )
    if job.params.get("context_override"):
        context_md = job.params["context_override"]
    additional_context = job.params.get("additional_context") or None
    system = translate.build_translate_system(context_md, members_md,
                                              additional_context=additional_context)
    translator = job.params.get("translator") or "gemini"
    client = providers.make_text_client(
        cfg, translator,
        model=job.params.get("translate_model") or None,
    )
    text_runner = _make_text_runner(
        job, cfg, "translate", primary_client=client,
        active_key_label=active_key_label,
    )
    glossary: dict[str, str] = {}
    _emit(job, "translate", "start",
          f"terjemahkan {len(utterances)} ujaran via {translator}")

    def load_ckpt(batch_index: int):
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        return _load_json_safe(p, None)

    def save_ckpt(batch_index: int, payload) -> None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
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


def _stage_timing(job: Job, cfg, tracker, rows: list[dict],
                  chunks, audio_path: Path | None,
                  *, active_key_label: str | None = None) -> list[dict]:
    timing_path = job.dir / "timing" / "utterances.json"
    if timing_path.exists():
        result = timing.load_timing(job.dir)
        _emit(job, "timing", "done", f"{len(result)} ujaran (checkpoint)")
        return result

    if chunks is None or audio_path is None:
        _emit(job, "timing", "done", "dilewati (translation_only mode)")
        return rows

    _invalidate_downstream(job, clear_qc=True)

    n = len(chunks)
    raw_timing_model = (job.params.get("timing_model") or "").strip()
    if job.params.get("timing_provider"):
        timing_provider = job.params["timing_provider"]
    elif job.params.get("timing_qc_provider"):
        timing_provider = job.params["timing_qc_provider"]
    elif raw_timing_model.startswith("gemini-") and "/" not in raw_timing_model:
        timing_provider = "gemini"
    elif "/" in raw_timing_model:
        timing_provider = "openrouter"
    else:
        timing_provider = job.params.get("audio_provider") or "gemini"

    timing_model = (
        raw_timing_model
        or (
            job.params.get("transcribe_model")
            if timing_provider == (job.params.get("audio_provider") or "gemini")
            and not providers._is_transcription_model(job.params.get("transcribe_model") or "")
            else ""
        )
        or (cfg.timing_model if timing_provider == "gemini" else "google/gemini-3.8-flash")
    )
    if timing_provider == "openrouter" and timing_model.startswith("gemini-"):
        timing_model = f"google/{timing_model}"
    elif timing_provider == "gemini" and timing_model.startswith("google/gemini-"):
        timing_model = timing_model.replace("google/", "")
    _emit(job, "timing", "start",
          f"koreksi timing {len(rows)} ujaran via {timing_model} ({n} chunk)")

    gemini = providers.make_audio_client(
        cfg, provider=timing_provider, model=timing_model
    )
    runner4 = _make_audio_runner(
        job, cfg, "timing", timing_model, primary_client=gemini,
        primary_provider=timing_provider,
        active_key_label=active_key_label,
    )
    utterances = [transcribe.Utterance.from_dict(r) for r in rows]

    def on_progress(chunk_idx: int, total: int) -> None:
        _emit(job, "timing", "progress", f"chunk {chunk_idx}/{total}")

    def load_ckpt(chunk_idx: int):
        p = job.dir / "chunks" / f"timing_{chunk_idx:03d}.json"
        return _load_json_safe(p, None)

    def save_ckpt(chunk_idx: int, payload: dict) -> None:
        p = job.dir / "chunks" / f"timing_{chunk_idx:03d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def audio_runner_3(p, s, u):
        return runner4(p, s, u, None)

    corrected, report = timing.reanchor_all(
        utterances, chunks, job.dir / "chunks",
        gemini=gemini, tracker=tracker, on_progress=on_progress,
        audio_runner=audio_runner_3,
        load_checkpoint=load_ckpt,
        save_checkpoint=save_ckpt,
    )

    sorted_corrected = sorted(corrected, key=lambda u: (float(u.start), float(u.end)))
    by_ja_type: dict[tuple[str, str], deque[tuple[int, transcribe.Utterance]]] = defaultdict(deque)
    by_ja: dict[str, deque[tuple[int, transcribe.Utterance]]] = defaultdict(deque)
    for c_idx, u in enumerate(sorted_corrected):
        by_ja_type[(u.ja, u.type)].append((c_idx, u))
        by_ja[u.ja].append((c_idx, u))

    ordered_rows = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1]["start"]), float(item[1]["end"]), item[0]),
    )
    used_c_idxs: set[int] = set()

    def _pop_unused(q):
        if not q:
            return None
        while q and q[0][0] in used_c_idxs:
            q.popleft()
        return q.popleft() if q else None

    result = []
    for _r_idx, r in ordered_rows:
        ja = r.get("ja", "")
        utype = r.get("type", "dialogue")
        hit = _pop_unused(by_ja_type.get((ja, utype))) or _pop_unused(by_ja.get(ja))
        if hit is not None:
            c_idx, u = hit
            used_c_idxs.add(c_idx)
            result.append({**r, "start": u.start, "end": u.end})
        else:
            result.append(r)

    result.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    timing.save_timing(result, report, job.dir)

    n_corrected = sum(1 for e in report if e.get("status") == "corrected")
    n_fallback = sum(1 for e in report if e.get("status") == "fallback")
    _emit(job, "timing", "done",
          f"{n_corrected} chunk dikoreksi, {n_fallback} fallback ke timestamp asli")
    return result


def _stage_qc(job: Job, cfg, tracker, rows: list[dict],
              audio_path: Path | None,
              *, active_key_label: str | None = None) -> list[dict]:
    qc_path = job.dir / "qc" / "utterances.json"
    if qc_path.exists():
        result = qc.load_qc(job.dir)
        saved_report = _load_json_safe(job.dir / "qc" / "report.json", None, dict)
        if saved_report is not None:
            _emit(job, "qc", "done", f"{len(result)} ujaran (checkpoint)",
                  result={"report": saved_report})
        else:
            _emit(job, "qc", "done", f"{len(result)} ujaran (checkpoint)")
        return result

    deep_audio_qc = bool(job.params.get("deep_audio_qc", False))
    raw_qc_model = (job.params.get("qc_model") or "").strip()
    if job.params.get("qc_provider"):
        qc_provider = job.params["qc_provider"]
    elif job.params.get("timing_provider"):
        qc_provider = job.params["timing_provider"]
    elif job.params.get("timing_qc_provider"):
        qc_provider = job.params["timing_qc_provider"]
    elif raw_qc_model.startswith("gemini-") and "/" not in raw_qc_model:
        qc_provider = "gemini"
    elif "/" in raw_qc_model:
        qc_provider = "openrouter"
    else:
        qc_provider = job.params.get("audio_provider") or "gemini"

    qc_model = (
        raw_qc_model
        or job.params.get("timing_model")
        or (
            job.params.get("transcribe_model")
            if qc_provider == (job.params.get("audio_provider") or "gemini")
            and not providers._is_transcription_model(job.params.get("transcribe_model") or "")
            else ""
        )
        or (cfg.qc_model if qc_provider == "gemini" else "google/gemini-3.8-flash")
    )
    if qc_provider == "openrouter" and qc_model.startswith("gemini-"):
        qc_model = f"google/{qc_model}"
    elif qc_provider == "gemini" and qc_model.startswith("google/gemini-"):
        qc_model = qc_model.replace("google/", "")
    mode_label = (
        "deep" if (deep_audio_qc and audio_path is not None)
        else ("hybrid" if audio_path is not None else "text-only")
    )
    _emit(job, "qc", "start", f"QC {len(rows)} ujaran (mode={mode_label})")

    context_md, members_md = _read_context(
        job.params.get("group"), job.params.get("context_preset")
    )
    if job.params.get("context_override"):
        context_md = job.params["context_override"]
    additional_context = job.params.get("additional_context") or None

    system = qc.build_qc_system(context_md, members_md,
                                additional_context=additional_context)

    glossary = _load_json_safe(job.dir / "episode_glossary.json", {}, dict)
    timing_report = _load_json_safe(job.dir / "timing" / "report.json", [], list)
    chunk_offsets = _load_json_safe(job.dir / "offsets.json", [], list)

    sha_path = job.dir / "chunks" / "qc_input.sha256"
    digest = _qc_input_digest(
        rows, glossary, timing_report, chunk_offsets,
        cfg.sub_cps_flag, audio_path is not None,
    )
    if sha_path.exists():
        try:
            prev_digest = sha_path.read_text(encoding="utf-8").strip()
        except OSError:
            prev_digest = ""
        if prev_digest != digest:
            (job.dir / "chunks" / "qc_pass1.json").unlink(missing_ok=True)
            for p in (job.dir / "chunks").glob("qc_audio_*.json"):
                p.unlink(missing_ok=True)
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(digest, encoding="utf-8")

    translator = job.params.get("translator") or "gemini"
    text_client = providers.make_text_client(
        cfg, translator,
        model=job.params.get("translate_model") or None,
    )
    text_runner = _make_text_runner(
        job, cfg, "qc", primary_client=text_client,
        active_key_label=active_key_label,
    )

    audio_client = None
    audio_runner_3 = None
    if audio_path is not None:
        audio_client = providers.make_audio_client(
            cfg, provider=qc_provider, model=qc_model
        )
        runner4 = _make_audio_runner(
            job, cfg, "qc", qc_model, primary_client=audio_client,
            primary_provider=qc_provider,
            active_key_label=active_key_label,
        )

        def audio_runner_3(p, s, u):
            return runner4(p, s, u, None)

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

    if isinstance(report_or_issues, dict):
        summary = report_or_issues.get("summary") or {}
        report_payload = report_or_issues
    else:
        issues_list = report_or_issues if isinstance(report_or_issues, list) else []
        summary = {
            "total_rows": len(fixed_rows),
            "local_flags": 0,
            "audio_slices_checked": 0,
            "translation_fixes": sum(1 for i in issues_list if i.get("type") == "translation"),
            "ja_fixes": sum(1 for i in issues_list if i.get("type") == "ja"),
            "timing_fixes": sum(1 for i in issues_list if i.get("type") == "timing"),
            "splits": sum(1 for i in issues_list if i.get("type") == "split"),
            "unresolved_warnings": sum(
                1 for i in issues_list if i.get("type") in {"warning", "audio_flag"}
            ),
        }
        report_payload = {
            "mode": "text_only" if audio_path is None else ("deep" if deep_audio_qc else "hybrid"),
            "deep_audio_qc": deep_audio_qc,
            "summary": summary,
            "issues": issues_list,
        }
    qc.save_qc(fixed_rows, report_payload, job.dir)

    _emit(
        job, "qc", "done",
        f"{summary.get('translation_fixes', 0)} terjemahan, "
        f"{summary.get('ja_fixes', 0)} JP, "
        f"{summary.get('timing_fixes', 0)} timing, "
        f"{summary.get('splits', 0)} split diperbaiki",
        result={"report": report_payload},
    )
    return fixed_rows


def _stage_format(job: Job, cfg, rows: list[dict]) -> list[str]:
    fmt = job.params.get("output_format", "both")
    _emit(job, "format", "start", f"post-processing & render ({fmt})")
    events = [SubEvent(start=float(r["start"]), end=float(r["end"]),
                       text=r.get("id_text", ""), type=r.get("type", "dialogue"))
              for r in rows]
    processed, flags = subtitle.postprocess(
        events, min_duration=cfg.sub_min_duration,
        max_duration=cfg.sub_max_duration, merge_gap=cfg.sub_merge_gap,
        cps_threshold=cfg.sub_cps_flag)
    (job.dir / "flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    files = ["transcript_jp.json", "flags.json"]
    if fmt in ("ass", "both"):
        template = (CONTEXT_DIR / "template.ass").read_text(encoding="utf-8")
        (job.dir / "result.ass").write_text(
            subtitle.render_ass(processed, template), encoding="utf-8")
        files.append("result.ass")
    if fmt in ("srt", "both"):
        (job.dir / "result.srt").write_text(
            subtitle.render_srt(processed), encoding="utf-8")
        files.append("result.srt")
    if (job.dir / "qc" / "report.json").exists():
        files.append("qc_report.json")
    if (job.dir / "source.mp4").exists():
        files.append("source.mp4")
    _emit(job, "format", "done",
          ", ".join(f for f in files if f.startswith("result")))
    return files
