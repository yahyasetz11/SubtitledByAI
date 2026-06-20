"""Job orchestration: stages, checkpoints, SSE-ready events (§1, §9 layer 3).

Every stage skips itself when its artifact already exists in output/{job_id}/,
so run_job() after a failure resumes instead of redoing work. Transcription
additionally checkpoints per chunk and translation per batch.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app import audio, providers, subtitle, transcribe, translate
from app.providers import UsageTracker
from app.subtitle import SubEvent

OUTPUT_DIR = Path("output")
CONTEXT_DIR = Path("context")
JOBS: dict[str, "Job"] = {}


class PipelineError(Exception):
    """Job cannot proceed (bad params, missing source file, ...)."""


@dataclass
class Job:
    id: str
    dir: Path
    params: dict
    status: str = "queued"  # queued | running | done | failed
    error: str | None = None
    events: list[dict] = field(default_factory=list)


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
              events=meta.get("events", []))
    JOBS[job.id] = job
    return job


def _save(job: Job) -> None:
    payload = {"id": job.id, "params": job.params, "status": job.status,
               "error": job.error, "events": job.events}
    (job.dir / "job.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit(job: Job, stage: str, status: str, message: str | None = None,
          result: dict | None = None) -> None:
    event: dict = {"stage": stage, "status": status}
    if message:
        event["message"] = message
    if result is not None:
        event["result"] = result
    job.events.append(event)
    _save(job)


def run_job(job_id: str) -> None:
    job = JOBS.get(job_id) or load_job(job_id)
    if job is None:
        raise KeyError(f"Job tidak ditemukan: {job_id}")
    cfg = providers.Config.load()
    tracker = UsageTracker(job.dir / "usage.json")
    job.status, job.error = "running", None
    _save(job)
    stage = "init"
    try:
        stage = "download"
        source = _stage_download(job, cfg)
        stage = "normalize"
        audio_path = _stage_normalize(job, source)
        stage = "chunk"
        chunks = _stage_chunk(job, audio_path)
        stage = "transcribe"
        utterances = _stage_transcribe(job, cfg, tracker, chunks)
        stage = "translate"
        rows = _stage_translate(job, cfg, tracker, utterances)
        stage = "format"
        files = _stage_format(job, cfg, rows)
        job.status = "done"
        _emit(job, "done", "done",
              result={"usage": tracker.summary(), "files": files})
    except Exception as exc:  # noqa: BLE001 — any stage error fails the job
        job.status = "failed"
        job.error = str(exc)
        _emit(job, stage, "error", message=str(exc))


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


def _stage_chunk(job: Job, audio_path: Path) -> list[audio.PlannedChunk]:
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

        chunks = audio.plan_chunks(duration, get_silences)
        audio.write_offsets(chunks, offsets_path)
        _emit(job, "chunk", "done", f"{len(chunks)} chunk")
    for c in chunks:
        dst = chunks_dir / f"chunk_{c.index:03d}.mp3"
        if not dst.exists():
            audio.cut_chunk(audio_path, dst, c.start, c.end)
    return chunks


def _stage_transcribe(job: Job, cfg, tracker,
                      chunks: list[audio.PlannedChunk]
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
    gemini = providers.make_transcriber(cfg)
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
                    on_upload_done=_on_upload_done)
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


def _stage_translate(job: Job, cfg, tracker,
                     utterances: list[transcribe.Utterance]) -> list[dict]:
    out_path = job.dir / "translated_id.json"
    if out_path.exists():
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        _emit(job, "translate", "done", f"{len(rows)} baris (checkpoint)")
        return rows
    context_md, members_md = _read_context(
        job.params.get("group"), job.params.get("context_preset")
    )
    if job.params.get("context_override"):
        context_md = job.params["context_override"]
    additional_context = job.params.get("additional_context") or None
    system = translate.build_translate_system(context_md, members_md,
                                              additional_context=additional_context)
    client = providers.make_translator(cfg, job.params["translator"])
    _emit(job, "translate", "start",
          f"terjemahkan {len(utterances)} ujaran via "
          f"{job.params['translator']}")

    def load_ckpt(batch_index: int) -> dict[int, str] | None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        if not p.exists():
            return None
        return {int(k): v
                for k, v in json.loads(p.read_text(encoding="utf-8")).items()}

    def save_ckpt(batch_index: int, mapping: dict[int, str]) -> None:
        p = job.dir / "chunks" / f"translated_{batch_index:03d}.json"
        p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2),
                     encoding="utf-8")

    def on_batch(done: int, total: int) -> None:
        _emit(job, "translate", "progress", f"batch {done}/{total}")

    mapping = translate.translate_all(client, utterances, system, tracker,
                                      on_batch=on_batch,
                                      load_checkpoint=load_ckpt,
                                      save_checkpoint=save_ckpt)
    rows = [{**u.to_dict(), "id_text": mapping[u.id]} for u in utterances]
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    _emit(job, "translate", "done", f"{len(rows)} baris")
    return rows


def _stage_format(job: Job, cfg, rows: list[dict]) -> list[str]:
    fmt = job.params.get("output_format", "both")
    _emit(job, "format", "start", f"post-processing & render ({fmt})")
    events = [SubEvent(start=float(r["start"]), end=float(r["end"]),
                       text=r["id_text"], type=r.get("type", "dialogue"))
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
    if (job.dir / "source.mp4").exists():
        files.append("source.mp4")
    _emit(job, "format", "done",
          ", ".join(f for f in files if f.startswith("result")))
    return files
