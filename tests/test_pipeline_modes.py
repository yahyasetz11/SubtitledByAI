"""Tests for Task 2 pipeline mode helpers."""
import json
from unittest.mock import MagicMock

import pytest

from app.pipeline import _load_utterances_from_source
from app.transcribe import Utterance


# ---------------------------------------------------------------------------
# 2a — _load_utterances_from_source
# ---------------------------------------------------------------------------

def _make_json_file(tmp_path, data):
    p = tmp_path / "transcript_jp.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_utterances_from_json(tmp_path):
    data = [{"id": 1, "start": 1.0, "end": 3.0, "type": "dialogue", "ja": "\u3053\u3093\u306b\u3061\u306f"}]
    p = _make_json_file(tmp_path, data)
    utts = _load_utterances_from_source("json", json_path=p)
    assert len(utts) == 1
    assert utts[0].ja == "\u3053\u3093\u306b\u3061\u306f"


def test_load_utterances_from_ass(tmp_path):
    ass = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,\u30c6\u30b9\u30c8\n"
    )
    p = tmp_path / "source.ass"
    p.write_text(ass, encoding="utf-8")
    utts = _load_utterances_from_source("ass", ass_path=p)
    assert len(utts) == 1
    assert utts[0].ja == "\u30c6\u30b9\u30c8"


def test_load_utterances_invalid_type():
    with pytest.raises(ValueError, match="source_type"):
        _load_utterances_from_source("unknown")


# ---------------------------------------------------------------------------
# 2b — _stage_format_transcript
# ---------------------------------------------------------------------------

from app.pipeline import _stage_format_transcript


def test_stage_format_transcript_json_only(tmp_path):
    job = MagicMock()
    job.dir = tmp_path
    job.id = "test-job-001"
    job.status = "running"
    job.error = None
    job.failed_stage = None
    job.params = {"output_format": "json", "save_mp4": False}
    job.events = []
    utterances = [Utterance(id=1, start=0.0, end=2.0, type="dialogue", ja="\u30c6\u30b9\u30c8")]
    (tmp_path / "transcript_jp.json").write_text(
        json.dumps([u.to_dict() for u in utterances]), encoding="utf-8"
    )
    files = _stage_format_transcript(job, utterances)
    assert "transcript_jp.json" in files
    assert not (tmp_path / "result_jp.ass").exists()


def test_stage_format_transcript_ass(tmp_path, monkeypatch):
    job = MagicMock()
    job.dir = tmp_path
    job.id = "test-job-002"
    job.status = "running"
    job.error = None
    job.failed_stage = None
    job.params = {"output_format": "ass", "save_mp4": False}
    job.events = []
    utterances = [Utterance(id=1, start=0.0, end=2.0, type="dialogue", ja="\u30c6\u30b9\u30c8")]
    (tmp_path / "transcript_jp.json").write_text("[]", encoding="utf-8")
    # Write a minimal template.ass into an isolated tmp context dir
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "template.ass").write_text(
        "[Script Info]\n[Events]\nFormat: Layer, Start, End, Style, Actor, "
        "MarginL, MarginR, MarginV, Effect, Text\n",
        encoding="utf-8",
    )
    import app.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "CONTEXT_DIR", ctx_dir)
    files = _stage_format_transcript(job, utterances)
    assert "result_jp.ass" in files
    assert (tmp_path / "result_jp.ass").exists()


# ---------------------------------------------------------------------------
# 2c — _stage_load_for_translation
# ---------------------------------------------------------------------------

def test_stage_load_for_translation_from_job_id(tmp_path, monkeypatch):
    prior_job_dir = tmp_path / "prior_job"
    prior_job_dir.mkdir()
    utts_data = [{"id": 1, "start": 0.0, "end": 2.0,
                  "type": "dialogue", "ja": "Hello"}]
    (prior_job_dir / "transcript_jp.json").write_text(
        json.dumps(utts_data), encoding="utf-8"
    )
    import app.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "OUTPUT_DIR", tmp_path)

    job = MagicMock()
    job.dir = tmp_path / "new_job"
    job.dir.mkdir()
    job.id = "new_job"
    job.status = "running"
    job.error = None
    job.failed_stage = None
    job.params = {"source_job_id": "prior_job", "mode": "translation_only"}
    job.events = []

    utts = pipeline_mod._stage_load_for_translation(job)
    assert len(utts) == 1
    assert utts[0].ja == "Hello"
    assert (job.dir / "transcript_jp.json").exists()
