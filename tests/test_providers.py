import pytest

from app.providers import Config, ConfigError


def test_config_loads_keys_and_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_MODEL", raising=False)

    cfg = Config.load()

    assert cfg.gemini_api_key == "g-key"
    assert cfg.transcribe_model == "gemini-2.5-pro"
    assert cfg.translate_models == {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
    }
    assert cfg.available_providers() == {
        "gemini": True, "openai": False, "anthropic": False,
    }
    assert cfg.sub_min_duration == pytest.approx(0.7)
    assert cfg.sub_max_duration == pytest.approx(7.5)
    assert cfg.sub_merge_gap == pytest.approx(0.5)
    assert cfg.sub_cps_flag == pytest.approx(25.0)


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "gemini-3.0-pro")
    monkeypatch.setenv("SUB_CPS_FLAG", "30")

    cfg = Config.load()

    assert cfg.transcribe_model == "gemini-3.0-pro"
    assert cfg.sub_cps_flag == pytest.approx(30.0)
    assert cfg.available_providers()["openai"] is True


def test_config_ignores_comment_as_cookies_value(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("YTDLP_COOKIES_FILE",
                       "# opsional, untuk video region-locked/membership")
    cfg = Config.load()
    assert cfg.ytdlp_cookies_file is None


def test_config_missing_gemini_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        Config.load()


from app.providers import call_with_retries


class FakeHTTPError(Exception):
    """Mimics provider SDK errors: .status_code + .response.headers."""

    def __init__(self, status_code, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code

        class _Resp:
            headers = {"retry-after": retry_after} if retry_after else {}
        self.response = _Resp()


def test_retry_succeeds_after_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeHTTPError(503)
        return "ok"

    assert call_with_retries(flaky, sleep=lambda s: None) == "ok"
    assert calls["n"] == 3


def test_retry_respects_retry_after_header():
    sleeps = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeHTTPError(429, retry_after="7")
        return "ok"

    assert call_with_retries(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [7.0]


def test_retry_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_500():
        calls["n"] += 1
        raise FakeHTTPError(500)

    with pytest.raises(FakeHTTPError):
        call_with_retries(always_500, sleep=lambda s: None)
    assert calls["n"] == 4  # 1 attempt + 3 retries


def test_non_retryable_error_raises_immediately():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise FakeHTTPError(400)

    with pytest.raises(FakeHTTPError):
        call_with_retries(bad_request, sleep=lambda s: None)
    assert calls["n"] == 1


def test_network_errors_are_retryable():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return "ok"

    assert call_with_retries(flaky, sleep=lambda s: None) == "ok"


import json

from app.providers import UsageTracker


def test_usage_accumulates_and_persists(tmp_path):
    path = tmp_path / "usage.json"
    tracker = UsageTracker(path)
    tracker.add("gemini-2.5-pro", input_tokens=40_000, output_tokens=10_000)
    tracker.add("gemini-2.5-pro", input_tokens=12_000, output_tokens=2_000)
    tracker.add("gemini-2.5-flash", input_tokens=8_000, output_tokens=3_000)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["models"]["gemini-2.5-pro"]["input_tokens"] == 52_000
    assert data["models"]["gemini-2.5-pro"]["output_tokens"] == 12_000

    summary = tracker.summary()
    assert summary["input_tokens"] == 60_000
    assert summary["output_tokens"] == 15_000
    # pro: 52k*1.25/1M + 12k*10/1M = 0.065 + 0.12 = 0.185
    # flash: 8k*0.30/1M + 3k*2.50/1M = 0.0024 + 0.0075 = 0.0099
    assert summary["cost_usd"] == pytest.approx(0.1949, abs=1e-4)
    assert "60.0K in / 15.0K out" in summary["line"]
    assert "$0.19" in summary["line"]


def test_usage_reloads_existing_file(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(path).add("gpt-4o", 1000, 500)
    tracker2 = UsageTracker(path)  # e.g. after a checkpoint retry
    tracker2.add("gpt-4o", 1000, 500)
    assert tracker2.summary()["input_tokens"] == 2000


def test_usage_unknown_model_costs_zero(tmp_path):
    tracker = UsageTracker(tmp_path / "usage.json")
    tracker.add("some-future-model", 1_000_000, 1_000_000)
    summary = tracker.summary()
    assert summary["cost_usd"] == 0.0
    assert summary["unpriced_models"] == ["some-future-model"]


from types import SimpleNamespace

from app.providers import (
    LLMResponse, _from_anthropic, _from_gemini, _from_openai, make_translator,
)


def _gemini_resp(text="hi", finish="STOP"):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20,
        ),
        candidates=[SimpleNamespace(
            finish_reason=SimpleNamespace(name=finish),
        )],
    )


def test_from_gemini_maps_fields():
    resp = _from_gemini(_gemini_resp())
    assert resp == LLMResponse(text="hi", input_tokens=100, output_tokens=20,
                               truncated=False)


def test_from_gemini_detects_truncation():
    assert _from_gemini(_gemini_resp(finish="MAX_TOKENS")).truncated is True


def test_from_openai_maps_fields():
    raw = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="halo"), finish_reason="length",
        )],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10),
    )
    resp = _from_openai(raw)
    assert resp == LLMResponse(text="halo", input_tokens=50, output_tokens=10,
                               truncated=True)


def test_from_anthropic_maps_fields():
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ya")],
        usage=SimpleNamespace(input_tokens=30, output_tokens=5),
        stop_reason="end_turn",
    )
    resp = _from_anthropic(raw)
    assert resp == LLMResponse(text="ya", input_tokens=30, output_tokens=5,
                               truncated=False)


def test_make_translator_requires_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config.load()
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_translator(cfg, "openai")


def test_make_translator_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = Config.load()
    with pytest.raises(ConfigError):
        make_translator(cfg, "deepseek")


# ---------------------------------------------------------------------------
# Multi-key / active-key feature
# ---------------------------------------------------------------------------

from app.providers import get_all_gemini_keys, get_active_gemini_label, set_active_gemini_key


def test_get_all_gemini_keys_default_only(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.delenv("GEMINI_API_KEY_PERSONAL", raising=False)
    assert get_all_gemini_keys() == {"Default": "g-default"}


def test_get_all_gemini_keys_includes_named_labels(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_PERSONAL", "g-personal")
    monkeypatch.setenv("GEMINI_API_KEY_WORK", "g-work")
    keys = get_all_gemini_keys()
    assert keys == {"Default": "g-default", "PERSONAL": "g-personal", "WORK": "g-work"}


def test_get_all_gemini_keys_skips_empty_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_EMPTY", "")
    keys = get_all_gemini_keys()
    assert "EMPTY" not in keys


def test_get_active_gemini_label_is_default_on_startup():
    assert get_active_gemini_label() == "Default"


def test_set_active_gemini_key_changes_config_load_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_PERSONAL", "g-personal")
    set_active_gemini_key("PERSONAL")
    assert get_active_gemini_label() == "PERSONAL"
    cfg = Config.load()
    assert cfg.gemini_api_key == "g-personal"


def test_set_active_gemini_key_invalid_label_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.delenv("GEMINI_API_KEY_GHOST", raising=False)
    with pytest.raises(ConfigError, match="GHOST"):
        set_active_gemini_key("GHOST")


def test_set_active_gemini_key_to_default_restores_base_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_PERSONAL", "g-personal")
    set_active_gemini_key("PERSONAL")
    set_active_gemini_key("Default")
    cfg = Config.load()
    assert cfg.gemini_api_key == "g-default"
