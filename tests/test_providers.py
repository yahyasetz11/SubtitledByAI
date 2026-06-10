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
