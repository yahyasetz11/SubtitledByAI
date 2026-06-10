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
