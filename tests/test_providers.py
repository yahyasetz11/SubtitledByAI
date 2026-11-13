import pytest

from app.providers import Config, ConfigError


def test_config_loads_keys_and_defaults(monkeypatch):
    from app.providers import load_models_config
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TRANSCRIBE_MODEL", raising=False)
    monkeypatch.delenv("TRANSLATE_MODEL_GEMINI", raising=False)
    monkeypatch.delenv("TRANSLATE_MODEL_OPENAI", raising=False)
    monkeypatch.delenv("TRANSLATE_MODEL_ANTHROPIC", raising=False)
    monkeypatch.delenv("TIMING_MODEL", raising=False)
    monkeypatch.delenv("QC_MODEL", raising=False)

    cfg = Config.load()
    models = load_models_config()

    default_trans = (
        models["defaults"]["transcription"].get("gemini", "gemini-3.6-flash")
        if isinstance(models["defaults"]["transcription"], dict)
        else models["defaults"]["transcription"]
    )
    assert cfg.gemini_api_key == "g-key"
    assert cfg.transcribe_model == default_trans
    assert cfg.translate_models == {
        "gemini": models["defaults"]["translation"]["gemini"],
        "openai": models["defaults"]["translation"]["openai"],
        "anthropic": models["defaults"]["translation"]["anthropic"],
    }
    assert cfg.timing_model == models["defaults"]["timing"]
    assert cfg.qc_model == models["defaults"]["qc"]
    assert cfg.available_providers() == {
        "gemini": True, "openai": False, "anthropic": False,
    }
    assert cfg.sub_min_duration == pytest.approx(0.7)
    assert cfg.sub_max_duration == pytest.approx(7.5)
    assert cfg.sub_merge_gap == pytest.approx(0.5)
    assert cfg.sub_cps_flag == pytest.approx(25.0)
    assert cfg.chunk_min_seconds == pytest.approx(4 * 60)   # 240 s
    assert cfg.chunk_max_seconds == pytest.approx(6 * 60)   # 360 s


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "gemini-3.0-pro")
    monkeypatch.setenv("TRANSLATE_MODEL_GEMINI", "gemini-2.5-flash")
    monkeypatch.setenv("TRANSLATE_MODEL_OPENAI", "gpt-4.1")
    monkeypatch.setenv("TRANSLATE_MODEL_ANTHROPIC", "claude-haiku-4-5")
    monkeypatch.setenv("TIMING_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("QC_MODEL", "gemini-3.1-pro-preview")
    monkeypatch.setenv("SUB_CPS_FLAG", "30")

    cfg = Config.load()

    assert cfg.transcribe_model == "gemini-3.0-pro"
    assert cfg.translate_models == {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4.1",
        "anthropic": "claude-haiku-4-5",
    }
    assert cfg.timing_model == "gemini-2.5-flash"
    assert cfg.qc_model == "gemini-3.1-pro-preview"
    assert cfg.sub_cps_flag == pytest.approx(30.0)
    assert cfg.available_providers()["openai"] is True


def test_config_blank_model_env_falls_back_to_models_json(monkeypatch):
    from app.providers import load_models_config
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "   ")
    monkeypatch.setenv("TRANSLATE_MODEL_GEMINI", "")
    monkeypatch.setenv("TRANSLATE_MODEL_OPENAI", "  ")
    monkeypatch.setenv("TRANSLATE_MODEL_ANTHROPIC", "")
    monkeypatch.setenv("TIMING_MODEL", " ")
    monkeypatch.setenv("QC_MODEL", "")

    cfg = Config.load()
    models = load_models_config()

    default_trans = (
        models["defaults"]["transcription"].get("gemini", "gemini-3.6-flash")
        if isinstance(models["defaults"]["transcription"], dict)
        else models["defaults"]["transcription"]
    )
    assert cfg.transcribe_model == default_trans
    assert cfg.translate_models == {
        "gemini": models["defaults"]["translation"]["gemini"],
        "openai": models["defaults"]["translation"]["openai"],
        "anthropic": models["defaults"]["translation"]["anthropic"],
    }
    assert cfg.timing_model == models["defaults"]["timing"]
    assert cfg.qc_model == models["defaults"]["qc"]


def test_config_invalid_models_config_raises_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setattr("app.providers.MODELS_CONFIG_PATH", tmp_path / "missing.json")
    with pytest.raises(ConfigError):
        Config.load()

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr("app.providers.MODELS_CONFIG_PATH", bad_json)
    with pytest.raises(ConfigError):
        Config.load()

    bad_keys = tmp_path / "bad_keys.json"
    bad_keys.write_text('{"defaults": {}}', encoding="utf-8")
    monkeypatch.setattr("app.providers.MODELS_CONFIG_PATH", bad_keys)
    with pytest.raises(ConfigError):
        Config.load()

    bad_type = tmp_path / "bad_type.json"
    bad_type.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("app.providers.MODELS_CONFIG_PATH", bad_type)
    with pytest.raises(ConfigError):
        Config.load()


def test_config_chunk_minutes_converted_to_seconds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("CHUNK_MIN_MINUTES", "10")
    monkeypatch.setenv("CHUNK_MAX_MINUTES", "14")

    cfg = Config.load()

    assert cfg.chunk_min_seconds == pytest.approx(600.0)
    assert cfg.chunk_max_seconds == pytest.approx(840.0)


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


def test_load_models_config_returns_expected_keys():
    from app.providers import load_models_config
    cfg = load_models_config()
    assert "transcription" in cfg
    assert "translation" in cfg
    assert "defaults" in cfg


def test_models_config_transcription_list_nonempty():
    from app.providers import load_models_config
    cfg = load_models_config()
    assert len(cfg["transcription"]["gemini"]) > 0
    assert len(cfg["transcription"]["openrouter"]) > 0


def test_models_config_defaults_match_lists():
    from app.providers import load_models_config
    cfg = load_models_config()
    assert cfg["defaults"]["transcription"]["gemini"] in cfg["transcription"]["gemini"]
    assert cfg["defaults"]["transcription"]["openrouter"] in cfg["transcription"]["openrouter"]
    assert cfg["defaults"]["translation"]["gemini"] in cfg["translation"]["gemini"]
    assert cfg["defaults"]["translation"]["openai"] in cfg["translation"]["openai"]
    assert cfg["defaults"]["translation"]["anthropic"] in cfg["translation"]["anthropic"]
    assert cfg["defaults"]["translation"]["9router"] in cfg["translation"]["9router"]
    assert cfg["defaults"]["translation"]["openrouter"] in cfg["translation"]["openrouter"]
    assert cfg["defaults"]["timing"] in cfg["transcription"]["gemini"]
    assert cfg["defaults"]["qc"] in cfg["transcription"]["gemini"]


def test_models_config_path_exists():
    from pathlib import Path
    from app.providers import MODELS_CONFIG_PATH
    assert MODELS_CONFIG_PATH.exists()


# ---------------------------------------------------------------------------
# make_transcriber / make_translator model override tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from app.providers import make_transcriber, make_translator


def _fake_cfg():
    return Config(
        gemini_api_key="key",
        openai_api_key="okey",
        anthropic_api_key="akey",
        transcribe_model="gemini-2.5-pro",
        translate_models={"gemini": "gemini-2.5-flash", "openai": "gpt-4o", "anthropic": "claude-sonnet-4-6"},
        ytdlp_cookies_file=None,
        sub_min_duration=0.7,
        sub_max_duration=7.5,
        sub_merge_gap=0.5,
        sub_cps_flag=25,
        chunk_min_seconds=240.0,
        chunk_max_seconds=360.0,
        timing_model="gemini-2.5-flash",
        qc_window_size=10,
        qc_batch_size=20,
        qc_model="gemini-2.5-flash",
    )


def test_make_transcriber_uses_model_override():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_transcriber(cfg, model="gemini-3.1-pro-preview")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-3.1-pro-preview")


def test_make_transcriber_falls_back_to_cfg_model():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_transcriber(cfg)
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-2.5-pro")


def test_make_translator_uses_model_override():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_translator(cfg, "gemini", model="gemini-3.5-flash")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-3.5-flash")


def test_make_translator_falls_back_to_cfg_model():
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_translator(cfg, "gemini")
        MockClient.assert_called_once_with(cfg.gemini_api_key, "gemini-2.5-flash")


from app.providers import (
    make_audio_client,
    make_text_client,
    resolve_gemini_key,
)


def test_resolve_gemini_key_uses_label_without_mutating_global(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_WORK", "g-work")
    assert get_active_gemini_label() == "Default"
    assert resolve_gemini_key("WORK") == "g-work"
    assert get_active_gemini_label() == "Default"


def test_resolve_gemini_key_unknown_label_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    with pytest.raises(ConfigError, match="NOPE"):
        resolve_gemini_key("NOPE")


def test_resolve_gemini_key_none_and_empty_with_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for label in (None, "", "   "):
        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            resolve_gemini_key(label, fallback_key="")


def test_resolve_gemini_key_none_and_empty_use_fallback_or_active_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    assert resolve_gemini_key(None, fallback_key="  g-fb  ") == "g-fb"
    assert resolve_gemini_key("", fallback_key="") == "g-default"
    assert resolve_gemini_key("   ", fallback_key="   ") == "g-default"


def test_resolve_gemini_key_default_label_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY belum diisi di .env"):
        resolve_gemini_key("Default")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY belum diisi di .env"):
        resolve_gemini_key(" Default ")


def test_make_audio_client_uses_specific_key_label(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_audio_client(cfg, model="gemini-3.5-flash", key_label="ALT")
        MockClient.assert_called_once_with("g-alt", "gemini-3.5-flash")
    assert get_active_gemini_label() == "Default"


def test_make_audio_client_blank_key_label_delegates_and_falls_back(monkeypatch):
    cfg = _fake_cfg()
    for blank in ("", "   ", None):
        with patch("app.providers.make_transcriber", wraps=make_transcriber) as spy_transcriber, \
             patch("app.providers.GeminiClient") as MockClient:
            make_audio_client(cfg, key_label=blank)
            spy_transcriber.assert_called_once_with(cfg, model=None)
            MockClient.assert_called_once_with(cfg.gemini_api_key, cfg.transcribe_model)

    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    with patch("app.providers.GeminiClient") as MockClient:
        make_audio_client(cfg, model=None, key_label="ALT")
        MockClient.assert_called_once_with("g-alt", cfg.transcribe_model)


def test_make_text_client_uses_specific_gemini_key_label(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-default")
    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    cfg = _fake_cfg()
    with patch("app.providers.GeminiClient") as MockClient:
        make_text_client(cfg, "gemini", model="gemini-3.6-flash", key_label="ALT")
        MockClient.assert_called_once_with("g-alt", "gemini-3.6-flash")


def test_make_text_client_blank_key_label_delegates_and_falls_back(monkeypatch):
    cfg = _fake_cfg()
    for blank in ("", "   ", None):
        with patch("app.providers.make_translator", wraps=make_translator) as spy_translator, \
             patch("app.providers.GeminiClient") as MockClient:
            make_text_client(cfg, "gemini", key_label=blank)
            spy_translator.assert_called_once_with(cfg, "gemini", model=None)
            MockClient.assert_called_once_with(
                cfg.gemini_api_key, cfg.translate_models["gemini"]
            )

    monkeypatch.setenv("GEMINI_API_KEY_ALT", "g-alt")
    with patch("app.providers.GeminiClient") as MockClient:
        make_text_client(cfg, "gemini", model=None, key_label="ALT")
        MockClient.assert_called_once_with("g-alt", cfg.translate_models["gemini"])


def test_make_text_client_delegates_openai_and_anthropic():
    cfg = _fake_cfg()
    with patch("app.providers.OpenAIClient") as MockOAI:
        make_text_client(cfg, "openai", model="gpt-4.1")
        MockOAI.assert_called_once_with("okey", "gpt-4.1")
    with patch("app.providers.AnthropicClient") as MockAnt:
        make_text_client(cfg, "anthropic", model="claude-haiku-4-5")
        MockAnt.assert_called_once_with("akey", "claude-haiku-4-5")


from app.providers import (
    AudioFallbackSlot,
    TranslateFallbackSlot,
    execute_audio_with_fallback,
    execute_with_fallback,
)


def test_execute_with_fallback_steps_slots_on_503():
    attempts = []
    fallbacks_logged = []

    class FakeTextClient:
        def __init__(self, label, exc=None):
            self.model = label
            self._exc = exc

        def generate(self, system, user, audio=None):
            attempts.append(self.model)
            if self._exc:
                raise self._exc
            return LLMResponse(text='[{"id": 1, "id_text": "ok"}]', input_tokens=10, output_tokens=5)

    clients = {
        ("gemini", "Default", "primary-m"): FakeTextClient("primary-m", FakeHTTPError(503)),
        ("gemini", "ALT", "slot1-m"): FakeTextClient("slot1-m", FakeHTTPError(429)),
        ("openai", None, "gpt-4o"): FakeTextClient("gpt-4o", None),
    }

    primary = TranslateFallbackSlot(provider="gemini", key_label="Default", model="primary-m")
    slots = [
        TranslateFallbackSlot(provider="gemini", key_label="ALT", model="slot1-m"),
        TranslateFallbackSlot(provider="openai", key_label=None, model="gpt-4o"),
    ]

    resp, used_client = execute_with_fallback(
        primary=primary,
        fallback_slots=slots,
        build_client=lambda s: clients[(s.provider, s.key_label, s.model)],
        call_fn=lambda c: c.generate("sys", "usr"),
        on_fallback=lambda idx, slot, exc: fallbacks_logged.append((idx, slot.model, exc.status_code)),
        sleep=lambda _: None,
    )
    assert used_client.model == "gpt-4o"
    assert "ok" in resp.text
    assert attempts == ["primary-m", "primary-m", "slot1-m", "slot1-m", "gpt-4o"]
    assert fallbacks_logged == [(1, "slot1-m", 503), (2, "gpt-4o", 429)]


def test_execute_with_fallback_steps_on_403_but_stops_on_400():
    attempts = []

    class FakeClient:
        def __init__(self, model, status):
            self.model = model
            self._status = status

        def generate(self, system, user, audio=None):
            attempts.append(self.model)
            raise FakeHTTPError(self._status)

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="ALT", model="m1")

    with pytest.raises(FakeHTTPError) as exc_info:
        execute_with_fallback(
            primary=primary,
            fallback_slots=[slot1],
            build_client=lambda s: FakeClient(s.model, 403 if s.model == "m0" else 400),
            call_fn=lambda c: c.generate("s", "u"),
            sleep=lambda _: None,
        )
    assert exc_info.value.status_code == 400
    assert attempts == ["m0", "m1"]


def test_execute_with_fallback_caps_at_three_slots_and_raises_last_error():
    built = []

    class FakeClient:
        def __init__(self, model):
            self.model = model

        def generate(self, system, user, audio=None):
            raise FakeHTTPError(502 if self.model == "m3" else 503)

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slots = [
        AudioFallbackSlot(key_label="K1", model="m1"),
        AudioFallbackSlot(key_label="K2", model="m2"),
        AudioFallbackSlot(key_label="K3", model="m3"),
        AudioFallbackSlot(key_label="K4", model="m4"),
    ]

    def build(s):
        built.append(s.model)
        return FakeClient(s.model)

    with pytest.raises(FakeHTTPError) as exc_info:
        execute_with_fallback(
            primary=primary,
            fallback_slots=slots,
            build_client=build,
            call_fn=lambda c: c.generate("s", "u"),
            sleep=lambda _: None,
        )
    assert exc_info.value.status_code == 502
    assert built == ["m0", "m1", "m2", "m3"]


def test_execute_with_fallback_stops_immediately_on_primary_400_and_retries_network_error():
    built = []

    class BadClient:
        model = "m0"

        def generate(self, system, user, audio=None):
            raise FakeHTTPError(400)

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="ALT", model="m1")

    with pytest.raises(FakeHTTPError):
        execute_with_fallback(
            primary=primary,
            fallback_slots=[slot1],
            build_client=lambda s: (built.append(s.model), BadClient())[1],
            call_fn=lambda c: c.generate("s", "u"),
            sleep=lambda _: None,
        )
    assert built == ["m0"]

    attempts = []

    class NetClient:
        def __init__(self, model, fail_with=None):
            self.model = model
            self._fail_with = fail_with

        def generate(self, system, user, audio=None):
            attempts.append(self.model)
            if self._fail_with:
                raise self._fail_with
            return LLMResponse(text="ok", input_tokens=1, output_tokens=1)

    resp, winner = execute_with_fallback(
        primary=primary,
        fallback_slots=[slot1],
        build_client=lambda s: NetClient(s.model, ConnectionError("reset") if s.model == "m0" else None),
        call_fn=lambda c: c.generate("s", "u"),
        sleep=lambda _: None,
    )
    assert resp.text == "ok"
    assert winner.model == "m1"
    assert attempts == ["m0", "m0", "m1"]


def test_execute_audio_with_fallback_reuploads_only_when_key_changes(tmp_path):
    uploads = []
    calls = []
    upload_done_count = 0
    chunk = tmp_path / "chunk_001.mp3"
    chunk.write_bytes(b"mp3")

    class FakeAudioClient:
        def __init__(self, api_key, model):
            self.api_key = api_key
            self.model = model

        def upload_audio(self, path):
            uploads.append((self.api_key, path))
            return f"ref:{self.api_key}"

        def generate(self, system, user, audio=None):
            calls.append((self.api_key, self.model, audio))
            if self.model == "m-primary":
                raise FakeHTTPError(503)
            if self.model == "m-slot1":
                raise FakeHTTPError(503)
            return LLMResponse(text="ok", input_tokens=5, output_tokens=3)

    primary = AudioFallbackSlot(key_label="Default", model="m-primary")
    slots = [
        AudioFallbackSlot(key_label="Default", model="m-slot1"),  # same key -> reuse upload
        AudioFallbackSlot(key_label="ALT", model="m-slot2"),      # new key -> re-upload
    ]
    key_map = {"Default": "k1", "ALT": "k2"}

    def on_up():
        nonlocal upload_done_count
        upload_done_count += 1

    resp, used_client = execute_audio_with_fallback(
        chunk_path=chunk,
        primary=primary,
        fallback_slots=slots,
        build_client=lambda s: FakeAudioClient(key_map[s.key_label], s.model),
        resolve_key=lambda s: key_map[s.key_label],
        call_fn=lambda c, ref: c.generate("sys", "usr", audio=ref),
        on_upload_done=on_up,
        sleep=lambda _: None,
    )
    assert used_client.model == "m-slot2"
    assert resp.text == "ok"
    assert uploads == [("k1", chunk), ("k2", chunk)]
    assert upload_done_count == 2
    assert calls[0][2] == "ref:k1"
    assert calls[2][2] == "ref:k1"  # slot 1 reused k1 upload
    assert calls[4][2] == "ref:k2"  # slot 2 uploaded under k2


def test_execute_audio_with_fallback_retries_upload_failure_and_steps_on_403(tmp_path):
    uploads = []
    chunk = tmp_path / "chunk_002.mp3"
    chunk.write_bytes(b"mp3")

    class FakeAudioClient:
        def __init__(self, api_key, model):
            self.api_key = api_key
            self.model = model

        def upload_audio(self, path):
            uploads.append((self.api_key, self.model))
            if self.api_key == "k1":
                raise FakeHTTPError(403)
            return f"ref:{self.api_key}"

        def generate(self, system, user, audio=None):
            return LLMResponse(text=f"done:{audio}", input_tokens=2, output_tokens=1)

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slots = [AudioFallbackSlot(key_label="ALT", model="m1")]
    key_map = {"Default": "k1", "ALT": "k2"}

    resp, winner = execute_audio_with_fallback(
        chunk_path=chunk,
        primary=primary,
        fallback_slots=slots,
        build_client=lambda s: FakeAudioClient(key_map[s.key_label], s.model),
        resolve_key=lambda s: key_map[s.key_label],
        call_fn=lambda c, ref: c.generate("sys", "usr", audio=ref),
        sleep=lambda _: None,
    )
    assert winner.model == "m1"
    assert resp.text == "done:ref:k2"
    assert uploads == [("k1", "m0"), ("k2", "m1")]


def test_execute_audio_with_fallback_retries_transient_upload_503_and_validates_attempts(tmp_path):
    chunk = tmp_path / "chunk_003.mp3"
    chunk.write_bytes(b"mp3")
    upload_attempts = 0
    upload_done = 0

    class FlakyUploadClient:
        model = "m0"

        def upload_audio(self, path):
            nonlocal upload_attempts
            upload_attempts += 1
            if upload_attempts == 1:
                raise FakeHTTPError(503)
            return "ref:k1"

        def generate(self, system, user, audio=None):
            return LLMResponse(text=f"ok:{audio}", input_tokens=1, output_tokens=1)

    def on_up():
        nonlocal upload_done
        upload_done += 1

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    resp, _ = execute_audio_with_fallback(
        chunk_path=chunk,
        primary=primary,
        fallback_slots=[],
        build_client=lambda s: FlakyUploadClient(),
        resolve_key=lambda s: "k1",
        call_fn=lambda c, ref: c.generate("s", "u", audio=ref),
        on_upload_done=on_up,
        sleep=lambda _: None,
    )
    assert resp.text == "ok:ref:k1"
    assert upload_attempts == 2
    assert upload_done == 1

    with pytest.raises(ValueError, match="attempts_per_slot"):
        execute_with_fallback(
            primary=primary,
            fallback_slots=[],
            build_client=lambda s: FlakyUploadClient(),
            call_fn=lambda c: c.generate("s", "u"),
            attempts_per_slot=0,
            sleep=lambda _: None,
        )


@pytest.mark.parametrize("fallback_slots", [[], None])
def test_execute_with_fallback_empty_or_none_slots_exhausts_primary_and_reraises(fallback_slots):
    attempts = []
    fallbacks = []

    class FailingClient:
        model = "m0"

        def generate(self, system, user, audio=None):
            attempts.append(self.model)
            raise FakeHTTPError(503)

    primary = AudioFallbackSlot(key_label="Default", model="m0")

    with pytest.raises(FakeHTTPError) as exc_info:
        execute_with_fallback(
            primary=primary,
            fallback_slots=fallback_slots,
            build_client=lambda s: FailingClient(),
            call_fn=lambda c: c.generate("s", "u"),
            attempts_per_slot=3,
            on_fallback=lambda idx, slot, exc: fallbacks.append((idx, slot.model, exc)),
            sleep=lambda _: None,
        )
    assert exc_info.value.status_code == 503
    assert attempts == ["m0", "m0", "m0"]
    assert fallbacks == []


@pytest.mark.parametrize(
    "build_exc",
    [FakeHTTPError(503), FakeHTTPError(403), ConnectionError("dns failed")],
)
def test_execute_with_fallback_steps_to_slot1_when_build_client_raises_trigger(build_exc):
    built = []
    fallbacks = []

    class OkClient:
        model = "m1"

        def generate(self, system, user, audio=None):
            return LLMResponse(text="ok", input_tokens=1, output_tokens=1)

    def build(slot):
        built.append(slot.model)
        if slot.model == "m0":
            raise build_exc
        return OkClient()

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="ALT", model="m1")

    resp, winner = execute_with_fallback(
        primary=primary,
        fallback_slots=[slot1],
        build_client=build,
        call_fn=lambda c: c.generate("s", "u"),
        on_fallback=lambda idx, slot, exc: fallbacks.append((idx, slot.model, exc)),
        sleep=lambda _: None,
    )
    assert resp.text == "ok"
    assert winner.model == "m1"
    assert built == ["m0", "m1"]
    assert fallbacks == [(1, "m1", build_exc)]


@pytest.mark.parametrize("build_exc", [ConfigError("bad key"), FakeHTTPError(400)])
def test_execute_with_fallback_fails_immediately_when_build_client_raises_non_trigger(build_exc):
    built = []
    fallbacks = []

    def build(slot):
        built.append(slot.model)
        raise build_exc

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="ALT", model="m1")

    with pytest.raises(type(build_exc)) as exc_info:
        execute_with_fallback(
            primary=primary,
            fallback_slots=[slot1],
            build_client=build,
            call_fn=lambda c: c.generate("s", "u"),
            on_fallback=lambda idx, slot, exc: fallbacks.append((idx, slot.model, exc)),
            sleep=lambda _: None,
        )
    assert exc_info.value is build_exc
    assert built == ["m0"]
    assert fallbacks == []


def test_execute_audio_with_fallback_raises_when_upload_returns_none(tmp_path):
    chunk = tmp_path / "chunk_004.mp3"
    chunk.write_bytes(b"mp3")
    built = []
    upload_calls = 0
    upload_done = 0
    generate_calls = 0

    class NullUploadClient:
        def __init__(self, model):
            self.model = model

        def upload_audio(self, path):
            nonlocal upload_calls
            upload_calls += 1
            return None

        def generate(self, system, user, audio=None):
            nonlocal generate_calls
            generate_calls += 1
            return LLMResponse(text="unexpected", input_tokens=1, output_tokens=1)

    def build(slot):
        built.append(slot.model)
        return NullUploadClient(slot.model)

    def on_up():
        nonlocal upload_done
        upload_done += 1

    primary = AudioFallbackSlot(key_label="Default", model="m0")
    slot1 = AudioFallbackSlot(key_label="Default", model="m1")

    with pytest.raises(RuntimeError, match=r"None.*chunk_004\.mp3"):
        execute_audio_with_fallback(
            chunk_path=chunk,
            primary=primary,
            fallback_slots=[slot1],
            build_client=build,
            resolve_key=lambda s: "k1",
            call_fn=lambda c, ref: c.generate("s", "u", audio=ref),
            on_upload_done=on_up,
            sleep=lambda _: None,
        )
    assert upload_calls == 1
    assert upload_done == 0
    assert generate_calls == 0
    assert built == ["m0"]


def test_models_config_phase2_structure():
    from app.providers import load_models_config
    cfg = load_models_config()
    assert isinstance(cfg["transcription"], dict)
    assert "gemini" in cfg["transcription"]
    assert "openrouter" in cfg["transcription"]
    assert "qwen/qwen3.8-omni-flash" in cfg["transcription"]["openrouter"]

    assert isinstance(cfg["translation"], dict)
    assert "9router" in cfg["translation"]
    assert "openrouter" in cfg["translation"]
    assert "ag/gemini-3.8-flash-high" in cfg["translation"]["9router"]

    assert cfg["defaults"]["audio_provider"] == "gemini"
    assert cfg["defaults"]["text_provider"] == "9router"
    assert cfg["defaults"]["transcription"]["gemini"] in cfg["transcription"]["gemini"]
    assert cfg["defaults"]["transcription"]["openrouter"] in cfg["transcription"]["openrouter"]
    assert cfg["defaults"]["translation"]["9router"] in cfg["translation"]["9router"]


def test_pricing_includes_new_gemini_and_qwen_models():
    from app.providers import PRICING, load_models_config
    assert "gemini-3.8-flash" in PRICING
    assert "qwen/qwen3.8-omni-flash" in PRICING

    models_cfg = load_models_config()
    for section in ("transcription", "translation"):
        for provider, model_list in models_cfg.get(section, {}).items():
            if provider == "9router":
                continue
            for model in model_list:
                assert model in PRICING, f"Model '{model}' from {section}.{provider} missing from PRICING"


def test_transcription_dict_indexing_slicing_and_containment():
    from app.providers import TranscriptionDict

    td = TranscriptionDict({
        "gemini": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"],
        "openrouter": ["qwen/qwen3.8-omni-flash", "openai/gpt-audio-mini"],
    })

    # Integer indexing (backwards compatibility defaults to gemini list)
    assert td[0] == "gemini-3.8-flash"
    assert td[1] == "gemini-3.6-flash"
    assert td[-1] == "gemini-2.5-flash"

    # Slicing support
    assert td[:2] == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert td[1:] == ["gemini-3.6-flash", "gemini-2.5-flash"]

    # String containment: provider keys
    assert "gemini" in td
    assert "openrouter" in td

    # String containment: model names across providers
    assert "gemini-3.6-flash" in td
    assert "qwen/qwen3.8-omni-flash" in td
    assert "openai/gpt-audio-mini" in td
    assert "nonexistent-model" not in td

    # Dict containment (unhashable types should not crash with TypeError)
    assert {"gemini": "gemini-3.6-flash"} in td
    assert {"openrouter": "qwen/qwen3.8-omni-flash"} in td
    assert {
        "gemini": "gemini-3.6-flash",
        "openrouter": "qwen/qwen3.8-omni-flash",
    } in td

    # Dict containment: exact match with sublist
    assert {"gemini": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"]} in td

    # Dict containment negative cases
    assert {"gemini": "unknown-model"} not in td
    assert {"unknown_provider": "gemini-3.6-flash"} not in td
    assert {
        "gemini": "gemini-3.6-flash",
        "openrouter": "unknown-model",
    } not in td
    assert {} not in td


def test_transcription_default_dict_equality_and_str():
    from app.providers import TranscriptionDefaultDict

    tdd = TranscriptionDefaultDict({
        "gemini": "gemini-3.6-flash",
        "openrouter": "qwen/qwen3.8-omni-flash",
    })

    # String equality matches gemini or openrouter default
    assert tdd == "gemini-3.6-flash"
    assert tdd == "qwen/qwen3.8-omni-flash"
    assert "gemini-3.6-flash" == tdd
    assert "qwen/qwen3.8-omni-flash" == tdd

    # Misleading __eq__ == "gemini" is removed
    assert tdd != "gemini"
    assert "gemini" != tdd
    assert tdd != "unknown-model"

    # Dict equality
    assert tdd == {
        "gemini": "gemini-3.6-flash",
        "openrouter": "qwen/qwen3.8-omni-flash",
    }
    assert tdd != {"gemini": "different-model"}

    # __str__ returns gemini default
    assert str(tdd) == "gemini-3.6-flash"

    # Unhashable: __hash__ removed
    with pytest.raises(TypeError):
        hash(tdd)


