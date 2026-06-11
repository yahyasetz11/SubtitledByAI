import os

import pytest

# Set BEFORE app modules import, so startup validation in tests never trips
# on a missing key. Individual tests override with monkeypatch.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch):
    # Config.load() reads the developer's real .env via load_dotenv, which
    # re-populates vars the tests delete; keep tests independent of .env.
    monkeypatch.setattr("app.providers.load_dotenv", lambda *a, **k: None)
