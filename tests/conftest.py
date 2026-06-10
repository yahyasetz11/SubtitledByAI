import os

# Set BEFORE app modules import, so startup validation in tests never trips
# on a missing key. Individual tests override with monkeypatch.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
