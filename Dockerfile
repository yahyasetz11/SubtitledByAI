# ── Stage: base image ────────────────────────────────────────────────────────
# We start from an official Python image.
# "slim" means it's a smaller version — no extra tools pre-installed.
# Pinning to 3.12 ensures this image behaves the same on every machine, forever.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# apt-get is the Linux package manager (like pip, but for system tools).
# ffmpeg is required for audio normalization and chunking.
# We clean up the apt cache afterwards to keep the image size small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# This is the "home" folder of our app inside Docker.
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# We copy requirements.txt first (before copying the rest of the code).
# Why? Docker caches each step. If requirements.txt hasn't changed,
# Docker reuses the cached pip install — rebuilds are much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source code ───────────────────────────────────────────────────────────
# Copy the app code and context files into the image.
# context/ is included so the image works standalone (volumes will override
# these when you run locally via docker-compose).
COPY app/       app/
COPY static/    static/
COPY context/   context/

# ── Port declaration ──────────────────────────────────────────────────────────
# Tell Docker this container listens on port 8000.
# This is documentation only — the actual port mapping happens in docker-compose.
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# This runs when you do "docker compose up".
# --host 0.0.0.0 means "accept connections from outside the container"
# (without this, the server only listens inside the container and you can't reach it).
# No --reload here: that's for development. In Docker we want a stable server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
