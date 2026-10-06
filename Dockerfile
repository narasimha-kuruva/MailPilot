# MailPilot API image. Secrets are never baked in: pass configuration with
# --env-file at run time and mount the Gmail OAuth token and the data
# directory. See README, "Docker".

# --- Build stage: install the package and its dependencies into a venv ---
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# README.md is part of the package metadata (pyproject's `readme`).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# --- Runtime stage: the venv only, run as an unprivileged user ---
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 mailpilot
COPY --from=builder /opt/venv /opt/venv

# Relative paths in the configuration (./secrets/token.json, ./data/chroma)
# resolve here; mount the real secrets and data over these directories.
WORKDIR /app
RUN mkdir -p /app/secrets /app/data && chown -R mailpilot:mailpilot /app
USER mailpilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"]

CMD ["uvicorn", "mailpilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
