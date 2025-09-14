# ── Base ────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Faster, cleaner Python & pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Defaults that help in Cloud Run / Chroma demos
# (You can override these in Cloud Run → Variables)
ENV CHROMA_PERSIST_DIR=/tmp/chroma \
    ANONYMIZED_TELEMETRY=false

WORKDIR /app

# ── System deps (minimal; psycopg[binary] works without build tools) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ── Python deps first (better layer cache) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel pip-tools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# ── App code ────────────────────────────────────────────────────────────────────
COPY . .

# Non-root user
RUN useradd --create-home appuser && \
    mkdir -p ${CHROMA_PERSIST_DIR} && \
    chown -R appuser:appuser /app ${CHROMA_PERSIST_DIR}
USER appuser

# Cloud Run injects $PORT; expose for local clarity
EXPOSE 8080

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# IMPORTANT: bind to $PORT provided by Cloud Run
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
