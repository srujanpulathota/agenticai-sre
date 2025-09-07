# Python slim base
FROM python:3.11-slim

# Reduce noise & ensure deterministic logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# (1) System deps often needed by common Python wheels; safe + small
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl libpq5 \
    && rm -rf /var/lib/apt/lists/*

# (2) Leverage Docker layer cache for deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# (3) Copy the rest of your app
COPY . .

# (4) Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# (5) Expose & run (PORT is honored if provided)
ENV PORT=8080
EXPOSE 8080
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
