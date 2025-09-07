FROM python:3.11-slim

# Faster, cleaner Python & pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (psycopg binaries still benefit from libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Install deps first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Copy the app
COPY . .

# Run as non-root
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run injects PORT; just expose for local clarity
EXPOSE 8080

# IMPORTANT: bind to $PORT provided by Cloud Run
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]