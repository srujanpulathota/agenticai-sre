import os, time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Structured logging to Cloud Logging
from google.cloud import logging as gcloud_logging

SERVICE = os.getenv("SERVICE_NAME", "web-edge")
app = FastAPI(title="web-edge demo")

# Cloud Logging client: structured logs go to jsonPayload, resource is cloud_run_revision
_gl_client = gcloud_logging.Client()
_struct = _gl_client.logger("app-web-edge")  # log name (appears in logName)

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": SERVICE, "ts": int(time.time())}

@app.get("/ok")
def ok():
    _struct.log_struct(
        {"service": SERVICE, "msg": "demo OK", "path": "/ok"},
        severity="INFO",
    )
    return {"ok": True}

@app.get("/boom")
def boom(msg: str = "nginx: upstream prematurely closed connection while reading response header from upstream"):
    # Emit a structured ERROR entry; include service_name to make triage happy
    _struct.log_struct(
        {
            "service": SERVICE,
            "text": msg,
            "labels": {"service_name": SERVICE},
            "hint": "call /boom?msg=your+text to simulate various errors",
        },
        severity="ERROR",
    )
    # Return a 502 to resemble upstream errors
    raise HTTPException(status_code=502, detail="Simulated upstream 502")
