import time, socket, requests
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(title="Diagnostics API")

class HealthResp(BaseModel):
    ok: bool
    status: int | None = None
    latency_ms: int | None = None
    error: str | None = None

@app.get("/healthcheck", response_model=HealthResp)
def healthcheck(url: str = Query(...)):
    t0 = time.time()
    try:
        r = requests.get(url, timeout=3)
        return HealthResp(ok=r.status_code < 500, status=r.status_code, latency_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return HealthResp(ok=False, error=str(e))

class DNSResp(BaseModel):
    ok: bool
    resolve_ms: int | None = None
    error: str | None = None

@app.get("/dns", response_model=DNSResp)
def dns(host: str = Query(...)):
    t0 = time.time()
    try:
        socket.gethostbyname(host)
        return DNSResp(ok=True, resolve_ms=int((time.time()-t0)*1000))
    except Exception as e:
        return DNSResp(ok=False, error=str(e))
