import os
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# --- Fast startup defaults / Cloud Run friendly ---
# Ensure Chroma (if used) writes to the writable path in Cloud Run
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma")

app = FastAPI(title="Agentic SRE Triage API (RAG)")


# ---- Lazy import helpers (avoid heavy work at import time) ----
_triage_fn = None
_upsert_case_fn = None


def _get_triage_fn():
    global _triage_fn
    if _triage_fn is None:
        # Lazy import so the server can start listening immediately
        from agent import triage as _triage  # noqa: WPS433
        _triage_fn = _triage
    return _triage_fn


def _get_upsert_case_fn():
    global _upsert_case_fn
    if _upsert_case_fn is None:
        backend = os.getenv("RAG_BACKEND", "chroma").lower()
        if backend == "pgvector":
            from rag_store_pg import upsert_case as _upsert  # noqa: WPS433
        else:
            # default to chroma
            from rag_store_chroma import upsert_case as _upsert  # noqa: WPS433
        _upsert_case_fn = _upsert
    return _upsert_case_fn


# ---- Health / readiness ----
@app.get("/healthz")
def healthz():
    # Extremely lightweight health check
    return {"status": "ok", "time": int(time.time())}


@app.get("/_ready")
def ready():
    """
    Minimal readiness check.
    We keep it cheap—just ensure lazy imports are resolvable.
    """
    try:
        _ = _get_triage_fn()
        _ = _get_upsert_case_fn()
        return {"ready": True}
    except Exception as exc:
        # If something is wrong with imports/config, surface a 503
        return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)


# ---- API: types (imported lazily to avoid any heavy model init during import) ----
# We import schema types at call-time inside handlers to keep module import cheap.
# If your schemas are lightweight, you can safely move these to the top-level.
from typing import Any  # noqa: E402


@app.post("/triage")
def do_triage(req: Any):  # typed dynamically to avoid importing schemas at module import
    try:
        # Import schemas here to keep module import minimal
        from schemas import TriageRequest, TriageDecision  # noqa: WPS433

        # Pydantic validation
        if not isinstance(req, dict):
            # FastAPI normally parses JSON into dict; still guard explicitly
            raise HTTPException(status_code=400, detail="Invalid payload")

        triage_req = TriageRequest(**req)
        triage_fn = _get_triage_fn()
        decision = triage_fn(triage_req.log)

        # Validate/shape the response
        return TriageDecision(**decision).model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        # Surface as 500 with message (avoid leaking secrets)
        raise HTTPException(status_code=500, detail=f"Triage failed: {exc}") from exc


@app.post("/feedback")
def feedback(payload: dict):
    try:
        if "text" not in payload or not payload["text"]:
            raise HTTPException(status_code=400, detail="'text' is required")

        doc_id = payload.get("id") or str(uuid.uuid4())
        text = payload["text"]
        meta = {k: v for k, v in payload.items() if k != "text"}
        meta["created_at"] = int(time.time())

        upsert_case = _get_upsert_case_fn()
        upsert_case(doc_id, text, meta)

        return {"ok": True, "id": doc_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback upsert failed: {exc}") from exc
