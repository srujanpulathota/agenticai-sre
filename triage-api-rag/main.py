import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Body   # ← add Body
from fastapi.responses import JSONResponse
from typing import Any

os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/chroma")

app = FastAPI(title="Agentic SRE Triage API (RAG)")

_triage_fn = None
_upsert_case_fn = None

def _get_triage_fn():
    global _triage_fn
    if _triage_fn is None:
        from agent import triage as _triage
        _triage_fn = _triage
    return _triage_fn

def _get_upsert_case_fn():
    global _upsert_case_fn
    if _upsert_case_fn is None:
        backend = os.getenv("RAG_BACKEND", "chroma").lower()
        if backend == "pgvector":
            from rag_store_pg import upsert_case as _upsert
        else:
            from rag_store_chroma import upsert_case as _upsert
        _upsert_case_fn = _upsert
    return _upsert_case_fn


@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": int(time.time())}

@app.get("/_ready")
def ready():
    try:
        _ = _get_triage_fn()
        _ = _get_upsert_case_fn()
        return {"ready": True}
    except Exception as exc:
        return JSONResponse({"ready": False, "error": str(exc)}, status_code=503)


# ---------- FIXED /triage ----------
@app.post("/triage")
def do_triage(req: Any = Body(...)):  # ← force body, not query
    try:
        from schemas import TriageRequest, TriageDecision

        if not isinstance(req, dict):
            raise HTTPException(status_code=400, detail="Invalid payload")

        # Accept either {"log": {...}} or a raw log object
        log = req.get("log", req)

        triage_fn = _get_triage_fn()
        decision = triage_fn(log)  # should return TriageDecision

        # If it’s already a Pydantic model, dump to dict; if dict, return as-is
        if isinstance(decision, TriageDecision):
            return decision.model_dump()
        elif isinstance(decision, dict):
            return TriageDecision(**decision).model_dump()
        else:
            raise HTTPException(status_code=500, detail="Unexpected decision type from triage()")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Triage failed: {exc}") from exc


# ---------- (small hardening) /feedback ----------
@app.post("/feedback")
def feedback(payload: dict = Body(...)):  # ← force body, not query
    try:
        if not payload.get("text"):
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

@app.get("/_diag/openai")
def diag_openai():
    import os, socket, ssl
    from langchain_openai import ChatOpenAI
    info = {
        "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_BASE": os.getenv("OPENAI_BASE") or "",
        "MODEL": os.getenv("MODEL"),
        "OPENAI_PROJECT": os.getenv("OPENAI_PROJECT"),
        "OPENAI_ORG_ID": os.getenv("OPENAI_ORG_ID"),
    }
    try:
        _ = ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0)
        info["llm_init_ok"] = True
    except Exception as e:
        info["llm_init_ok"] = False
        info["llm_init_error"] = repr(e)

    host = (os.getenv("OPENAI_BASE") or "https://api.openai.com").replace("https://","").split("/")[0]
    try:
        sock = socket.create_connection((host, 443), timeout=3)
        ssl.create_default_context().wrap_socket(sock, server_hostname=host).close()
        info["egress_ok"] = True
    except Exception as e:
        info["egress_ok"] = False
        info["egress_error"] = repr(e)
    return info

