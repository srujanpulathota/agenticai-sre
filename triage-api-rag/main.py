import uuid, time, os
from fastapi import FastAPI
from schemas import TriageRequest, TriageDecision
from agent import triage

if os.getenv("RAG_BACKEND", "chroma") == "pgvector":
    from rag_store_pg import upsert_case
else:
    from rag_store_chroma import upsert_case

app = FastAPI(title="Agentic SRE Triage API (RAG)")

@app.post("/triage", response_model=TriageDecision)
def do_triage(req: TriageRequest):
    return triage(req.log)

@app.post("/feedback")
def feedback(payload: dict):
    import uuid as _uuid, time as _time
    doc_id = payload.get("id") or str(_uuid.uuid4())
    text = payload["text"]
    meta = {k: v for k, v in payload.items() if k != "text"}
    meta["created_at"] = int(_time.time())
    upsert_case(doc_id, text, meta)
    return {"ok": True, "id": doc_id}
