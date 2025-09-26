# rag_store_chroma.py
import os
import json
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "exceptions_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
ANON_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in ("1", "true", "yes")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set; embeddings cannot be created.")

os.makedirs(CHROMA_PATH, exist_ok=True)

class OpenAIEmbeddingFunctionV1:
    def __init__(self, model: str, api_key: str, base_url: str | None = None, organization: str | None = None):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        # DO NOT pass 'project' here
        self.client = OpenAI(**kwargs)

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        resp = self.client.embeddings.create(model=self.model, input=input)
        return [d.embedding for d in resp.data]

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=ANON_TELEMETRY),
)

emb_fn = OpenAIEmbeddingFunctionV1(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE or None,
    organization=OPENAI_ORG_ID or None,
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn,
)

def _coerce_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = "" if v is None else v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = ",".join("" if x is None else str(x) for x in v)
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

def upsert_case(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    safe_meta = _coerce_metadata(metadata)
    collection.upsert(ids=[doc_id], documents=[text], metadatas=[safe_meta])

def retrieve_similar(text: str, k: int = 4) -> List[Dict[str, Any]]:
    q = collection.query(query_texts=[text], n_results=k)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]
