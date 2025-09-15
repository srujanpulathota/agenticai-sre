# rag_store_chroma.py
import os
import json
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ------------------------------------------------------------------------------
# Config (override via env if needed)
# ------------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")  # Cloud Run-friendly
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "exceptions_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # leave unset for official OpenAI
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # needed for sk-proj-* keys

# Default OFF to silence noisy telemetry in logs; set true to re-enable
ANON_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in ("1", "true", "yes")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set; embeddings cannot be created.")

# Ensure the persistence path exists (especially useful on local or GCE VM)
os.makedirs(CHROMA_PATH, exist_ok=True)

# ------------------------------------------------------------------------------
# Embedding function compatible with Chroma ≥0.4.16 (expects __call__(input=...))
# ------------------------------------------------------------------------------
class OpenAIEmbeddingFunctionV1:
    """
    Callable used by Chroma to embed texts via the OpenAI v1 SDK.
    MUST have signature: __call__(self, input: List[str]) -> List[List[float]]
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        self.client = OpenAI(**kwargs)

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        # OpenAI v1: embeddings.create(model=..., input=[...])
        resp = self.client.embeddings.create(model=self.model, input=input)
        return [d.embedding for d in resp.data]

# ------------------------------------------------------------------------------
# Chroma client & collection
# ------------------------------------------------------------------------------
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=ANON_TELEMETRY),
)

emb_fn = OpenAIEmbeddingFunctionV1(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE or None,
    organization=OPENAI_ORG_ID or None,
    project=OPENAI_PROJECT or None,
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb_fn,
)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _coerce_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata must be primitives (str/int/float/bool). Coerce lists/dicts to strings.
    """
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = "" if v is None else v
        elif isinstance(v, (list, tuple)):
            # Join primitive lists; JSON-dump mixed/nested
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = ",".join("" if x is None else str(x) for x in v)
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

# ------------------------------------------------------------------------------
# Public API used by agent.py
# ------------------------------------------------------------------------------
def upsert_case(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    safe_meta = _coerce_metadata(metadata)
    collection.upsert(ids=[doc_id], documents=[text], metadatas=[safe_meta])

def retrieve_similar(text: str, k: int = 4) -> List[Dict[str, Any]]:
    q = collection.query(query_texts=[text], n_results=k)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]
