import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- Config ----------
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "exceptions_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # empty for official OpenAI
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")

ANON_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in ("1", "true", "yes")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set; embeddings cannot be created.")

# ---------- Embedding function for OpenAI v1 SDK ----------
class OpenAIEmbeddingFunctionV1:
    """
    Callable that Chroma can use to embed texts via the OpenAI v1 SDK.
    Signature matches Chroma's expectation: fn(List[str]) -> List[List[float]].
    """
    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 organization: str | None = None, project: str | None = None):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        self.client = OpenAI(**kwargs)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

# Instantiate clients
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

# ---------- Public API used by agent.py ----------
def upsert_case(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

def retrieve_similar(text: str, k: int = 4) -> List[Dict[str, Any]]:
    q = collection.query(query_texts=[text], n_results=k)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]
