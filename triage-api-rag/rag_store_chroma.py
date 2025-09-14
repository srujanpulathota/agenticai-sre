import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# -------- Config from env --------
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma")  # Cloud Run-friendly default
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "exceptions_kb")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # e.g., https://api.fireworks.ai/inference/v1

# Chroma telemetry: default OFF (can enable with ANONYMIZED_TELEMETRY=true)
ANON_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in ("1", "true", "yes")

# -------- Client & embeddings --------
if not OPENAI_API_KEY:
    # Be explicit—this is the most common misconfig during deploys
    raise RuntimeError("OPENAI_API_KEY is not set; cannot initialize Chroma OpenAIEmbeddingFunction.")

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=ANON_TELEMETRY),
)

emb = embedding_functions.OpenAIEmbeddingFunction(
    model_name=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    api_base=OPENAI_BASE,  # None is fine if using api.openai.com
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=emb,
)

# -------- API used by agent.py --------
def upsert_case(doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
    """
    Insert or update a single KB entry.
    """
    collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])


def retrieve_similar(text: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Return top-k similar items as {text, meta} dicts.
    """
    q = collection.query(query_texts=[text], n_results=k)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]
