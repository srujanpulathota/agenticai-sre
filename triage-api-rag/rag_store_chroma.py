import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma")
emb = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

collection = client.get_or_create_collection(name="exceptions_kb", embedding_function=emb)

def upsert_case(doc_id: str, text: str, metadata: dict):
    collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

def retrieve_similar(text: str, k: int = 4):
    q = collection.query(query_texts=[text], n_results=k)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    return [{"text": d, "meta": m} for d, m in zip(docs, metas)]
