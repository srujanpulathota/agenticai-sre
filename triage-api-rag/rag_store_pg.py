import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

def _conn():
    inst = f"{os.getenv('PROJECT')}:{os.getenv('REGION')}:{os.getenv('PG_INSTANCE')}"
    return f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PGPASS')}@/{os.getenv('PG_DB')}?host=/cloudsql/{inst}"

emb = OpenAIEmbeddings(model="text-embedding-3-small")
COLLECTION = "exceptions_kb"

def get_store():
    return PGVector(connection_string=_conn(), collection_name=COLLECTION, embedding_function=emb)

def upsert_case(doc_id, text, metadata):
    get_store().add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])

def retrieve_similar(text, k=4, where: dict | None = None):
    docs = get_store().similarity_search(text, k=k, filter=where)
    return [{"text": d.page_content, "meta": d.metadata} for d in docs]
