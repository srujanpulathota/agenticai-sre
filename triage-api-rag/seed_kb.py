from rag_store_chroma import upsert_case  # swap to rag_store_pg if using pgvector

samples = [
  ("db-timeout-1", "FATAL: remaining connection slots are reserved for non-replication superuser connections",
   {"service":"checkout-api","root_cause":"DB pool exhaustion",
    "fix":"Increase pool size or reduce concurrency; recycle pods","runbook_url":"https://wiki/runbooks/db-pool",
    "tags":["postgres","timeout"],"source":"seed"}),
  ("nginx-502-1", "upstream prematurely closed connection while reading response header from upstream",
   {"service":"web-edge","root_cause":"App crash or slow upstream",
    "fix":"Check app pods, restart if needed","runbook_url":"https://wiki/runbooks/502s",
    "tags":["nginx","502"],"source":"seed"}),
]
for id_, text, meta in samples:
    upsert_case(id_, text, meta)
print("seeded OK")
