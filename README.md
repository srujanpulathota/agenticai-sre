
# Agentic SRE Triage with RAG (n8n + LangChain)

## 0) Prereqs
- GCP project with: Cloud Run, Cloud Build, Secret Manager, Pub/Sub, Cloud Logging enabled.
- n8n deployed (Cloud Run) with a public webhook URL.
- Slack bot or webhook (optional).
- (Optional prod) Cloud SQL Postgres with pgvector.

## 1) Deploy triage-api-rag (LangChain + RAG)

### Local dev (venv)
```bash
cd triage-api-rag
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# set OPENAI_API_KEY, choose RAG_BACKEND=chroma for POC
python seed_kb.py
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Cloud Run deploy
```bash
gcloud builds submit --tag gcr.io/$PROJECT/triage-api-rag
gcloud run deploy triage-api-rag \
  --image gcr.io/$PROJECT/triage-api-rag \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars=MODEL=gpt-4o-mini,RAG_BACKEND=chroma \
  --set-secrets=OPENAI_API_KEY=OPENAI_API_KEY:latest
```

For **PGVector** in prod:
- Create Cloud SQL + `vector` extension (see infra/).
- Set env vars: `RAG_BACKEND=pgvector`, `PROJECT`, `REGION`, `PG_DB`, `PG_USER`, `PGPASS`, `PG_INSTANCE`.
- Add `--add-cloudsql-instances $PROJECT:$REGION:triage-pg` on deploy.

## 2) Deploy diag-api (Diagnostics)
```bash
cd ../diag-api
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn diag:app --host 0.0.0.0 --port 8080

gcloud builds submit --tag gcr.io/$PROJECT/diag-api
gcloud run deploy diag-api --image gcr.io/$PROJECT/diag-api \
  --region us-central1 --allow-unauthenticated
```

## 3) Deploy pubsub-proxy (Pub/Sub → n8n)
Set `N8N_WEBHOOK_URL` to your n8n Webhook (Production) URL.
```bash
cd ../pubsub-proxy
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

gcloud builds submit --tag gcr.io/$PROJECT/pubsub-proxy
gcloud run deploy pubsub-to-n8n --image gcr.io/$PROJECT/pubsub-proxy \
  --region us-central1 --allow-unauthenticated \
  --set-env-vars=N8N_WEBHOOK_URL="https://<n8n>/webhook/<id>",FORWARD_SECRET="<STRONG>"
```

Create **push** subscription:
```bash
gcloud pubsub subscriptions create logs-error-sub \
  --topic=logs-error-topic \
  --push-endpoint="https://<pubsub-to-n8n-url>/pubsub"
```

## 4) Terraform (optional quick setup)
```bash
cd ../infra
terraform init
terraform apply -var="project_id=$PROJECT" -var="region=us-central1"
```

After SQL is up, run once:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  metadata JSONB NOT NULL,
  embedding vector(1536)
);
CREATE INDEX ON kb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## 5) n8n workflow import
- Open n8n → “Import from file” → `n8n-workflow/agentic-triage-rag.json`.
- Update placeholders:
  - TRIAGE_API_URL
  - DIAG_API_URL
  - SLACK_CHANNEL
  - HEALTH URL(s)

## 6) Test end-to-end
```bash
gcloud logging write agentic_demo_log "Checkout API 500s spiking" --severity=ERROR
```
Expected: Pub/Sub → proxy → n8n → triage-api (RAG) → (optional) diag-api → Slack.

## 7) Feedback loop (KB learning)
When an incident is resolved, POST to `/feedback`:
```bash
curl -X POST https://<triage-api>/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "DB pool exhaustion error: remaining connection slots reserved ...",
    "service": "checkout-api",
    "root_cause": "DB pool exhaustion",
    "fix": "Increase pool; reduce concurrency",
    "runbook_url": "https://wiki/runbooks/db-pool",
    "tags": ["postgres","timeout","prod"],
    "source": "INC-12345"
  }'
```

## 8) Security
- Place triage/diag behind IAP or require API keys (custom header).
- Use Secret Manager for API keys and DB passwords.
- Restrict Pub/Sub push to a service account if needed.

## 9) Costs & Scale
- Cloud Run scales-to-zero.
- Pub/Sub is low cost.
- LLM usage: keep prompts compact; parsing via Pydantic avoids retries.

---

### Python venv best practice (as requested)
1. `python3 -m venv venv`  
2. Activate: `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\Activate.ps1` (Windows)  
3. `pip install -r requirements.txt`  
4. Run (`uvicorn ...` or `gunicorn ...`)  
5. `deactivate`
