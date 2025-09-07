# Pub/Sub topic for ERROR logs
resource "google_pubsub_topic" "logs_err_topic" {
  name = "logs-error-topic"
}

# Route severity>=ERROR logs to Pub/Sub
resource "google_logging_project_sink" "error_to_pubsub" {
  name        = "sink-errors-to-pubsub"
  destination = "pubsub.googleapis.com/${google_pubsub_topic.logs_err_topic.id}"
  filter      = "severity>=ERROR"
}

# Allow sink writer to publish to topic
data "google_iam_policy" "pubsub_publish" {
  binding {
    role    = "roles/pubsub.publisher"
    members = [google_logging_project_sink.error_to_pubsub.writer_identity]
  }
}
resource "google_pubsub_topic_iam_policy" "topic_policy" {
  topic       = google_pubsub_topic.logs_err_topic.name
  policy_data = data.google_iam_policy.pubsub_publish.policy_data
}

# (Optional) Cloud SQL for PGVector
resource "google_sql_database_instance" "triage_pg" {
  name             = "triage-pg"
  database_version = "POSTGRES_15"
  region           = var.region
  settings {
    tier = "db-custom-2-8192"
    ip_configuration { ipv4_enabled = false }
    database_flags { name = "shared_preload_libraries" value = "vector" }
  }
}
resource "google_sql_database" "rag" {
  name     = "rag"
  instance = google_sql_database_instance.triage_pg.name
}
resource "google_sql_user" "rag" {
  instance = google_sql_database_instance.triage_pg.name
  name     = "rag"
  password = "REPLACE_ME_STRONG"
}

# After SQL is created, run the following manually once:
# CREATE EXTENSION IF NOT EXISTS vector;
# CREATE TABLE IF NOT EXISTS kb (
#   id TEXT PRIMARY KEY,
#   text TEXT NOT NULL,
#   metadata JSONB NOT NULL,
#   embedding vector(1536)
# );
# CREATE INDEX ON kb USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
