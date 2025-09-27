# agent.py
import os, json, hashlib, re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin, quote

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import TriageDecision

# ----------------- Config from environment -----------------
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BACKEND = os.getenv("RAG_BACKEND", "chroma").lower()

# OpenAI creds & options (read by ChatOpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # leave unset for official OpenAI
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

# Agent fallbacks / deep links
RUNBOOK_BASE = os.getenv("RUNBOOK_BASE", "https://srujanpulathota.github.io/runbooks").rstrip("/")
DEFAULT_PROJECT = os.getenv("DEFAULT_PROJECT", "my-gcp-project")
DEFAULT_REGION = os.getenv("DEFAULT_REGION", "us-central1")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "you@gmail.com")

# ----------------- RAG backend selector -----------------
if BACKEND == "pgvector":
    from rag_store_pg import retrieve_similar  # type: ignore
else:
    from rag_store_chroma import retrieve_similar  # type: ignore

# ----------------- Lazy LLM init -----------------
_LLM: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    """
    Build ChatOpenAI using env-provided creds. Do NOT pass a 'project' parameter.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    if not OPENAI_API_KEY:
        raise ValueError("Please provide an OpenAI API key via env var OPENAI_API_KEY.")

    _LLM = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE or None,
        organization=OPENAI_ORG_ID or None,
        max_retries=1,
        max_tokens=500,  # allow room for cmds + url
    )
    return _LLM


# ----------------- Prompt & parser -----------------
_parser = PydanticOutputParser(pydantic_object=TriageDecision)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an SRE triage assistant.\n"
                "Return ONLY a valid JSON object that matches the schema.\n"
                "Use 'similar_cases' when relevant.\n\n"
                "STRICT RULES:\n"
                "1) 'dedupe_key' MUST be stable across identical incidents.\n"
                "2) 'notify_channels' MUST be non-empty; include at least one channel. "
                "   If email is appropriate, use 'email:{ALERT_EMAIL}'.\n"
                "3) 'runbook' MUST be a full http(s) URL. Prefer a URL from similar_cases[i].meta.url "
                "(if it clearly matches). If not available, choose the best runbook path under RUNBOOK_BASE. "
                "Never return plain text in 'runbook'.\n"
                "4) 'suggest_cmds' MUST contain 2-5 SAFE, read-only first steps (no destructive actions). "
                "Favor commands to inspect logs, service health, and configurations. "
                "If priority is P1, you may include exactly ONE controlled restart/rollout as the last command.\n"
                "5) Keep outputs concise and actionable.\n"
            ).replace("{ALERT_EMAIL}", ALERT_EMAIL)
        ),
        (
            "human",
            "RUNBOOK_BASE: {RUNBOOK_BASE}\n"
            "Schema:\n{schema}\n\n"
            "Similar cases (JSON):\n{cases}\n\n"
            "Incoming log JSON:\n```json\n{log}\n```",
        ),
    ]
).partial(
    schema=_parser.get_format_instructions(),
    RUNBOOK_BASE=RUNBOOK_BASE,
)

# ----------------- Helpers -----------------
def _stable_key(log: Dict[str, Any]) -> str:
    text = json.dumps(
        {k: log.get(k) for k in ("logName", "resource", "textPayload", "jsonPayload")},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _looks_like_url(s: Optional[str]) -> bool:
    try:
        u = urlparse(s or "")
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


# map keywords to runbook slugs
RUNBOOK_MAP = {
    "502": "502s",
    "upstream": "upstream-failure",
    "gateway": "502s",
    "timeout": "timeouts",
    "db": "db-connection",
    "postgres": "db-connection",
    "connection": "db-connection",
    "oom": "oom",
    "memory": "oom",
}


def _pick_runbook_from_text(probable: str, service: str, full_text: str) -> str:
    hay = f"{probable} {service} {full_text}".lower()
    for key, slug in RUNBOOK_MAP.items():
        if key in hay:
            return urljoin(RUNBOOK_BASE + "/", slug)
    return urljoin(RUNBOOK_BASE + "/", "triage-general")


def _logs_deeplink(project_id: str, service: str, needle: str, minutes: int = 60) -> str:
    """
    Build a Google Cloud Log Explorer deep link for the last N minutes that
    filters to this Cloud Run service and a key phrase.
    """
    base = "https://console.cloud.google.com/logs/query"
    query = (
        'resource.type="cloud_run_revision"\n'
        f'resource.labels.service_name="{service}"\n'
        f'textPayload:"{needle[:80]}"'
    )
    # The modern log explorer uses parameters in the URL path after semicolons.
    # We'll keep it simple.
    params = [
        f"query={quote(query)}",
        f"timeRange=PT{int(minutes)}M",
        f"project={project_id}",
    ]
    return f"{base};{';'.join(params)}"


def _gen_cmds(service: str, probable: str, resource_type: Optional[str]) -> List[str]:
    svc = service or "app"
    cmds: List[str] = [
        # read-only inspection commands
        (
            "gcloud logging read "
            f"'resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"{svc}\" "
            "AND severity>=\"ERROR\"' --limit=50 --format=json"
        ),
        f"gcloud run services describe {svc} --region {DEFAULT_REGION} --project {DEFAULT_PROJECT}",
        "curl -s -o /dev/null -w '%{http_code}\\n' https://<YOUR_SERVICE_URL>",
    ]
    if re.search(r"(db|postgres|connection|pool|timeout)", (probable or ""), re.I):
        cmds.append("gcloud sql instances list --format='value(name,state)'")
        cmds.append("gcloud sql instances describe <INSTANCE> --format=json")
    # cap to 5
    return cmds[:5]


def _coerce_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str) and x.strip():
        try:
            j = json.loads(x)
            return [str(i) for i in j] if isinstance(j, list) else [x]
        except Exception:
            return [x]
    return []


# ----------------- Public API -----------------
def triage(log: Dict[str, Any]) -> TriageDecision:
    """
    Given a single Cloud Logging-like log dict, return a structured TriageDecision.
    Guarantees:
      - runbook is a working URL (prefer similar_cases URLs → RUNBOOK_BASE map → logs deep link)
      - suggest_cmds contains 2–5 entries
      - notify_channels non-empty (defaults to email)
      - dedupe_key stable
    """
    # 1) Build the user query text and fetch similar cases
    query_text = (
        log.get("textPayload")
        or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
        or json.dumps(log, ensure_ascii=False)
    )
    cases = retrieve_similar(query_text, k=4)

    # 2) Provide a stable hint for dedupe if missing
    log.setdefault("_dedupe_hint", _stable_key(log))

    # 3) Call the LLM
    msg = _prompt.format_messages(
        cases=json.dumps(cases, ensure_ascii=False),
        log=json.dumps(log, ensure_ascii=False),
    )
    llm = _get_llm()
    out = llm.invoke(msg)

    # 4) Parse into TriageDecision
    decision = _parser.parse(out.content)  # -> TriageDecision object

    # 5) Post-process & harden fields
    # service fallback
    service = decision.service or log.get("labels", {}).get("service_name") or "unknown-service"
    decision.service = service

    # ensure dedupe_key
    if not (decision.dedupe_key and decision.dedupe_key.strip()):
        decision.dedupe_key = _stable_key(log)

    # ensure notify_channels (default to email channel for the demo)
    nc = _coerce_list(decision.notify_channels)
    if not nc:
        nc = [f"email:{ALERT_EMAIL}"]
    decision.notify_channels = nc

    # prefer a URL from similar cases metadata
    sim_urls: List[str] = []
    try:
        for c in cases:
            meta = c.get("meta") if isinstance(c, dict) else None
            u = (meta or {}).get("url") if isinstance(meta, dict) else None
            if _looks_like_url(u):
                sim_urls.append(u)  # keep order
    except Exception:
        pass

    # ensure runbook is a URL
    rb = decision.runbook
    if not _looks_like_url(rb):
        # try a URL from similar cases first
        chosen = next((u for u in sim_urls if _looks_like_url(u)), None)
        if not chosen:
            # map by keywords under RUNBOOK_BASE
            chosen = _pick_runbook_from_text(decision.probable_cause or "", service, query_text)
            if not _looks_like_url(chosen):
                # last resort: logs explorer deep link for this service
                chosen = _logs_deeplink(DEFAULT_PROJECT, service, decision.probable_cause or "error")
        decision.runbook = chosen

    # ensure suggest_cmds exist (2–5)
    cmds = _coerce_list(decision.suggest_cmds)
    if len(cmds) < 2:
        resource_type = (log.get("resource") or {}).get("type")
        cmds = _gen_cmds(service, decision.probable_cause or "", resource_type)
    # If P1 and we want to allow ONE controlled restart as last step, append it here.
    if (decision.priority or "").upper() == "P1":
        # keep it optional and safe—comment out if you never want restarts auto-suggested
        # cmds.append(f"gcloud run services update {service} --region {DEFAULT_REGION} --revision-suffix=rollout-$(date +%s)")
        pass
    # cap 2–5
    if len(cmds) < 2:
        cmds = cmds + ["echo 'inspect further'"]
    decision.suggest_cmds = cmds[:5]

    return decision
