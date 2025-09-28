# agent.py
import os, json, hashlib, re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin, quote

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import TriageDecision

# ----------------- Config -----------------
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BACKEND = os.getenv("RAG_BACKEND", "chroma").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # blank for official
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

RUNBOOK_BASE = os.getenv("RUNBOOK_BASE", "https://srujanpulathota.github.io/runbooks").rstrip("/")
DEFAULT_PROJECT = os.getenv("DEFAULT_PROJECT", "my-gcp-project")
DEFAULT_REGION  = os.getenv("DEFAULT_REGION", "us-central1")
ALERT_EMAIL     = os.getenv("ALERT_EMAIL", "you@gmail.com")

DEFAULT_PRIORITY = os.getenv("DEFAULT_PRIORITY", "P2")  # fallback for ERRORs if nothing matches
CRITICAL_SERVICES = {s.strip() for s in os.getenv("CRITICAL_SERVICES", "").split(",") if s.strip()}

ALLOWED_CMD_PREFIXES = ("kubectl","gcloud","curl","psql","grep","tail","journalctl","dig","nslookup","helm")

# Priority policy (kept short; tune keywords freely)
PRIORITY_POLICY = """
Return priority as one of P1, P2, P3, P4, P5:
- P1: Full outage / data loss / security incident / critical service totally down / widespread impact now.
- P2: Major functionality broken or high error rate (e.g., gateway 502/upstream, DB unavailable, CrashLoop/OOM).
- P3: Partial degradation, intermittent errors, slow responses, transient failures.
- P4: Minor bug / non-critical warnings.
- P5: Informational / noise.
Examples: "Application gateway error" or "nginx upstream 502" => P2 by default.
""".strip()

# ----------------- RAG select -----------------
if BACKEND == "pgvector":
    from rag_store_pg import retrieve_similar  # type: ignore
else:
    from rag_store_chroma import retrieve_similar  # type: ignore

# ----------------- LLM -----------------
_LLM: Optional[ChatOpenAI] = None
def _get_llm() -> ChatOpenAI:
    global _LLM
    if _LLM is not None:
        return _LLM
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")
    _LLM = ChatOpenAI(
        model=MODEL, temperature=0,
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE or None,
        organization=OPENAI_ORG_ID or None,
        max_retries=1, max_tokens=500,
    )
    return _LLM

# ----------------- Prompt & parser -----------------
_parser = PydanticOutputParser(pydantic_object=TriageDecision)

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     (
         "You are an SRE triage assistant. Return ONLY a JSON object matching the schema.\n"
         "Use 'similar_cases' when relevant.\n\n"
         f"Priority policy:\n{PRIORITY_POLICY}\n\n"
         "STRICT RULES:\n"
         "1) 'dedupe_key' stable across identical incidents.\n"
         "2) 'notify_channels' must include at least one channel (e.g. 'email:{ALERT_EMAIL}').\n"
         "3) 'runbook' must be a full http(s) URL (prefer similar_cases meta.url; else choose under RUNBOOK_BASE).\n"
         "4) 'suggest_cmds' MUST be 2–5 SHELL COMMANDS ONLY (no prose). Each starts with one of: "
         "kubectl,gcloud,curl,psql,grep,tail,journalctl,dig,nslookup,helm. Prefer read-only; if priority=P1 you may include a single controlled restart last.\n"
         "5) Keep outputs concise and actionable."
     ).replace("{ALERT_EMAIL}", ALERT_EMAIL)),
    ("human",
     "RUNBOOK_BASE: {RUNBOOK_BASE}\n"
     "Schema:\n{schema}\n\n"
     "Similar cases (JSON):\n{cases}\n\n"
     "Incoming log JSON:\n```json\n{log}\n```")
]).partial(schema=_parser.get_format_instructions(), RUNBOOK_BASE=RUNBOOK_BASE)

# ----------------- Helpers -----------------
def _stable_key(log: Dict[str, Any]) -> str:
    text = json.dumps({k: log.get(k) for k in ("logName","resource","textPayload","jsonPayload")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def _looks_like_url(s: Optional[str]) -> bool:
    try:
        u = urlparse(s or ""); return u.scheme in ("http","https") and bool(u.netloc)
    except Exception:
        return False

RUNBOOK_MAP = {
    "502": "502s", "upstream": "upstream-failure", "gateway": "502s",
    "timeout": "timeouts", "db": "db-connection", "postgres":"db-connection",
    "connection":"db-connection", "oom":"oom", "memory":"oom",
}
def _pick_runbook_from_text(probable: str, service: str, full_text: str) -> str:
    hay = f"{probable} {service} {full_text}".lower()
    for k,slug in RUNBOOK_MAP.items():
        if k in hay: return urljoin(RUNBOOK_BASE + "/", slug)
    return urljoin(RUNBOOK_BASE + "/", "triage-general")

def _logs_deeplink(project_id: str, service: str, needle: str, minutes: int=60) -> str:
    base = "https://console.cloud.google.com/logs/query"
    query = ('resource.type="cloud_run_revision"\n'
             f'resource.labels.service_name="{service}"\n'
             f'textPayload:"{(needle or "")[:80]}"')
    params = [f"query={quote(query)}", f"timeRange=PT{int(minutes)}M", f"project={project_id}"]
    return f"{base};{';'.join(params)}"

def _coerce_list(x: Any) -> List[str]:
    if isinstance(x, list): return [str(i) for i in x]
    if isinstance(x, str) and x.strip():
        try:
            j = json.loads(x); 
            if isinstance(j, list): return [str(i) for i in j]
        except Exception: pass
        return [p.strip() for p in re.split(r"[\n,]+", x) if p.strip()]
    return []

_CMD_RX = re.compile(r"^(kubectl|gcloud|curl|psql|grep|tail|journalctl|dig|nslookup|helm)\b", re.I)
def _looks_like_cmd(s: str) -> bool:
    s = (s or "").strip()
    return bool(s and len(s.split())>1 and _CMD_RX.match(s))

def _normalize_cmds(raw: Any) -> List[str]:
    lines = []
    for item in _coerce_list(raw):
        lines.extend(re.split(r"[;\n]+", item))
    out, seen = [], set()
    for l in (x.strip() for x in lines if x.strip()):
        if _looks_like_cmd(l):
            k = l.lower()
            if k not in seen:
                seen.add(k); out.append(l)
    return out

# ---------- Deterministic priority ----------
_ALLOWED = {"P1","P2","P3","P4","P5"}
def _normalize_priority(p: Optional[str], default: str="P3") -> str:
    if not p: return default
    s = str(p).strip().upper()
    if s in _ALLOWED: return s
    if s in {"1","HIGH","SEV1"}: return "P1"
    if s in {"2","SEV2"}: return "P2"
    if s in {"3","MEDIUM","SEV3"}: return "P3"
    if s in {"4","LOW","SEV4"}: return "P4"
    if s in {"5","INFO","SEV5"}: return "P5"
    return default

P1_KEYS = {"outage","data loss","security breach","ransomware","compromise"}
P2_KEYS = {"gateway","upstream","502","db unavailable","crashloop","oom","timeout","service unavailable"}
P3_KEYS = {"degraded","intermittent","slow","retrying","transient"}

def _deterministic_priority(log: Dict[str, Any], draft: Optional[str], probable: Optional[str], service: str) -> str:
    # Build a big haystack of text to search
    text = " ".join([
        (log.get("textPayload") or ""),
        json.dumps(log.get("jsonPayload", {}), ensure_ascii=False),
        (probable or "")
    ]).lower()

    # explicit hint from producer
    hint = (log.get("jsonPayload") or {}).get("priority") or (log.get("_priority_hint") or "")
    if hint:
        return _normalize_priority(str(hint), default="P3")

    # keyword rules
    if any(k in text for k in P1_KEYS): 
        return "P1"
    if any(k in text for k in P2_KEYS): 
        # escalate to P1 for critical services if configured
        if service in CRITICAL_SERVICES:
            return "P1"
        return "P2"
    if any(k in text for k in P3_KEYS): 
        return "P3"

    # no keywords: use draft if sane, else default by severity
    draft_norm = _normalize_priority(draft, default="P3")
    if draft_norm in _ALLOWED:
        return draft_norm

    sev = (log.get("severity") or "DEFAULT").upper()
    if sev in {"EMERGENCY","ALERT","CRITICAL"}: return "P1"
    if sev in {"ERROR"}: return _normalize_priority(DEFAULT_PRIORITY, "P2")  # P2 default for errors
    if sev in {"WARNING"}: return "P3"
    return "P4"

# ----------------- Public API -----------------
def triage(log: Dict[str, Any]) -> TriageDecision:
    query_text = (
        log.get("textPayload")
        or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
        or json.dumps(log, ensure_ascii=False)
    )
    cases = retrieve_similar(query_text, k=4)
    log.setdefault("_dedupe_hint", _stable_key(log))

    msg = _prompt.format_messages(cases=json.dumps(cases, ensure_ascii=False),
                                  log=json.dumps(log, ensure_ascii=False))
    out = _get_llm().invoke(msg)
    decision = _parser.parse(out.content)

    # Harden & enrich
    service = decision.service or log.get("labels", {}).get("service_name") or "unknown-service"
    decision.service = service

    if not (decision.dedupe_key and decision.dedupe_key.strip()):
        decision.dedupe_key = _stable_key(log)

    # Notify channels
    nc = _coerce_list(decision.notify_channels)
    if not nc: nc = [f"email:{ALERT_EMAIL}"]
    decision.notify_channels = nc

    # Runbook: ensure URL / map / link
    rb = decision.runbook
    if not _looks_like_url(rb):
        # try similar-cases URL first
        sim_urls: List[str] = []
        try:
            for c in cases:
                meta = c.get("meta") if isinstance(c, dict) else None
                u = (meta or {}).get("url") if isinstance(meta, dict) else None
                if u and _looks_like_url(u): sim_urls.append(u)
        except Exception:
            pass
        chosen = next(iter(sim_urls), None)
        if not chosen:
            chosen = _pick_runbook_from_text(decision.probable_cause or "", service, query_text)
            if not _looks_like_url(chosen):
                chosen = _logs_deeplink(DEFAULT_PROJECT, service, decision.probable_cause or "error")
        decision.runbook = chosen
    else:
        if rb.rstrip("/") == RUNBOOK_BASE.rstrip("/"):
            decision.runbook = _pick_runbook_from_text(decision.probable_cause or "", service, query_text)

    # Commands: normalize; fallback if weak
    cmds = _normalize_cmds(decision.suggest_cmds)
    if len(cmds) < 2:
        # simple heuristics
        low = (decision.probable_cause or "").lower()
        if any(k in low for k in ["nginx","gateway","upstream","502"]):
            cmds = [
                f"kubectl logs -l service={service} --tail=100",
                f"kubectl get services {service} -o yaml",
                f"kubectl describe pod -l service={service}",
                "curl -s -o /dev/null -w '%{http_code}\\n' https://<web-edge-service-url>",
            ]
        else:
            cmds = [
                ("gcloud logging read "
                 f"'resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"{service}\" "
                 "AND severity>=\"ERROR\"' --limit=50 --format=json"),
                f"gcloud run services describe {service} --region {DEFAULT_REGION} --project {DEFAULT_PROJECT}",
                "curl -I https://<service-url>",
            ]
    if (decision.priority or "").upper() == "P1":
        restart = f"kubectl rollout restart deployment/{service}"
        if not any(restart in c for c in cmds): cmds.append(restart)
    decision.suggest_cmds = cmds[:5]

    # Priority: make deterministic
    decision.priority = _deterministic_priority(
        log=log,
        draft=decision.priority,
        probable=decision.probable_cause,
        service=service,
    )

    return decision
