# agent.py
import os, json, hashlib
from typing import Dict, Any, Optional

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
# IMPORTANT: do not use OPENAI_PROJECT here (causes create(project=...) issues)

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
        # keep it cheap & resilient
        max_retries=1,
        max_tokens=400,
    )
    return _LLM

# ----------------- Prompt & parser -----------------
_parser = PydanticOutputParser(pydantic_object=TriageDecision)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an SRE triage assistant. Use 'similar_cases' when relevant. "
            "Output ONLY valid JSON per schema. Include a stable dedupe_key.",
        ),
        (
            "human",
            "Schema:\n{schema}\n\nSimilar cases:\n{cases}\n\nLog JSON:\n```json\n{log}\n```",
        ),
    ]
).partial(schema=_parser.get_format_instructions())

# ----------------- Helpers -----------------
def _stable_key(log: Dict[str, Any]) -> str:
    text = json.dumps(
        {k: log.get(k) for k in ("logName", "resource", "textPayload", "jsonPayload")},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

# ----------------- Public API -----------------
def triage(log: Dict[str, Any]) -> TriageDecision:
    """
    Given a single Cloud Logging-like log dict, return a structured TriageDecision.
    """
    query = (
        log.get("textPayload")
        or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
        or json.dumps(log, ensure_ascii=False)
    )

    cases = retrieve_similar(query, k=4)
    log.setdefault("_dedupe_hint", _stable_key(log))

    msg = _prompt.format_messages(
        cases=json.dumps(cases, ensure_ascii=False),
        log=json.dumps(log, ensure_ascii=False),
    )

    llm = _get_llm()
    out = llm.invoke(msg)
    return _parser.parse(out.content)
