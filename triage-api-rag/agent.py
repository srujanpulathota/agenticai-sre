import os, json, hashlib
from typing import Dict, Any, Optional

import httpx
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from schemas import TriageDecision

# ----------------- Config from environment -----------------
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BACKEND = os.getenv("RAG_BACKEND", "chroma").lower()

# OpenAI creds & options
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")  # leave unset for official OpenAI; else e.g. https://api.openai.com/v1
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")  # needed for sk-proj-* keys

# ----------------- RAG backend selector -----------------
if BACKEND == "pgvector":
    from rag_store_pg import retrieve_similar  # type: ignore
else:
    from rag_store_chroma import retrieve_similar  # type: ignore

# ----------------- Lazy LLM init -----------------
_LLM: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    """
    Build a ChatOpenAI bound to an explicit OpenAI client that uses a proxy-free httpx Client.
    This avoids the 'proxies' kwarg incompatibility between langchain-openai and openai versions.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    if not OPENAI_API_KEY:
        # Be explicit to aid ops when the secret/env is missing
        raise ValueError("Please provide an OpenAI API key via env var OPENAI_API_KEY.")

    # Proxy-free HTTP client (prevents unsupported 'proxies' kwarg paths)
    http_client = httpx.Client(proxies=None, timeout=None)

    oa_kwargs = {"api_key": OPENAI_API_KEY, "http_client": http_client}
    if OPENAI_BASE:
        oa_kwargs["base_url"] = OPENAI_BASE
    if OPENAI_ORG_ID:
        oa_kwargs["organization"] = OPENAI_ORG_ID
    if OPENAI_PROJECT:
        oa_kwargs["project"] = OPENAI_PROJECT

    oa_client = OpenAI(**oa_kwargs)

    _LLM = ChatOpenAI(
        model=MODEL,
        temperature=0,
        client=oa_client,  # hand in the prepared OpenAI client
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
    # Build a deterministic digest over stable log fields
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
    # Query text for retrieval: prefer textPayload, else jsonPayload, else whole log
    query = (
        log.get("textPayload")
        or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
        or json.dumps(log, ensure_ascii=False)
    )

    # Retrieve similar cases from the vector store
    cases = retrieve_similar(query, k=4)

    # Ensure a stable dedupe hint exists
    log.setdefault("_dedupe_hint", _stable_key(log))

    # Build prompt and invoke LLM
    msg = _prompt.format_messages(
        cases=json.dumps(cases, ensure_ascii=False),
        log=json.dumps(log, ensure_ascii=False),
    )
    llm = _get_llm()
    out = llm.invoke(msg)

    # Validate & parse to the schema
    return _parser.parse(out.content)
