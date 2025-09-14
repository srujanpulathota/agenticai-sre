import os, json, hashlib
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from schemas import TriageDecision

# --- Config from env ---
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BACKEND = os.getenv("RAG_BACKEND", "chroma").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_BASE")              # e.g., https://api.fireworks.ai/inference/v1
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")        # used by newer OpenAI "project keys" (sk-proj-...)

# --- RAG retriever selection ---
if BACKEND == "pgvector":
    from rag_store_pg import retrieve_similar
else:
    from rag_store_chroma import retrieve_similar

# --- Lazy LLM init (so the server starts even if envs are missing) ---
_LLM: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _LLM
    if _LLM is not None:
        return _LLM

    if not OPENAI_API_KEY:
        # Keep message explicit — helps users diagnose missing secret mapping.
        raise ValueError(
            "Please provide an OpenAI API key via env var OPENAI_API_KEY."
        )

    client_kwargs = {}
    if OPENAI_BASE:
        client_kwargs["base_url"] = OPENAI_BASE
    if OPENAI_ORG_ID:
        client_kwargs["organization"] = OPENAI_ORG_ID
    # NOTE: Some langchain-openai versions read OPENAI_PROJECT from the environment.
    # Passing it as a kwarg may be unsupported; we rely on the env var.

    _LLM = ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
        **client_kwargs,
    )
    return _LLM


# --- Output parsing / prompt ---
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


def _stable_key(log: Dict[str, Any]) -> str:
    # Build a deterministic key over the most stable fields
    text = json.dumps(
        {k: log.get(k) for k in ("logName", "resource", "textPayload", "jsonPayload")},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def triage(log: Dict[str, Any]) -> TriageDecision:
    """
    Given a single Cloud Logging-like log dict, return a structured TriageDecision.
    """
    # Query text for RAG retrieval: prefer plain text payload, otherwise json payload, else full log
    query = (
        log.get("textPayload")
        or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
        or json.dumps(log, ensure_ascii=False)
    )

    # Retrieve similar past cases from the chosen vector store
    cases = retrieve_similar(query, k=4)

    # Add a stable dedupe hint if caller didn't supply one
    log.setdefault("_dedupe_hint", _stable_key(log))

    # Build the prompt and call the LLM
    msg = _prompt.format_messages(
        cases=json.dumps(cases, ensure_ascii=False),
        log=json.dumps(log, ensure_ascii=False),
    )
    llm = _get_llm()
    out = llm.invoke(msg)

    # Parse to the Pydantic model (validates schema)
    return _parser.parse(out.content)
