import os, json, hashlib
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from schemas import TriageDecision

MODEL = os.getenv("MODEL", "gpt-4o-mini")
BACKEND = os.getenv("RAG_BACKEND", "chroma")

if BACKEND == "pgvector":
    from rag_store_pg import retrieve_similar
else:
    from rag_store_chroma import retrieve_similar

_llm = ChatOpenAI(model=MODEL, temperature=0)
_parser = PydanticOutputParser(pydantic_object=TriageDecision)

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an SRE triage assistant. Use 'similar_cases' when relevant. "
     "Output ONLY valid JSON per schema. Include a stable dedupe_key."),
    ("human",
     "Schema:\n{schema}\n\nSimilar cases:\n{cases}\n\nLog JSON:\n```json\n{log}\n```")
]).partial(schema=_parser.get_format_instructions())

def _stable_key(log: Dict[str, Any]) -> str:
    text = json.dumps({k: log.get(k) for k in ("logName","resource","textPayload","jsonPayload")}, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]

def triage(log: Dict[str, Any]) -> TriageDecision:
    query = (log.get("textPayload")
             or json.dumps(log.get("jsonPayload", {}), ensure_ascii=False)
             or json.dumps(log, ensure_ascii=False))
    cases = retrieve_similar(query, k=4)
    log.setdefault("_dedupe_hint", _stable_key(log))
    msg = _prompt.format_messages(cases=json.dumps(cases, ensure_ascii=False),
                                  log=json.dumps(log, ensure_ascii=False))
    out = _llm.invoke(msg)
    return _parser.parse(out.content)
