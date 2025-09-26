# schemas.py
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class TriageDecision(BaseModel):
    severity: str
    service: str
    priority: str
    probable_cause: str
    runbook: str
    dedupe_key: str

    # Lists default to empty so n8n never breaks on None
    notify_channels: List[str] = Field(default_factory=list)
    suggest_cmds: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

    # Nice-to-have context
    similar_cases: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None
