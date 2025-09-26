from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TriageRequest(BaseModel):
    log: Dict[str, Any]

class TriageDecision(BaseModel):
    severity: str = Field(..., description="ERROR/WARNING/CRITICAL etc")
    service: str = Field(..., description="Service/component inferred")
    dedupe_key: str
    probable_cause: str
    priority: str = Field(..., description="P1..P5")
    runbook: str
    notify_channels: List[str] = []
    suggest_cmds: List[str] = []
