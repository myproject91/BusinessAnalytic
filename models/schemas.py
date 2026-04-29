from pydantic import BaseModel
from typing import Optional

class AnalyzeResponse(BaseModel):
    profile: dict
    stats: dict
    anomalies: dict
    sentiment: dict
    insight: dict
