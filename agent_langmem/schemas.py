from datetime import datetime
from typing import List
from pydantic import BaseModel

class UserPreference(BaseModel):
    """Store the user's preference"""
    preference: str
    context: str | None = None
    created_at: datetime
    updated_at: datetime
    frequency: int

class WikiFact(BaseModel):
    """Store a fact extracted from a convinced source"""
    fact: str
    source: str
    created_at: datetime
    updated_at: datetime

class SessionSummary(BaseModel):
    """Store a summary of the user's session"""
    summary: str
    session_id: str
    created_at: datetime
    updated_at: datetime