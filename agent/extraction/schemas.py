from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class PreferenceType(str, Enum):
    STYLE = "style"
    PERSONAL = "personal"
    TOPIC = "topic"
    CONSTRAINT = "constraint"


class UserPreference(BaseModel):
    """Represents a single extracted user preference."""
    preference_type: PreferenceType = Field(..., description="The category of the preference")
    content: str = Field(..., description="The specific preference content (e.g., 'User is a Python developer')")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")


class UserProfileUpdate(BaseModel):
    """The output structure for the preference extraction model."""
    extracted_preferences: List[UserPreference] = Field(
        default_factory=list,
        description="List of new preferences found in the recent conversation"
    )
    reasoning: str = Field(..., description="Brief reasoning for why these preferences were extracted")