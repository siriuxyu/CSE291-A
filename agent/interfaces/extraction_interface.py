"""
Extraction Interface - Shared Data Structures
This file contains ONLY shared data structures for Extraction layer.
For implementation, see agent/extraction/
"""

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# ============================================================================
# INTERFACE DATA STRUCTURES
# ============================================================================


class PreferenceType(Enum):
    """Types of user preferences that can be extracted"""
    COMMUNICATION_STYLE = "communication_style"
    DOMAIN_INTEREST = "domain_interest"
    INTERACTION_PATTERN = "interaction_pattern"
    TOOL_PREFERENCE = "tool_preference"
    RESPONSE_FORMAT = "response_format"
    LANGUAGE_PREFERENCE = "language_preference"
    CONTEXT_REFERENCE = "context_reference"


@dataclass
class ExtractedPreference:
    """A single extracted user preference or pattern"""
    preference_type: PreferenceType
    content: str
    confidence_score: float
    evidence: List[str]
    first_seen: datetime
    last_seen: datetime
    frequency: int

    def __post_init__(self):
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("confidence_score must be between 0 and 1")


@dataclass
class SessionSummary:
    """Summary of a conversation session"""
    session_id: str
    user_id: str
    summary_text: str
    preferences: List[ExtractedPreference]
    message_count: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]


@dataclass
class ExtractionConfig:
    """Configuration for context extraction"""
    min_confidence_threshold: float = 0.4
    max_preferences_per_session: int = 8
    summary_max_length: int = 400