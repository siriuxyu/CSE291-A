"""
Extraction Module

Export context extractor implementations and preference extraction utilities.
"""

from .extractor import PatternBasedContextExtractor
from .preference import extract_preferences
from .schemas import UserPreference, UserProfileUpdate, PreferenceType
from .prompts import PREFERENCE_EXTRACTION_SYSTEM_PROMPT

__all__ = [
    "PatternBasedContextExtractor",
    "extract_preferences",
    "UserPreference",
    "UserProfileUpdate",
    "PreferenceType",
    "PREFERENCE_EXTRACTION_SYSTEM_PROMPT"
]
