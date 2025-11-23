"""
Context Extractor Implementation
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage

from agent.interfaces import (
    ExtractedPreference,
    SessionSummary,
    ExtractionConfig,
    PreferenceType,
)


class ContextExtractor:
    """
    Context extractor for extracting user preferences and session summaries.
    """

    def __init__(self):
        self.stop_words = self._load_stop_words()

    # ========================================================================
    # PUBLIC INTERFACE
    # DO NOT change signatures without coordination
    # ========================================================================

    async def extract_session_summary(
        self,
        messages: List[BaseMessage],
        session_id: str,
        user_id: str,
        config: Optional[ExtractionConfig] = None
    ) -> SessionSummary:
        """Extract summary and preferences from a session"""
        config = config or ExtractionConfig()

        preferences = await self.extract_preferences(messages, user_id, config)
        summary_text = await self.generate_summary(messages, config.summary_max_length)

        return SessionSummary(
            session_id=session_id,
            user_id=user_id,
            summary_text=summary_text,
            preferences=preferences,
            message_count=len(messages),
            start_time=None,
            end_time=None,
            duration_seconds=None,
        )

    async def extract_preferences(
        self,
        messages: List[BaseMessage],
        user_id: str,
        config: Optional[ExtractionConfig] = None
    ) -> List[ExtractedPreference]:
        """Extract user preferences from messages"""
        config = config or ExtractionConfig()
        prefs: List[ExtractedPreference] = []

        for msg in messages:
            if not isinstance(msg, HumanMessage):
                continue
            text = str(msg.content).lower()
            if not text:
                continue

            hints = [
                ("prefer", 0.9),
                ("like", 0.8),
                ("want", 0.7),
                ("usually", 0.6),
                ("interested in", 0.8),
            ]

            matched = [h for h in hints if h[0] in text]
            if not matched:
                continue

            pref_type = self._classify_preference_type(text)
            confidence = min(1.0, max(score for _, score in matched))
            evidence = [getattr(msg, "id", None) or text[:50]]

            prefs.append(
                ExtractedPreference(
                    preference_type=pref_type,
                    content=msg.content if isinstance(msg.content, str) else str(msg.content),
                    confidence_score=confidence,
                    evidence=evidence,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    frequency=1,
                )
            )

        prefs = [p for p in prefs if p.confidence_score >= config.min_confidence_threshold]
        return prefs[: config.max_preferences_per_session]

    async def merge_preferences(
        self,
        old_preferences: List[ExtractedPreference],
        new_preferences: List[ExtractedPreference]
    ) -> List[ExtractedPreference]:
        """Merge old and new preferences, updating confidence scores"""
        merged: Dict[str, ExtractedPreference] = {}

        def _key(p: ExtractedPreference) -> str:
            return f"{p.preference_type.value}:{p.content.strip().lower()}"

        for pref in old_preferences + new_preferences:
            k = _key(pref)
            if k not in merged:
                merged[k] = pref
                continue

            existing = merged[k]
            total_freq = existing.frequency + pref.frequency
            existing.confidence_score = (
                existing.confidence_score * existing.frequency + pref.confidence_score * pref.frequency
            ) / total_freq
            existing.frequency = total_freq
            existing.last_seen = max(existing.last_seen, pref.last_seen)

        merged_list = sorted(merged.values(), key=lambda p: p.confidence_score, reverse=True)
        return merged_list

    def format_preferences_for_prompt(
        self,
        preferences: List[ExtractedPreference],
        max_items: int = 5
    ) -> str:
        """Format preferences as text for system prompt"""
        if not preferences:
            return ""

        lines = ["## User Context"]
        for pref in sorted(preferences, key=lambda p: p.confidence_score, reverse=True)[:max_items]:
            lines.append(f"- {pref.preference_type.name.title()}: {pref.content}")
        return "\n".join(lines)

    async def generate_summary(
        self,
        messages: List[BaseMessage],
        max_length: int = 400
    ) -> str:
        """Generate a text summary of the session"""
        if not messages:
            return "No messages to summarize."

        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        first = user_msgs[0].content if user_msgs else messages[0].content
        last = user_msgs[-1].content if user_msgs else messages[-1].content

        summary = f"User started with: {first}. Latest user note: {last}."
        summary = str(summary)
        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."
        return summary

    # ========================================================================
    # HELPERS - examples
    # ========================================================================

    def _load_stop_words(self) -> set:
        """Load stop words for text processing"""
        return {
            "the", "is", "are", "was", "were", "a", "an", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "i", "you", "he", "she", "it", "we", "they", "this", "that"
        }

    def _classify_preference_type(self, content: str) -> PreferenceType:
        """Classify text into a preference type"""
        text = content.lower()
        if any(k in text for k in ["concise", "brief", "detailed", "explain"]):
            return PreferenceType.COMMUNICATION_STYLE
        if any(k in text for k in ["code", "python", "api", "math", "ml", "ai"]):
            return PreferenceType.DOMAIN_INTEREST
        if any(k in text for k in ["bullet", "list", "json", "table"]):
            return PreferenceType.RESPONSE_FORMAT
        if any(k in text for k in ["tool", "calculator", "browser", "search"]):
            return PreferenceType.TOOL_PREFERENCE
        if any(k in text for k in ["english", "chinese", "spanish", "french"]):
            return PreferenceType.LANGUAGE_PREFERENCE
        return PreferenceType.INTERACTION_PATTERN