"""
Memory Core Implementation

This is the main implementation file for Memory Core team.
Implements the MemoryManager interface to coordinate between Storage and Extraction layers.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage

from agent.interfaces import (
    MemoryMetrics,
    SessionMetadata,
    StorageDocument,
    StorageType,
    SessionNotFoundError,
    MemoryStorageError,
)


class ContextMemoryManager:
    """
    Main Memory Core implementation

    Coordinates between:
    - Storage layer (Vector DB)
    - Extraction layer (Preference extraction)
    - Agent layer (graph.py, server.py)
    """

    def __init__(
        self,
        storage: Any,  # VectorStorageBackend from agent.storage
        extractor: Any,  # PatternBasedContextExtractor from agent.extraction
        embedding: Any,  # EmbeddingService from agent.storage
        max_short_term_messages: int = 100,
        max_short_term_tokens: int = 50000,
        enable_long_term: bool = True
    ):
        """
        Initialize Memory Core

        Args:
            storage: Storage backend implementation (from Storage team)
            extractor: Context extractor implementation (from Extraction team)
            embedding: Embedding service implementation (from Storage team)
            max_short_term_messages: Max messages before persistence
            max_short_term_tokens: Max tokens before persistence
            enable_long_term: Whether to enable long-term memory
        """
        self._storage = storage
        self._extractor = extractor
        self._embedding = embedding
        self._max_short_term_messages = max_short_term_messages
        self._max_short_term_tokens = max_short_term_tokens
        self._enable_long_term = enable_long_term

        # In-memory caches
        self._active_sessions: Dict[str, List[BaseMessage]] = {}
        self._session_metadata: Dict[str, SessionMetadata] = {}
        self._session_token_counts: Dict[str, int] = {}

    async def add_message_async(
        self,
        session_id: str,
        user_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to session's short-term memory (async version)

        TODO: Implement this method
        Steps:
        1. Validate session_id and user_id
        2. Initialize session if doesn't exist
        3. Add message to in-memory cache (self._active_sessions)
        4. Update token count estimation
        5. Check if persistence threshold reached
        6. If yes, call _persist_session_to_storage()
        """
        # TODO: Memory Core team - implement this
        raise NotImplementedError("TODO: Implement add_message_async")

    def add_message_to_session(
        self,
        session_id: str,
        user_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to session (sync version)

        TODO: Implement this method
        This is called by agent/graph.py (line 176-192)
        Use asyncio.run() or asyncio.create_task() to call add_message_async
        """
        # TODO: Memory Core team - implement this
        # Hint: asyncio.run(self.add_message_async(session_id, user_id, message, metadata))
        raise NotImplementedError("TODO: Implement add_message_to_session")

    def get_enhanced_system_prompt(
        self,
        user_id: str,
        base_prompt: str,
        max_context_items: int = 5,
        relevance_threshold: float = 0.7
    ) -> str:
        """
        Enhance system prompt with user's long-term context

        TODO: Implement this method
        This is called by agent/graph.py (line 94-105)

        Steps:
        1. Call self._storage.get_user_contexts(user_id) to get long-term contexts
        2. Filter contexts by relevance_threshold if needed
        3. Extract preferences from contexts
        4. Call self._extractor.format_preferences_for_prompt() to format
        5. Append formatted context to base_prompt
        6. Return enhanced prompt
        """
        # TODO: Memory Core team - implement this
        # Example skeleton:
        # contexts = asyncio.run(self._storage.get_user_contexts(user_id, limit=max_context_items))
        # if not contexts:
        #     return base_prompt
        # preferences = self._parse_contexts_to_preferences(contexts)
        # context_text = self._extractor.format_preferences_for_prompt(preferences, max_context_items)
        # return f"{base_prompt}\n\n{context_text}"
        raise NotImplementedError("TODO: Implement get_enhanced_system_prompt")

    async def restore_session_messages(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Restore messages from a previous session

        TODO: Implement this method
        This is called by server.py (line 83-106)

        Steps:
        1. Check in-memory cache first (self._active_sessions)
        2. If not in cache, call self._storage.get_documents_by_session(session_id, user_id)
        3. Convert StorageDocument objects to message dicts
        4. Return in format: [{"role": "user", "content": "..."}, ...]
        """
        # TODO: Memory Core team - implement this
        raise NotImplementedError("TODO: Implement restore_session_messages")

    async def finalize_session(
        self,
        session_id: str,
        force: bool = False
    ) -> bool:
        """
        Finalize a session by extracting insights and updating long-term memory

        TODO: Implement this method
        This is called by server.py (line 232)

        Steps:
        1. Check if session exists and not already finalized (unless force=True)
        2. Get messages from self._active_sessions or storage
        3. Call self._extractor.extract_session_summary() to extract insights
        4. Call self._update_long_term_context() to merge preferences
        5. Call self._persist_session_summary() to save summary
        6. Update session metadata as finalized
        7. Optionally clear from in-memory cache
        8. Return True if successful
        """
        # TODO: Memory Core team - implement this
        raise NotImplementedError("TODO: Implement finalize_session")

    def get_session_metadata(
        self,
        session_id: str
    ) -> Optional[SessionMetadata]:
        """
        Get session metadata without loading all messages

        TODO: Implement this method
        Check self._session_metadata cache first
        """
        # TODO: Memory Core team - implement this
        return self._session_metadata.get(session_id)

    def get_memory_metrics(
        self,
        user_id: Optional[str] = None
    ) -> MemoryMetrics:
        """
        Get memory usage metrics

        TODO: Implement this method
        Aggregate metrics from in-memory caches and storage
        """
        # TODO: Memory Core team - implement this
        raise NotImplementedError("TODO: Implement get_memory_metrics")

    async def clear_session(
        self,
        session_id: str,
        preserve_summary: bool = True
    ) -> None:
        """
        Clear session messages from memory

        TODO: Implement this method
        Steps:
        1. Remove from self._active_sessions
        2. If preserve_summary=False, also delete from storage
        """
        # TODO: Memory Core team - implement this
        raise NotImplementedError("TODO: Implement clear_session")

    # ========================================================================
    # HELPER METHODS - Implement these to support the main methods above
    # ========================================================================

    def _should_persist(self, session_id: str) -> bool:
        """
        Check if session should be persisted to storage

        TODO: Implement this helper
        Check if message count or token count exceeds thresholds
        """
        # TODO: Implement
        if session_id not in self._active_sessions:
            return False

        message_count = len(self._active_sessions[session_id])
        token_count = self._session_token_counts.get(session_id, 0)

        return (message_count >= self._max_short_term_messages or
                token_count >= self._max_short_term_tokens)

    async def _persist_session_to_storage(self, session_id: str, user_id: str) -> None:
        """
        Persist session messages to storage

        TODO: Implement this helper
        Steps:
        1. Get messages from self._active_sessions[session_id]
        2. Convert each message to StorageDocument
        3. Call self._storage.store_documents_batch()
        """
        # TODO: Implement
        pass

    async def _update_long_term_context(
        self,
        user_id: str,
        session_summary: Any  # SessionSummary from extraction
    ) -> None:
        """
        Update user's long-term context with session insights

        TODO: Implement this helper
        Steps:
        1. Get existing preferences from storage
        2. Call self._extractor.merge_preferences() to merge old and new
        3. Store merged preferences back to storage
        """
        # TODO: Implement
        pass

    async def _persist_session_summary(self, session_summary: Any) -> None:
        """
        Persist session summary to storage

        TODO: Implement this helper
        Convert SessionSummary to StorageDocument and store
        """
        # TODO: Implement
        pass

    def _estimate_tokens(self, message: BaseMessage) -> int:
        """
        Estimate token count for a message

        TODO: Implement this helper
        Simple estimation: ~4 characters per token
        """
        # TODO: Implement
        content = str(message.content)
        return len(content) // 4

    def _parse_contexts_to_preferences(self, contexts: List[StorageDocument]) -> List:
        """
        Parse StorageDocument contexts to ExtractedPreference objects

        TODO: Implement this helper
        Extract preference data from document metadata
        """
        # TODO: Implement
        return []
