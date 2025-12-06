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

        Steps:
        1. Validate session_id and user_id
        2. Initialize session if doesn't exist
        3. Add message to in-memory cache (self._active_sessions)
        4. Update token count estimation
        5. Check if persistence threshold reached
        6. If yes, call _persist_session_to_storage()
        """
        # Validate inputs
        if not session_id or not user_id:
            raise ValueError("session_id and user_id are required")

        # Initialize session if doesn't exist
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = []
            self._session_token_counts[session_id] = 0
            self._session_metadata[session_id] = SessionMetadata(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                message_count=0,
                is_finalized=False,
                tags=[]
            )

        # Add message to in-memory cache
        self._active_sessions[session_id].append(message)

        # Update token count estimation
        token_estimate = self._estimate_tokens(message)
        self._session_token_counts[session_id] += token_estimate

        # Update session metadata
        session_meta = self._session_metadata[session_id]
        session_meta.message_count += 1
        session_meta.last_active = datetime.utcnow()

        # Check if persistence threshold reached
        if self._enable_long_term and self._should_persist(session_id):
            await self._persist_session_to_storage(session_id, user_id)

    def add_message_to_session(
        self,
        session_id: str,
        user_id: str,
        message: BaseMessage,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to session (sync version)

        This is a synchronous wrapper that can be called from non-async contexts.
        It uses asyncio.run() to execute the async version.
        """
        asyncio.run(self.add_message_async(session_id, user_id, message, metadata))

    def get_enhanced_system_prompt(
        self,
        user_id: str,
        base_prompt: str,
        max_context_items: int = 5,
        relevance_threshold: float = 0.7
    ) -> str:
        """
        Enhance system prompt with user's long-term context

        This is called to inject user preferences and context into the system prompt.

        Steps:
        1. Call self._storage.get_user_contexts(user_id) to get long-term contexts
        2. Filter contexts by relevance_threshold if needed
        3. Extract preferences from contexts
        4. Call self._extractor.format_preferences_for_prompt() to format
        5. Append formatted context to base_prompt
        6. Return enhanced prompt
        """
        if not self._enable_long_term:
            return base_prompt

        try:
            # Get user contexts from storage
            contexts = asyncio.run(
                self._storage.get_user_contexts(
                    user_id=user_id,
                    limit=max_context_items
                )
            )

            if not contexts:
                return base_prompt

            # Parse contexts to preferences
            preferences = self._parse_contexts_to_preferences(contexts)

            if not preferences:
                return base_prompt

            # Format preferences for prompt using extractor
            context_text = asyncio.run(
                self._extractor.format_preferences_for_prompt(
                    preferences=preferences,
                    max_items=max_context_items
                )
            )

            if not context_text:
                return base_prompt

            # Append to base prompt
            return f"{base_prompt}\n\n{context_text}"

        except Exception as e:
            # If enhancement fails, return base prompt
            # Don't fail the entire request
            return base_prompt

    async def restore_session_messages(
        self,
        session_id: str,
        user_id: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Restore messages from a previous session

        This is called by server.py to restore a conversation.

        Steps:
        1. Check in-memory cache first (self._active_sessions)
        2. If not in cache, call self._storage.get_documents_by_session(session_id, user_id)
        3. Convert StorageDocument objects to message dicts
        4. Return in format: [{"role": "user", "content": "..."}, ...]
        """
        # Check in-memory cache first
        if session_id in self._active_sessions:
            messages = self._active_sessions[session_id]
            result = []
            for msg in messages:
                # Convert BaseMessage to dict format
                msg_dict = {
                    "role": self._get_message_role(msg),
                    "content": str(msg.content)
                }
                result.append(msg_dict)

            # Limit if requested
            if max_messages and len(result) > max_messages:
                result = result[-max_messages:]

            return result

        # If not in cache, try loading from storage
        try:
            documents = await self._storage.get_documents_by_session(
                session_id=session_id,
                user_id=user_id,
                limit=max_messages
            )

            if not documents:
                return []

            # Convert StorageDocuments to message dicts
            result = []
            for doc in documents:
                # Extract role and content from document
                role = doc.metadata.get("role", "user")
                content = doc.content

                result.append({
                    "role": role,
                    "content": content
                })

            return result

        except Exception as e:
            raise MemoryStorageError(f"Failed to restore session {session_id}: {str(e)}")

    def _get_message_role(self, message: BaseMessage) -> str:
        """Helper to extract role from BaseMessage"""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, ToolMessage):
            return "tool"
        else:
            return "user"  # default

    async def finalize_session(
        self,
        session_id: str,
        user_id: str,
        force: bool = False
    ) -> bool:
        """
        Finalize a session by extracting insights and updating long-term memory

        This is called by server.py when a session ends.
        It coordinates the extraction and storage of user preferences.

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
        # Check if session exists
        if session_id not in self._active_sessions:
            # Try to load from storage
            raise SessionNotFoundError(f"Session {session_id} not found")

        metadata = self._session_metadata.get(session_id)
        if metadata and metadata.is_finalized and not force:
            return True  # Already finalized

        try:
            # Get messages from session
            messages = self._active_sessions[session_id]

            if not messages:
                return False

            # Extract session summary and preferences using the extractor
            from agent.interfaces import ExtractionConfig
            config = ExtractionConfig()
            session_summary = await self._extractor.extract_session_summary(
                messages=messages,
                session_id=session_id,
                user_id=user_id,
                config=config
            )

            # Update long-term context with new preferences
            if session_summary.preferences:
                await self._update_long_term_context(user_id, session_summary)

            # Persist session summary to storage
            await self._persist_session_summary(session_summary)

            # Update session metadata
            if metadata:
                metadata.is_finalized = True
                metadata.end_time = datetime.utcnow()

            return True

        except Exception as e:
            raise MemoryStorageError(f"Failed to finalize session {session_id}: {str(e)}")

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

        Aggregates metrics from in-memory caches and storage.
        If user_id is provided, filters metrics for that user.
        """
        # Calculate short-term metrics from in-memory cache
        total_messages = 0
        total_tokens = 0
        user_sessions = []

        for session_id, messages in self._active_sessions.items():
            metadata = self._session_metadata.get(session_id)

            # Filter by user if specified
            if user_id and metadata and metadata.user_id != user_id:
                continue

            total_messages += len(messages)
            total_tokens += self._session_token_counts.get(session_id, 0)

            if metadata:
                user_sessions.append(session_id)

        # Try to get storage size from storage backend
        storage_bytes = 0
        long_term_items = 0

        try:
            if hasattr(self._storage, 'get_storage_stats'):
                stats = asyncio.run(self._storage.get_storage_stats(user_id=user_id))
                if stats:
                    storage_bytes = stats.total_size_bytes
                    long_term_items = stats.total_documents
        except Exception:
            # If storage stats not available, use estimates
            pass

        return MemoryMetrics(
            short_term_message_count=total_messages,
            short_term_token_estimate=total_tokens,
            long_term_context_items=long_term_items,
            last_updated=datetime.utcnow(),
            storage_size_bytes=storage_bytes
        )

    async def clear_session(
        self,
        session_id: str,
        preserve_summary: bool = True
    ) -> None:
        """
        Clear session messages from memory

        Steps:
        1. Remove from self._active_sessions
        2. If preserve_summary=False, also delete from storage
        """
        # Remove from in-memory caches
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        if session_id in self._session_token_counts:
            del self._session_token_counts[session_id]

        if session_id in self._session_metadata:
            if not preserve_summary:
                # Also remove metadata
                del self._session_metadata[session_id]
            else:
                # Mark as finalized but keep metadata
                self._session_metadata[session_id].is_finalized = True

        # If not preserving summary, delete from storage as well
        if not preserve_summary and hasattr(self._storage, 'delete_session'):
            try:
                await self._storage.delete_session(session_id)
            except Exception:
                # Ignore storage deletion errors
                pass

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

        Steps:
        1. Get messages from self._active_sessions[session_id]
        2. Convert each message to StorageDocument
        3. Call self._storage.store_documents_batch()
        """
        if session_id not in self._active_sessions:
            return

        messages = self._active_sessions[session_id]
        if not messages:
            return

        # Convert messages to StorageDocuments
        documents = []
        for idx, msg in enumerate(messages):
            # Generate embedding if available
            embedding = None
            try:
                if self._embedding:
                    embedding = await self._embedding.embed_text(str(msg.content))
            except Exception:
                pass  # Continue without embedding

            # Create StorageDocument
            doc = StorageDocument(
                id=f"{session_id}_msg_{idx}_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                session_id=session_id,
                document_type=StorageType.SHORT_TERM_MESSAGE,
                content=str(msg.content),
                embedding=embedding,
                metadata={
                    "role": self._get_message_role(msg),
                    "message_index": idx,
                    "timestamp": datetime.utcnow().isoformat()
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            documents.append(doc)

        # Store batch to storage
        try:
            await self._storage.store_documents_batch(documents)
        except Exception as e:
            # Log error but don't fail
            pass

    async def _update_long_term_context(
        self,
        user_id: str,
        session_summary: Any  # SessionSummary from extraction
    ) -> None:
        """
        Update user's long-term context with session insights

        Steps:
        1. Get existing preferences from storage
        2. Call self._extractor.merge_preferences() to merge old and new
        3. Store merged preferences back to storage
        """
        try:
            # Get existing user contexts from storage
            existing_contexts = await self._storage.get_user_contexts(
                user_id=user_id,
                limit=100  # Get recent preferences
            )

            # Parse existing preferences from storage documents
            existing_preferences = self._parse_contexts_to_preferences(existing_contexts)

            # Merge with new preferences using extractor
            merged_preferences = await self._extractor.merge_preferences(
                old_preferences=existing_preferences,
                new_preferences=session_summary.preferences
            )

            # Convert merged preferences back to StorageDocuments and store
            # This will be implemented when Storage team provides the interface
            # For now, just log the merged preferences
            # TODO: Store merged_preferences back to storage
            pass

        except Exception as e:
            # Log error but don't fail the entire finalize process
            pass

    async def _persist_session_summary(self, session_summary: Any) -> None:
        """
        Persist session summary to storage

        Converts SessionSummary to StorageDocument and stores it.
        """
        try:
            # Generate embedding for summary text
            embedding = None
            try:
                if self._embedding:
                    embedding = await self._embedding.embed_text(session_summary.summary_text)
            except Exception:
                pass

            # Create StorageDocument for session summary
            summary_doc = StorageDocument(
                id=f"{session_summary.session_id}_summary_{datetime.utcnow().timestamp()}",
                user_id=session_summary.user_id,
                session_id=session_summary.session_id,
                document_type=StorageType.SESSION_SUMMARY,
                content=session_summary.summary_text,
                embedding=embedding,
                metadata={
                    "message_count": session_summary.message_count,
                    "start_time": session_summary.start_time.isoformat() if session_summary.start_time else None,
                    "end_time": session_summary.end_time.isoformat() if session_summary.end_time else None,
                    "duration_seconds": session_summary.duration_seconds,
                    "preference_count": len(session_summary.preferences)
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            # Store summary
            await self._storage.store_documents_batch([summary_doc])

            # Also store preferences as separate documents
            pref_docs = []
            for pref in session_summary.preferences:
                pref_embedding = None
                try:
                    if self._embedding:
                        pref_embedding = await self._embedding.embed_text(pref.content)
                except Exception:
                    pass

                pref_doc = StorageDocument(
                    id=f"{session_summary.session_id}_pref_{pref.preference_type.value}_{datetime.utcnow().timestamp()}",
                    user_id=session_summary.user_id,
                    session_id=session_summary.session_id,
                    document_type=StorageType.USER_PREFERENCE,
                    content=pref.content,
                    embedding=pref_embedding,
                    metadata={
                        "preference_type": pref.preference_type.value,
                        "confidence_score": pref.confidence_score,
                        "frequency": pref.frequency,
                        "first_seen": pref.first_seen.isoformat(),
                        "last_seen": pref.last_seen.isoformat()
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                pref_docs.append(pref_doc)

            if pref_docs:
                await self._storage.store_documents_batch(pref_docs)

        except Exception as e:
            # Log error but don't fail
            pass

    def _estimate_tokens(self, message: BaseMessage) -> int:
        """
        Estimate token count for a message

        Simple estimation: ~4 characters per token
        """
        content = str(message.content)
        return len(content) // 4

    def _parse_contexts_to_preferences(self, contexts: List[StorageDocument]) -> List:
        """
        Parse StorageDocument contexts to ExtractedPreference objects

        Extracts preference data from document metadata and converts
        StorageDocuments to ExtractedPreference objects.
        """
        from agent.interfaces import ExtractedPreference, PreferenceType

        preferences = []

        for doc in contexts:
            # Only process USER_PREFERENCE type documents
            if doc.document_type != StorageType.USER_PREFERENCE:
                continue

            try:
                # Extract preference data from metadata
                pref_type_str = doc.metadata.get("preference_type", "interaction_pattern")
                confidence = doc.metadata.get("confidence_score", 0.5)
                frequency = doc.metadata.get("frequency", 1)

                # Parse timestamps
                first_seen_str = doc.metadata.get("first_seen")
                last_seen_str = doc.metadata.get("last_seen")

                first_seen = datetime.fromisoformat(first_seen_str) if first_seen_str else doc.created_at
                last_seen = datetime.fromisoformat(last_seen_str) if last_seen_str else doc.updated_at

                # Convert preference type string to enum
                try:
                    pref_type = PreferenceType[pref_type_str.upper()]
                except (KeyError, AttributeError):
                    # Default to INTERACTION_PATTERN if invalid
                    pref_type = PreferenceType.INTERACTION_PATTERN

                # Create ExtractedPreference object
                pref = ExtractedPreference(
                    preference_type=pref_type,
                    content=doc.content,
                    confidence_score=confidence,
                    evidence=[doc.id],  # Use document ID as evidence
                    first_seen=first_seen,
                    last_seen=last_seen,
                    frequency=frequency
                )

                preferences.append(pref)

            except Exception as e:
                # Skip invalid documents
                continue

        return preferences
