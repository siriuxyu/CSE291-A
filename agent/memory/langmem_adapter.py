"""
LangMem integration for long-term memory management.

This module provides:
1. Memory store management with user_id-based namespaces
2. Memory tools (manage/search) that agents can use
3. Integration with existing VectorStorageBackend (ChromaDB)

The key feature is that memories are stored per-user, allowing:
- Each user has their own memory namespace
- Memories persist across sessions via ChromaDB
- Agent can search and manage memories during conversations
"""

from __future__ import annotations

import os
import uuid
import asyncio
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

from agent.utils import get_logger
from agent.interfaces import StorageDocument, StorageType, SearchResult

if TYPE_CHECKING:
    from .langmem_adapter import LangMemManager

logger = get_logger(__name__)

# Try importing langmem components for tools
try:
    from langgraph.store.memory import InMemoryStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    _LANGMEM_AVAILABLE = True
except ImportError:
    InMemoryStore = None  # type: ignore[assignment]
    create_manage_memory_tool = create_search_memory_tool = None  # type: ignore[assignment]
    _LANGMEM_AVAILABLE = False
    logger.warning(
        "LangMem not installed. Run `pip install langmem langchain-openai` to enable memory tools."
    )

# Import existing storage backend
try:
    from agent.storage import VectorStorageBackend, create_embedding_service
    _STORAGE_AVAILABLE = True
except ImportError:
    VectorStorageBackend = None  # type: ignore[assignment]
    create_embedding_service = None  # type: ignore[assignment]
    _STORAGE_AVAILABLE = False
    logger.warning(
        "VectorStorageBackend not available. Install chromadb for persistent storage."
    )


def _bool_from_env(name: str, default: str = "1") -> bool:
    """Parse boolean from environment variable."""
    value = os.environ.get(name, default).strip().lower()
    return value not in {"0", "false", "no", "off"}


def is_langmem_enabled() -> bool:
    """Return True when the LangMem dependency is installed and not disabled via env."""
    return _LANGMEM_AVAILABLE and _bool_from_env("LANGMEM_ENABLED", "1")


def is_storage_available() -> bool:
    """Return True when VectorStorageBackend is available."""
    return _STORAGE_AVAILABLE


def _run_async(coro):
    """Helper to run async coroutine from sync context, handling nested event loops."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context - use nest_asyncio or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a new event loop in the thread
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()
                
                future = executor.submit(run_in_new_loop)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def create_chromadb_memory_tools(user_id: str, manager: "LangMemManager") -> List:
    """
    Create memory tools that use ChromaDB for persistent storage.
    
    These tools allow the agent to search and store memories in ChromaDB,
    which is the same storage used by the /memory API endpoints.
    
    Args:
        user_id: The user identifier for memory namespace
        manager: The LangMemManager instance
        
    Returns:
        List of tools [search_memory, store_memory]
    """
    
    @tool
    def search_memory(query: str) -> str:
        """
        Search your long-term memory for relevant information about past conversations.
        
        Use this tool when you need to recall:
        - User preferences or personal information
        - Past conversation topics
        - Previously stored facts or context
        
        Args:
            query: Natural language query describing what you're looking for
            
        Returns:
            Relevant memories found, or a message if no memories match
        """
        try:
            results = _run_async(manager.search_user_memories(user_id, query, limit=10))
            
            if not results:
                return f"No memories found matching: '{query}'"
            
            # Format results for the agent
            formatted = []
            for i, r in enumerate(results, 1):
                content = r.get("content", "")
                score = r.get("score", 0)
                formatted.append(f"{i}. [Score: {score:.2f}] {content}")
            
            return f"Found {len(results)} relevant memories:\n" + "\n".join(formatted)
            
        except Exception as e:
            logger.error(f"Error in search_memory tool: {e}")
            return f"Error searching memory: {str(e)}"
    
    @tool
    def store_memory(content: str) -> str:
        """
        Store important information in long-term memory for future recall.
        
        Use this tool to save:
        - User preferences (name, likes, dislikes)
        - Important facts shared by the user
        - Context that should be remembered across sessions
        
        Args:
            content: The information to store (should be clear and descriptive)
            
        Returns:
            Confirmation of storage or error message
        """
        try:
            # Generate a unique key for this memory
            memory_key = f"memory_{uuid.uuid4().hex[:8]}"
            
            success = _run_async(manager.store_user_memory(user_id, memory_key, content))
            
            if success:
                return f"Successfully stored memory: '{content[:100]}...'" if len(content) > 100 else f"Successfully stored memory: '{content}'"
            else:
                return "Failed to store memory - storage may be unavailable"
                
        except Exception as e:
            logger.error(f"Error in store_memory tool: {e}")
            return f"Error storing memory: {str(e)}"
    
    return [search_memory, store_memory]


class LangMemManager:
    """
    Manages LangMem store and tools for user-specific memory namespaces.
    
    This class provides:
    - A single shared store instance (using VectorStorageBackend/ChromaDB)
    - User-specific memory tools with dynamic namespaces
    - Methods to search/manage memories for specific users
    """
    
    _instance: Optional["LangMemManager"] = None
    
    def __new__(cls):
        """Singleton pattern to ensure one store instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the LangMem manager with storage backends."""
        if self._initialized:
            return
            
        self._langmem_store = None  # For LangMem tools
        self._vector_storage = None  # For ChromaDB persistence
        self._user_tools_cache: Dict[str, List] = {}
        self._storage_initialized = False
        
        # Initialize stores
        if is_langmem_enabled():
            self._initialize_langmem_store()
        
        if is_storage_available():
            self._initialize_vector_storage()
        
        self._initialized = True
    
    def _initialize_langmem_store(self):
        """Initialize the LangMem InMemoryStore for tools."""
        if InMemoryStore is None:
            logger.warning("LangMem InMemoryStore unavailable")
            return

        dims = int(os.environ.get("LANGMEM_EMBED_DIM", "1536"))
        embed_model = os.environ.get(
            "LANGMEM_EMBED_MODEL",
            "openai:text-embedding-3-small",
        )

        try:
            self._langmem_store = InMemoryStore(
                index={
                    "dims": dims,
                    "embed": embed_model,
                }
            )
            logger.info(
                "LangMem InMemoryStore initialized",
                extra={
                    "function": "LangMemManager._initialize_langmem_store",
                    "details": {"dims": dims, "embed": embed_model},
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize LangMem store: {e}")
            self._langmem_store = None
    
    def _initialize_vector_storage(self):
        """Initialize VectorStorageBackend for persistent storage."""
        if VectorStorageBackend is None:
            logger.warning("VectorStorageBackend unavailable")
            return
        
        try:
            persist_path = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_db_data")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY not set - VectorStorageBackend requires it for embeddings")
                return
            
            self._vector_storage = VectorStorageBackend(
                collection_name="langmem_memories",
                persist_path=persist_path,
                embedding_provider="openai",
                openai_api_key=openai_api_key,
            )
            
            # Note: Actual initialization happens lazily in ensure_storage_initialized()
            logger.info(
                "VectorStorageBackend configured (will initialize on first use)",
                extra={
                    "function": "LangMemManager._initialize_vector_storage",
                    "details": {"persist_path": persist_path},
                },
            )
        except Exception as e:
            logger.error(f"Failed to configure VectorStorageBackend: {e}")
            self._vector_storage = None
    
    async def _async_init_storage(self):
        """Async initialization of vector storage."""
        if self._vector_storage and not self._storage_initialized:
            try:
                await self._vector_storage.initialize()
                self._storage_initialized = True
                logger.info("VectorStorageBackend initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize VectorStorageBackend: {e}")
    
    async def ensure_storage_initialized(self):
        """Ensure storage is initialized before use."""
        if self._vector_storage and not self._storage_initialized:
            await self._async_init_storage()
    
    @property
    def langmem_store(self):
        """Get the LangMem store instance (for graph compilation)."""
        return self._langmem_store
    
    @property
    def vector_storage(self):
        """Get the VectorStorageBackend instance."""
        return self._vector_storage
    
    @property
    def is_available(self) -> bool:
        """Check if any memory backend is available."""
        return self._langmem_store is not None or self._vector_storage is not None
    
    @property
    def has_persistent_storage(self) -> bool:
        """Check if persistent storage (ChromaDB) is available."""
        return self._vector_storage is not None and self._storage_initialized
    
    def get_user_namespace(self, user_id: str) -> Tuple[str, ...]:
        """
        Generate a namespace tuple for a specific user.
        
        Args:
            user_id: The user identifier
            
        Returns:
            A tuple representing the namespace (e.g., ("memories", "user123"))
        """
        base_namespace = os.environ.get("LANGMEM_BASE_NAMESPACE", "memories")
        return (base_namespace, user_id)
    
    def get_tools_for_user(self, user_id: str) -> List:
        """
        Get memory tools configured for a specific user.
        
        These tools use ChromaDB (VectorStorageBackend) for persistent storage,
        NOT LangMem's InMemoryStore.
        
        Args:
            user_id: The user identifier
            
        Returns:
            List of tools [search_memory_tool, store_memory_tool] for this user
        """
        if not is_langmem_enabled():
            return []
        
        # Check cache first
        if user_id in self._user_tools_cache:
            return self._user_tools_cache[user_id]
        
        # Create custom tools that use ChromaDB
        tools = create_chromadb_memory_tools(user_id, self)
        
        # Cache the tools
        self._user_tools_cache[user_id] = tools
        
        logger.debug(
            f"Created ChromaDB memory tools for user {user_id}",
            extra={
                "function": "get_tools_for_user",
                "details": {"user_id": user_id, "tool_count": len(tools)},
            },
        )
        
        return tools
    
    def clear_user_tools_cache(self, user_id: Optional[str] = None):
        """Clear cached tools for a user or all users."""
        if user_id:
            self._user_tools_cache.pop(user_id, None)
        else:
            self._user_tools_cache.clear()
    
    # ========================================================================
    # VectorStorageBackend-based Memory Operations
    # ========================================================================
    
    async def search_user_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.4,  # Lowered to return all matches
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a specific user using ChromaDB.
        
        Args:
            user_id: The user identifier
            query: Search query string
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of memory items matching the query
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            logger.warning("Persistent storage not available for search")
            return []
        
        try:
            results: List[SearchResult] = await self._vector_storage.search_similar(
                query_text=query,
                user_id=user_id,
                document_types=None,  # Search all document types
                top_k=limit,
                similarity_threshold=similarity_threshold,
            )
            
            return [
                {
                    "key": r.document.id,
                    "content": r.document.content,
                    "score": r.similarity_score,
                    "metadata": r.document.metadata,
                    "created_at": r.document.created_at.isoformat(),
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error searching memories for user {user_id}: {e}")
            return []
    
    async def store_user_memory(
        self,
        user_id: str,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_type: StorageType = StorageType.LONG_TERM_CONTEXT,
    ) -> bool:
        """
        Store a memory for a specific user in ChromaDB.
        
        Args:
            user_id: The user identifier
            key: Memory key/identifier (used as document ID)
            content: Memory content text
            metadata: Optional additional metadata
            document_type: Type of document to store
            
        Returns:
            True if successful, False otherwise
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            logger.warning("Persistent storage not available for store")
            return False
        
        try:
            now = datetime.now()
            doc = StorageDocument(
                id=key,
                user_id=user_id,
                session_id=None,
                document_type=document_type,
                content=content,
                embedding=None,  # Will be generated by storage backend
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            
            await self._vector_storage.store_document(doc, generate_embedding=True)
            
            logger.debug(
                f"Stored memory for user {user_id}",
                extra={
                    "function": "store_user_memory",
                    "details": {"user_id": user_id, "key": key},
                },
            )
            return True
        except Exception as e:
            logger.error(f"Error storing memory for user {user_id}: {e}")
            return False
    
    async def get_user_memory(
        self,
        user_id: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory for a user.
        
        Args:
            user_id: The user identifier
            key: Memory key/identifier
            
        Returns:
            Memory content or None if not found
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return None
        
        try:
            doc = await self._vector_storage.get_document(key, user_id)
            if doc:
                return {
                    "key": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat(),
                }
            return None
        except Exception as e:
            logger.error(f"Error getting memory for user {user_id}: {e}")
            return None
    
    async def delete_user_memory(
        self,
        user_id: str,
        key: str,
    ) -> bool:
        """
        Delete a specific memory for a user.
        
        Args:
            user_id: The user identifier
            key: Memory key/identifier
            
        Returns:
            True if successful, False otherwise
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return False
        
        try:
            success = await self._vector_storage.delete_document(key, user_id)
            if success:
                logger.debug(
                    f"Deleted memory for user {user_id}",
                    extra={
                        "function": "delete_user_memory",
                        "details": {"user_id": user_id, "key": key},
                    },
                )
            return success
        except Exception as e:
            logger.error(f"Error deleting memory for user {user_id}: {e}")
            return False
    
    async def list_user_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List all memories for a user.
        
        Args:
            user_id: The user identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory items
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return []
        
        try:
            docs = await self._vector_storage.get_user_contexts(
                user_id=user_id,
                document_types=[StorageType.LONG_TERM_CONTEXT, StorageType.USER_PREFERENCE],
                limit=limit,
            )
            
            return [
                {
                    "key": doc.id,
                    "content": doc.content,
                    "type": doc.document_type.value,
                    "metadata": doc.metadata,
                    "created_at": doc.created_at.isoformat(),
                }
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Error listing memories for user {user_id}: {e}")
            return []
    
    async def clear_user_memories(self, user_id: str) -> bool:
        """
        Clear all memories for a specific user.
        
        Args:
            user_id: The user identifier
            
        Returns:
            True if successful, False otherwise
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return False
        
        try:
            # Get all user memories and delete them
            docs = await self._vector_storage.get_user_contexts(
                user_id=user_id,
                limit=1000,
            )
            
            deleted_count = 0
            for doc in docs:
                if await self._vector_storage.delete_document(doc.id, user_id):
                    deleted_count += 1
            
            # Clear tool cache for this user
            self.clear_user_tools_cache(user_id)
            
            logger.info(
                f"Cleared {deleted_count} memories for user {user_id}",
                extra={
                    "function": "clear_user_memories",
                    "details": {"user_id": user_id, "count": deleted_count},
                },
            )
            return True
        except Exception as e:
            logger.error(f"Error clearing memories for user {user_id}: {e}")
            return False
    
    async def store_conversation_context(
        self,
        user_id: str,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store conversation context/summary for a session.
        
        Args:
            user_id: The user identifier
            session_id: The session identifier
            content: Context/summary content
            metadata: Optional additional metadata
            
        Returns:
            True if successful
        """
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return False
        
        try:
            now = datetime.now()
            doc_id = f"ctx_{user_id}_{session_id}_{uuid.uuid4().hex[:8]}"
            
            doc = StorageDocument(
                id=doc_id,
                user_id=user_id,
                session_id=session_id,
                document_type=StorageType.LONG_TERM_CONTEXT,
                content=content,
                embedding=None,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            
            await self._vector_storage.store_document(doc, generate_embedding=True)
            
            logger.debug(
                f"Stored conversation context for user {user_id}, session {session_id}",
                extra={
                    "function": "store_conversation_context",
                    "details": {"user_id": user_id, "session_id": session_id},
                },
            )
            return True
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            return False
    
    async def get_storage_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get storage statistics."""
        await self.ensure_storage_initialized()
        
        if not self.has_persistent_storage:
            return {"error": "Persistent storage not available"}
        
        try:
            stats = await self._vector_storage.get_storage_stats(user_id)
            return {
                "total_documents": stats.total_documents,
                "documents_by_type": {k.value: v for k, v in stats.documents_by_type.items()},
                "total_size_bytes": stats.total_size_bytes,
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {"error": str(e)}


# Global singleton instance
_langmem_manager: Optional[LangMemManager] = None


def get_langmem_manager() -> LangMemManager:
    """Get the global LangMem manager instance."""
    global _langmem_manager
    if _langmem_manager is None:
        _langmem_manager = LangMemManager()
    return _langmem_manager


# Convenience functions for backward compatibility
def get_memory_store():
    """Get the LangMem store instance (for graph compilation)."""
    return get_langmem_manager().langmem_store


def get_memory_tools() -> List:
    """
    Get generic memory tools (not user-specific).

    For user-specific tools, use get_langmem_manager().get_tools_for_user(user_id)
    
    Note: Generic tools use a default "anonymous" user_id.
    """
    if not is_langmem_enabled():
        return []

    manager = get_langmem_manager()
    return create_chromadb_memory_tools("anonymous", manager)
