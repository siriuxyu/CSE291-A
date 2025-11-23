"""
Vector Storage Implementation

Storage team can choose: ChromaDB, Weaviate, Pinecone, or PostgreSQL with pgvector.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from agent.interfaces import (
    StorageDocument,
    SearchResult,
    StorageStats,
    StorageType,
    StorageError,
    ConnectionError,
)


class VectorStorageBackend:
    """
    Vector Database Storage Backend

    ⚠️ INTERFACE RULES:
    - Public methods (no underscore): These are called by Memory team - DO NOT change signatures
    - Helper methods (with underscore): These are internal - you can freely modify/add/remove
    """

    def __init__(self, collection_name: str = "memory_context", **config):
        """
        Initialize vector storage

        ✅ INTERNAL: You can change implementation details as needed
        """
        self.collection_name = collection_name
        self.config = config
        self.client = None
        self.collection = None

    # ========================================================================
    # PUBLIC INTERFACE - Called by Memory team
    # ⚠️ DO NOT change method signatures without coordinating with Memory team
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize storage backend connection"""
        raise NotImplementedError("TODO: Implement initialize")

    async def store_document(
        self, document: StorageDocument, generate_embedding: bool = True
    ) -> str:
        """Store a document in vector database"""
        raise NotImplementedError("TODO: Implement store_document")

    async def store_documents_batch(
        self, documents: List[StorageDocument], generate_embeddings: bool = True
    ) -> List[str]:
        """Store multiple documents in batch"""
        raise NotImplementedError("TODO: Implement store_documents_batch")

    async def search_similar(
        self,
        query_text: str,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents using vector similarity"""
        raise NotImplementedError("TODO: Implement search_similar")

    async def search_by_embedding(
        self,
        embedding: List[float],
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """Search using pre-computed embedding"""
        raise NotImplementedError("TODO: Implement search_by_embedding")

    async def get_document(
        self, document_id: str, user_id: str
    ) -> Optional[StorageDocument]:
        """Retrieve a specific document by ID"""
        raise NotImplementedError("TODO: Implement get_document")

    async def get_documents_by_session(
        self,
        session_id: str,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
    ) -> List[StorageDocument]:
        """Retrieve all documents for a session"""
        raise NotImplementedError("TODO: Implement get_documents_by_session")

    async def get_user_contexts(
        self,
        user_id: str,
        document_types: Optional[List[StorageType]] = None,
        limit: int = 100,
    ) -> List[StorageDocument]:
        """Retrieve long-term contexts for a user"""
        raise NotImplementedError("TODO: Implement get_user_contexts")

    async def update_document(
        self, document_id: str, updates: Dict[str, Any], user_id: str
    ) -> bool:
        """Update document metadata or content"""
        raise NotImplementedError("TODO: Implement update_document")

    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document"""
        raise NotImplementedError("TODO: Implement delete_document")

    async def delete_session_documents(
        self, session_id: str, user_id: str, preserve_summaries: bool = True
    ) -> int:
        """Delete all documents for a session"""
        raise NotImplementedError("TODO: Implement delete_session_documents")

    async def get_storage_stats(self, user_id: Optional[str] = None) -> StorageStats:
        """Get storage usage statistics"""
        raise NotImplementedError("TODO: Implement get_storage_stats")

    async def close(self) -> None:
        """Close storage connections"""
        pass

    # ========================================================================
    # INTERNAL HELPERS - Storage team only
    # ✅ You can freely add, remove, or modify these methods
    # ========================================================================

    def _get_embedding_service(self):
        """Get embedding service instance"""
        raise NotImplementedError("TODO: Implement _get_embedding_service")

    def _convert_to_storage_document(self, raw_result: Any) -> StorageDocument:
        """Convert raw vector DB result to StorageDocument"""
        raise NotImplementedError("TODO: Implement _convert_to_storage_document")

    # Add more helper methods as needed:
    # def _build_filter(self, ...):
    # def _validate_document(self, ...):
    # etc.
