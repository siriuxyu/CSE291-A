"""
Storage Module

Export vector storage and embedding service implementations.
"""

from .vector_storage import VectorStorageBackend
from .embedding_service import (
    OpenAIEmbeddingService,
    SentenceTransformerEmbeddingService,
    create_embedding_service
)

__all__ = [
    "VectorStorageBackend",
    "OpenAIEmbeddingService",
    "SentenceTransformerEmbeddingService",
    "create_embedding_service",
]
