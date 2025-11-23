"""
Memory Core Interfaces - Shared Definitions

This package contains ONLY shared data structures, enums, and exceptions.
Implementation classes are in their respective modules:
- agent/memory/manager.py (ContextMemoryManager)
- agent/storage/vector_storage.py (VectorStorageBackend)
- agent/storage/embedding_service.py (OpenAIEmbeddingService, etc.)
- agent/extraction/extractor.py (PatternBasedContextExtractor)

Usage example:
    from agent.interfaces import (
        # Data structures
        StorageDocument,
        ExtractedPreference,
        SessionSummary,

        # Enums
        StorageType,
        PreferenceType,

        # Exceptions
        MemoryError,
        StorageError,
    )
"""

# Memory data structures
from .memory_interface import MemoryMetrics, SessionMetadata

# Storage data structures
from .storage_interface import (
    StorageDocument,
    SearchResult,
    StorageStats,
    StorageType
)

# Extraction data structures
from .extraction_interface import (
    ExtractedPreference,
    SessionSummary,
    ExtractionConfig,
    PreferenceType
)

# Exception classes
from .exceptions import (
    MemoryError,
    MemoryStorageError,
    SessionNotFoundError,
    ExtractionError,
    EmbeddingError,
    ValidationError,
    StorageError,
    ConnectionError,
    ConfigurationError
)

__all__ = [
    # Memory data structures
    "MemoryMetrics",
    "SessionMetadata",

    # Storage data structures
    "StorageDocument",
    "SearchResult",
    "StorageStats",
    "StorageType",

    # Extraction data structures
    "ExtractedPreference",
    "SessionSummary",
    "ExtractionConfig",
    "PreferenceType",

    # Exceptions
    "MemoryError",
    "MemoryStorageError",
    "SessionNotFoundError",
    "ExtractionError",
    "EmbeddingError",
    "ValidationError",
    "StorageError",
    "ConnectionError",
    "ConfigurationError",
]
