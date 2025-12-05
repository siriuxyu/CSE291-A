"""
Memory Module

Provides memory management capabilities:
- ContextMemoryManager: Legacy memory manager (see manager.py)
- LangMem integration: Long-term memory with user-specific namespaces (see langmem_adapter.py)
- Uses VectorStorageBackend (ChromaDB) for persistent storage
"""

from .manager import ContextMemoryManager
from .langmem_adapter import (
    LangMemManager,
    get_langmem_manager,
    get_memory_store,
    get_memory_tools,
    is_langmem_enabled,
    is_storage_available,
)

__all__ = [
    "ContextMemoryManager",
    "LangMemManager",
    "get_langmem_manager",
    "get_memory_store",
    "get_memory_tools",
    "is_langmem_enabled",
    "is_storage_available",
]
