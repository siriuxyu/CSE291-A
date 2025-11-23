"""
Storage Interface - Shared Data Structures

⚠️ IMPORTANT: These are PUBLIC interfaces that other teams depend on!
   - DO NOT change data structure fields without coordinating with Memory team
   - DO NOT remove or rename existing fields
   - You MAY add new optional fields or enum values if needed

This file contains ONLY shared data structures for Storage layer.
For implementation, see agent/storage/
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# ============================================================================
# PUBLIC DATA STRUCTURES - Memory team depends on these
# ============================================================================


class StorageType(Enum):
    """Type of document being stored"""
    SHORT_TERM_MESSAGE = "short_term_message"
    LONG_TERM_CONTEXT = "long_term_context"
    SESSION_SUMMARY = "session_summary"
    USER_PREFERENCE = "user_preference"
    EXTRACTED_FACT = "extracted_fact"


@dataclass
class StorageDocument:
    """Document to be stored in vector database"""
    id: str
    user_id: str
    session_id: Optional[str]
    document_type: StorageType
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    document: StorageDocument
    similarity_score: float
    rank: int


@dataclass
class StorageStats:
    """Storage usage statistics"""
    total_documents: int
    documents_by_type: Dict[StorageType, int]
    total_size_bytes: int
    oldest_document: datetime
    newest_document: datetime
