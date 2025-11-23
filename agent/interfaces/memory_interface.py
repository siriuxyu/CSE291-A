"""
Memory Interface - Shared Data Structures

This file contains ONLY shared data structures for Memory Core.
For implementation, see agent/memory/manager.py
"""

from typing import List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryMetrics:
    """Memory usage and performance metrics"""
    short_term_message_count: int
    short_term_token_estimate: int
    long_term_context_items: int
    last_updated: datetime
    storage_size_bytes: int


@dataclass
class SessionMetadata:
    """Session metadata"""
    session_id: str
    user_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    is_finalized: bool
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Note: MemoryManager class removed - see agent/memory/manager.py for implementation
# The implementation class ContextMemoryManager should have these methods:
#   - add_message_async(session_id, user_id, message, metadata)
#   - add_message_to_session(session_id, user_id, message, metadata)
#   - get_enhanced_system_prompt(user_id, base_prompt, max_context_items, relevance_threshold)
#   - restore_session_messages(session_id, user_id, max_messages)
#   - finalize_session(session_id, force)
#   - get_session_metadata(session_id)
#   - get_memory_metrics(user_id)
#   - clear_session(session_id, preserve_summary)
