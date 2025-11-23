# Interfaces - Shared Definitions


### ⚠️ PUBLIC 
**Storage team:**
- `get_user_contexts()` - Get user's long-term contexts
- `get_documents_by_session()` - Get session messages
- `store_documents_batch()` - Store messages

**Extraction team:**
- `extract_session_summary()` - Extract session insights
- `merge_preferences()` - Merge old and new preferences
- `format_preferences_for_prompt()` - Format preferences as text

### ✅ INTERNAL (Free to modify)
- All other methods (can add/remove/change)
- All `_helper()` methods
- `embedding_service.py` (Storage internal)
- Implementation details

---

## Data Structures (Public)

All dataclasses in this folder are public - don't change fields without coordination:
- `StorageDocument`, `StorageType`, `SearchResult`, `StorageStats`
- `ExtractedPreference`, `SessionSummary`, `PreferenceType`, `ExtractionConfig`
- `MemoryMetrics`, `SessionMetadata`
- All exceptions are **OK to modify**
