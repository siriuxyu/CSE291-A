# Session Management

## Overview

The Session Management system tracks user conversation sessions, supporting automatic session management, manual control, and persistent storage.

## Architecture

```
Client Request
    ↓
Server (session allocation logic)
    ↓
Storage (persist session information)
    ↓
Memory Manager (manage messages and preference extraction)
```

## Core Components

### 1. Session ID Generation and Management

**Location**: [`server.py:get_or_create_session()`](../server.py#L28)

**Strategies**:
- **Automatic Management**: If user doesn't provide `session_id`, server auto-assigns based on time window (default 30 minutes)
- **Manual Control**: User can explicitly provide `session_id` to continue existing conversation
- **Force New**: User can set `force_new_session=True` to explicitly start a new session

**Persistence**:
- Session information stored in **Storage backend** (not in-memory)
- Uses `StorageDocument` type `SESSION_SUMMARY`
- Metadata marked with `is_session_tracker: true` to distinguish from regular session summaries

### 2. Data Flow

#### 2.1 Create New Session

```python
POST /invoke
{
    "userid": "alice",
    "messages": [{"role": "user", "content": "Hello"}]
}

→ get_or_create_session()
  → Check Storage for active sessions
  → None found → Create new session_id: "alice_a1b2c3d4"
  → _create_session_record() stores to Storage

→ Response
{
    "user_id": "alice",
    "session_id": "alice_a1b2c3d4",  # Client saves this ID
    "final_response": "..."
}
```

#### 2.2 Continue Existing Session

```python
POST /invoke
{
    "userid": "alice",
    "session_id": "alice_a1b2c3d4",  # Use previously returned session_id
    "messages": [{"role": "user", "content": "Continue conversation"}]
}

→ get_or_create_session()
  → Use provided session_id
  → _update_session_activity() updates last activity time

→ Graph uses this session_id as thread_id
  → LangGraph automatically loads message history
```

#### 2.3 Force New Session

```python
POST /invoke
{
    "userid": "alice",
    "force_new_session": true,  # Force create new session
    "messages": [{"role": "user", "content": "New topic"}]
}

→ get_or_create_session()
  → Ignore existing sessions
  → Create new session_id: "alice_x9y8z7w6"

→ Return new session_id
{
    "session_id": "alice_x9y8z7w6"  # New session
}
```

### 3. Session Timeout Mechanism

- **Timeout Duration**: 30 minutes (configurable via `SESSION_TIMEOUT_MINUTES`)
- **Detection**: Check `last_activity` timestamp on each request
- **Timeout Behavior**: Automatically create new session

```python
# User inactive for 30 minutes
→ Next request automatically creates new session
→ Old session remains in Storage (for history queries)
```

## API Endpoints

### POST /invoke

Main conversation endpoint.

**Request Parameters**:
```json
{
    "userid": "string (required)",
    "session_id": "string (optional)",
    "force_new_session": "boolean (optional, default false)",
    "messages": [{"role": "user", "content": "..."}],
    "system_prompt": "string (optional)",
    "model": "string (optional)"
}
```

**Response**:
```json
{
    "user_id": "alice",
    "session_id": "alice_a1b2c3d4",  # Important: client should save this
    "final_response": "...",
    "messages": [...]
}
```

### GET /sessions/{userid}

Query all sessions for a user.

**Response**:
```json
{
    "userid": "alice",
    "total_sessions": 3,
    "active_sessions": 1,
    "sessions": [
        {
            "session_id": "alice_a1b2c3d4",
            "last_activity": "2025-01-15T10:30:00",
            "is_active": true,
            "minutes_since_activity": 5,
            "created_at": "2025-01-15T10:00:00"
        }
    ]
}
```

### GET /state/{session_id}

Query state for a specific session (LangGraph state).

## Storage Structure

### Session Tracking Document

```python
StorageDocument(
    id="session_tracker_{session_id}",
    user_id="alice",
    session_id="alice_a1b2c3d4",
    document_type=StorageType.SESSION_SUMMARY,
    content="Session tracking record for alice_a1b2c3d4",
    metadata={
        "is_session_tracker": True,  # Mark as session tracking record
        "last_activity": "2025-01-15T10:30:00",
        "created_at": "2025-01-15T10:00:00"
    }
)
```

### Session Summary (created during finalization)

```python
StorageDocument(
    id="{session_id}_summary_{timestamp}",
    user_id="alice",
    session_id="alice_a1b2c3d4",
    document_type=StorageType.SESSION_SUMMARY,
    content="User asked about...",  # AI-generated summary
    metadata={
        "is_session_tracker": False,  # Actual session summary
        "message_count": 10,
        "preference_count": 2,
        ...
    }
)
```

## Client Best Practices

### 1. First Conversation

```javascript
const response = await fetch('/invoke', {
    method: 'POST',
    body: JSON.stringify({
        userid: 'alice',
        messages: [{ role: 'user', content: 'Hello' }]
    })
});

const data = await response.json();
// Save session_id to local state or localStorage
localStorage.setItem('current_session', data.session_id);
```

### 2. Continue Conversation

```javascript
const sessionId = localStorage.getItem('current_session');

const response = await fetch('/invoke', {
    method: 'POST',
    body: JSON.stringify({
        userid: 'alice',
        session_id: sessionId,  // Use saved session_id
        messages: [{ role: 'user', content: 'Continue' }]
    })
});
```

### 3. Start New Topic

```javascript
const response = await fetch('/invoke', {
    method: 'POST',
    body: JSON.stringify({
        userid: 'alice',
        force_new_session: true,  // Force new session
        messages: [{ role: 'user', content: 'New topic' }]
    })
});

const data = await response.json();
// Update saved session_id
localStorage.setItem('current_session', data.session_id);
```

### 4. View Session History

```javascript
const response = await fetch('/sessions/alice');
const data = await response.json();

// Display all user conversations
data.sessions.forEach(session => {
    console.log(`Session: ${session.session_id}`);
    console.log(`Active: ${session.is_active}`);
    console.log(`Last activity: ${session.last_activity}`);
});
```

## Integration with Memory Manager

```python
# In graph.py's finalize_session_node
config = {
    "configurable": {
        "thread_id": session_id,  # Session ID
        "user_id": user_id        # User ID
    }
}

# Memory Manager uses:
# - session_id: Distinguish different conversations
# - user_id: Aggregate all user preferences
```

**Key Points**:
- One **user** can have multiple **sessions**
- Each **session** independently manages conversation history
- **Preferences** from all sessions are aggregated under the same user
- Preferences extracted during session finalization belong to user_id

## Configuration

```python
# In server.py
SESSION_TIMEOUT_MINUTES = 30  # Session timeout (minutes)
```

Configure via environment variables:
```bash
export SESSION_TIMEOUT_MINUTES=60  # 1 hour timeout
```

## Advantages

1. ✅ **Persistence**: Session information stored in Storage, survives server restarts
2. ✅ **Scalability**: Supports multi-server deployment (shared Storage)
3. ✅ **Flexibility**: Supports both automatic management and manual control
4. ✅ **Traceability**: All session history is queryable
5. ✅ **Unified Architecture**: Reuses existing Storage interface, no additional components needed

## Future Enhancements

### Optional Features

1. **Session Tags**
   ```python
   POST /invoke
   {
       "session_id": "...",
       "session_tags": ["work", "project_x"]  # Tag sessions
   }
   ```

2. **Session Search**
   ```python
   GET /sessions/search?userid=alice&tags=work&from=2025-01-01
   ```

3. **Session Export**
   ```python
   GET /sessions/export/{session_id}  # Export complete conversation record
   ```

4. **Session Merge**
   ```python
   POST /sessions/merge
   {
       "session_ids": ["session1", "session2"],
       "new_session_id": "merged_session"
   }
   ```

## Related Documentation

- [Memory Management](./memory-management.md)
- [Storage Interface](../agent/interfaces/README.md#storage-interface)
- [API Reference](./api-reference.md)
