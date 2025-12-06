from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from datetime import datetime, timedelta
from agent.graph import graph, get_memory_manager
from agent.context import Context
from agent.utils import setup_logging, get_logger
from agent.interfaces import StorageDocument, StorageType

# Initialize logging
setup_logging()
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

api = FastAPI()

# Session management
SESSION_TIMEOUT_MINUTES = 30

# Session tracking in storage instead of memory
# We'll use the Storage backend from Memory Manager


async def get_or_create_session(user_id: str, session_id: Optional[str] = None, force_new: bool = False) -> str:
    """
    Get existing session or create new one based on time window.
    Uses Storage backend to persist session information.

    Args:
        user_id: User identifier
        session_id: Optional explicit session ID from client
        force_new: Force creating a new session

    Returns:
        session_id to use
    """
    memory_mgr = get_memory_manager()
    storage = memory_mgr._storage if memory_mgr else None

    if not storage:
        # Fallback: if no storage, generate session_id directly
        logger.warning("No storage backend available, generating session_id without persistence")
        if session_id and not force_new:
            return session_id
        return f"{user_id}_{uuid.uuid4().hex[:8]}"

    now = datetime.now()

    # If user explicitly provides session_id, use it
    if session_id and not force_new:
        # Update session activity in storage
        await _update_session_activity(storage, user_id, session_id, now)
        logger.info(f"Using explicit session_id: {session_id} for user: {user_id}")
        return session_id

    # If force_new is True, create new session
    if force_new:
        new_session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        await _create_session_record(storage, user_id, new_session_id, now)
        logger.info(f"Force creating new session: {new_session_id} for user: {user_id}")
        return new_session_id

    # Auto-manage: check if user has active session
    active_session = await _get_active_session(storage, user_id, SESSION_TIMEOUT_MINUTES)

    if active_session:
        # Reuse active session
        await _update_session_activity(storage, user_id, active_session, now)
        logger.info(f"Reusing active session: {active_session} for user: {user_id}")
        return active_session

    # Create new session
    new_session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
    await _create_session_record(storage, user_id, new_session_id, now)
    logger.info(f"Creating new session: {new_session_id} for user: {user_id}")
    return new_session_id


async def _create_session_record(storage, user_id: str, session_id: str, created_at: datetime) -> None:
    """Create a session record in storage"""
    try:
        doc = StorageDocument(
            id=f"session_tracker_{session_id}",
            user_id=user_id,
            session_id=session_id,
            document_type=StorageType.SESSION_SUMMARY,  # Reuse existing type
            content=f"Session tracking record for {session_id}",
            embedding=None,
            metadata={
                "is_session_tracker": True,  # Mark as session tracking record
                "last_activity": created_at.isoformat(),
                "created_at": created_at.isoformat()
            },
            created_at=created_at,
            updated_at=created_at
        )
        await storage.store_documents_batch([doc])
    except Exception as e:
        logger.error(f"Failed to create session record: {e}")


async def _update_session_activity(storage, user_id: str, session_id: str, activity_time: datetime) -> None:
    """Update session last activity time"""
    try:
        doc_id = f"session_tracker_{session_id}"
        # Update the document with new activity time
        doc = StorageDocument(
            id=doc_id,
            user_id=user_id,
            session_id=session_id,
            document_type=StorageType.SESSION_SUMMARY,
            content=f"Session tracking record for {session_id}",
            embedding=None,
            metadata={
                "is_session_tracker": True,
                "last_activity": activity_time.isoformat()
            },
            created_at=activity_time,  # Will be ignored on update
            updated_at=activity_time
        )
        await storage.store_documents_batch([doc])
    except Exception as e:
        logger.error(f"Failed to update session activity: {e}")


async def _get_active_session(storage, user_id: str, timeout_minutes: int) -> Optional[str]:
    """Get the most recent active session for a user"""
    try:
        # Get all session tracking documents for this user
        docs = await storage.get_documents_by_user(
            user_id=user_id,
            document_type=StorageType.SESSION_SUMMARY,
            limit=100  # Get recent sessions
        )

        # Filter for session tracker documents
        session_docs = [
            doc for doc in docs
            if doc.metadata.get("is_session_tracker") == True
        ]

        if not session_docs:
            return None

        # Find most recent active session
        now = datetime.now()
        timeout_delta = timedelta(minutes=timeout_minutes)

        active_sessions = []
        for doc in session_docs:
            last_activity_str = doc.metadata.get("last_activity")
            if last_activity_str:
                last_activity = datetime.fromisoformat(last_activity_str)
                if now - last_activity < timeout_delta:
                    active_sessions.append((doc.session_id, last_activity))

        if not active_sessions:
            return None

        # Return the most recent active session
        active_sessions.sort(key=lambda x: x[1], reverse=True)
        return active_sessions[0][0]

    except Exception as e:
        logger.error(f"Failed to get active session: {e}")
        return None


########################################################
# Define the request body format
########################################################
class Message(BaseModel):
    role: str
    content: str

class ReactRequest(BaseModel):
    messages: List[Message]
    userid: str  # Required for user identification
    session_id: Optional[str] = None  # Optional: if None, auto-manage based on time window
    force_new_session: Optional[bool] = False  # Force creating a new session
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_search_results: Optional[int] = None


class ReactResponse(BaseModel):
    messages: List[Dict[str, Any]]
    final_response: str
    session_id: str  # Return the session_id used for this conversation
    user_id: str  # Return the user_id


class ChatRequest(BaseModel):
    message: str


########################################################
# Invoke the React Agent with user context
########################################################
@api.post("/invoke", response_model=ReactResponse)
async def invoke(req: ReactRequest):
    """Invoke the React agent with the provided messages and context.

    The userid is used to identify the user.
    session_id is used to maintain conversation history - if not provided, auto-managed based on time window.
    Set force_new_session=True to explicitly start a new conversation.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()

    try:
        # Get or create session ID
        session_id = await get_or_create_session(
            user_id=req.userid,
            session_id=req.session_id,
            force_new=req.force_new_session
        )

        logger.info("Received invoke request", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'session_id': session_id,
            'details': {
                'message_count': len(req.messages),
                'model': req.model or "default",
                'has_system_prompt': req.system_prompt is not None,
                'force_new_session': req.force_new_session,
                'first_message_preview': req.messages[0].content[:100] if req.messages else None
            }
        })
        
        # Convert messages to the format expected by the graph
        messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        
        logger.debug("Processing messages", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'details': {'messages': [{'role': m['role'], 'content': m['content'][:100]} for m in messages]}
        })
        
        # Create context with optional parameters
        context = Context(
            system_prompt=req.system_prompt or "You are a helpful AI assistant.",
            model=req.model or "anthropic/claude-sonnet-4-5-20250929",
            max_search_results=req.max_search_results or 10
        )
        
        logger.debug("Context created", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'session_id': session_id,
            'details': {
                'system_prompt': context.system_prompt,
                'model': context.model,
                'max_search_results': context.max_search_results
            }
        })

        # Configure thread with session_id (for conversation history) and user_id (for memory)
        config = {
            "configurable": {
                "thread_id": session_id,  # Use session_id for conversation history
                "user_id": req.userid      # Also pass user_id for memory manager
            }
        }
        
        # Stream the response from the graph
        response_messages = []
        final_response = ""
        chunk_count = 0
        
        logger.info("Starting agent execution", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'function': 'graph.astream'
        })
        
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            chunk_count += 1
            
            # Debug: log chunk structure
            logger.debug(f"Chunk received (count: {chunk_count})", extra={
                'request_id': request_id,
                'user_id': req.userid,
                'details': {'chunk': str(chunk)[:200]}  # Truncate for readability
            })
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
                logger.debug("Processing model call chunk", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {'chunk_type': 'call_model'}
                })
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
                logger.debug("Processing tool execution chunk", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {'chunk_type': 'tools'}
                })
            
            for msg in msgs_to_process:
                # Get message content based on message type
                msg_content = ""
                if hasattr(msg, 'content'):
                    msg_content = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content = msg.data['content']
                else:
                    msg_content = str(msg)
                
                # Debug: log message content
                logger.debug("Processing message", extra={
                    'request_id': request_id,
                    'user_id': req.userid,
                    'details': {
                        'message_type': type(msg).__name__,
                        'has_content': hasattr(msg, 'content'),
                        'has_tool_calls': hasattr(msg, 'tool_calls'),
                        'content_preview': str(msg_content)[:200] if msg_content else None,
                        'tool_calls': [{'name': tc.get('name', 'unknown'), 'args': tc.get('args', {})} for tc in getattr(msg, 'tool_calls', [])[:3]]
                    }
                })
                
                # Get role
                msg_role = "assistant"
                if hasattr(msg, 'type'):
                    msg_role = msg.type if msg.type in ['user', 'assistant', 'system'] else "assistant"
                elif hasattr(msg, 'role'):
                    msg_role = msg.role
                
                response_messages.append({
                    "role": msg_role,
                    "content": msg_content,
                    "tool_calls": getattr(msg, 'tool_calls', None)
                })
                
                # Get the final response (last non-tool message)
                if msg_content and not getattr(msg, 'tool_calls', None):
                    final_response = msg_content
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Request completed successfully", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'session_id': session_id,
            'duration_ms': round(duration_ms, 2),
            'details': {
                'chunks_processed': chunk_count,
                'response_messages_count': len(response_messages),
                'final_response_length': len(final_response),
                'final_response_preview': final_response[:200] if final_response else None
            }
        })

        return ReactResponse(
            messages=response_messages,
            final_response=final_response,
            session_id=session_id,
            user_id=req.userid
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("Request failed", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'session_id': session_id if 'session_id' in locals() else 'unknown',
            'duration_ms': round(duration_ms, 2)
        }, exc_info=True)
        return ReactResponse(
            messages=[],
            final_response=f"Error: {str(e)}",
            session_id=session_id if 'session_id' in locals() else 'error',
            user_id=req.userid
        )


########################################################
# Simple chat endpoint (no context history)
########################################################
@api.post("/chat")
async def chat(req: ChatRequest):
    """Simple chat endpoint for single message interactions without context history."""
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        logger.info("Received chat request", extra={
            'request_id': request_id,
            'details': {
                'message_length': len(req.message),
                'message_preview': req.message[:100]
            }
        })
        
        messages = [{"role": "user", "content": req.message}]
        
        context = Context(
            system_prompt="You are a helpful AI assistant.",
            model="anthropic/claude-sonnet-4-5-20250929"
        )
        
        # Generate a unique temporary thread_id for this single-use chat
        temp_thread_id = f"temp_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": temp_thread_id}}
        
        logger.debug("Starting chat execution", extra={
            'request_id': request_id,
            'details': {'temp_thread_id': temp_thread_id}
        })
        
        final_response = ""
        chunk_count = 0
        
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            chunk_count += 1
            
            # Debug: log chunk structure
            logger.debug(f"Chat chunk received (count: {chunk_count})", extra={
                'request_id': request_id,
                'details': {'chunk': str(chunk)[:200]}
            })
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
            
            for msg in msgs_to_process:
                # Debug: log message structure
                logger.debug("Processing chat message", extra={
                    'request_id': request_id,
                    'details': {
                        'message_type': type(msg).__name__,
                        'has_content': hasattr(msg, 'content'),
                        'has_tool_calls': hasattr(msg, 'tool_calls')
                    }
                })
                
                # Get message content based on message type
                msg_content = ""
                if hasattr(msg, 'content'):
                    msg_content = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content = msg.data['content']
                else:
                    msg_content = str(msg)
                
                # Debug: log message content
                logger.debug("Chat message content", extra={
                    'request_id': request_id,
                    'details': {
                        'content_preview': str(msg_content)[:200] if msg_content else None,
                        'tool_calls': [{'name': tc.get('name', 'unknown'), 'args': tc.get('args', {})} for tc in getattr(msg, 'tool_calls', [])[:3]]
                    }
                })
                
                # Get final response (last non-tool message)
                if msg_content and not getattr(msg, 'tool_calls', None):
                    final_response = msg_content
        
        duration_ms = (time.time() - start_time) * 1000
        
        logger.info("Chat request completed", extra={
            'request_id': request_id,
            'duration_ms': round(duration_ms, 2),
            'details': {
                'chunks_processed': chunk_count,
                'response_length': len(final_response),
                'response_preview': final_response[:200] if final_response else None
            }
        })
        
        return {"response": final_response}
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error("Chat request failed", extra={
            'request_id': request_id,
            'duration_ms': round(duration_ms, 2)
        }, exc_info=True)
        return {"response": f"Error: {str(e)}"}


########################################################
# Session management endpoints
########################################################
@api.get("/sessions/{userid}")
async def get_user_sessions(userid: str):
    """Get all sessions for a specific user from Storage."""
    try:
        memory_mgr = get_memory_manager()
        storage = memory_mgr._storage if memory_mgr else None

        if not storage:
            return {
                "userid": userid,
                "sessions": [],
                "message": "Storage backend not available"
            }

        # Get session tracking documents for this user
        docs = await storage.get_documents_by_user(
            user_id=userid,
            document_type=StorageType.SESSION_SUMMARY,
            limit=100
        )

        # Filter for session tracker documents
        session_docs = [
            doc for doc in docs
            if doc.metadata.get("is_session_tracker") == True
        ]

        if not session_docs:
            return {
                "userid": userid,
                "sessions": [],
                "message": "No sessions found for this user"
            }

        now = datetime.now()
        session_list = []

        for doc in session_docs:
            last_activity_str = doc.metadata.get("last_activity")
            if last_activity_str:
                last_activity = datetime.fromisoformat(last_activity_str)
                is_active = (now - last_activity) < timedelta(minutes=SESSION_TIMEOUT_MINUTES)
                session_list.append({
                    "session_id": doc.session_id,
                    "last_activity": last_activity.isoformat(),
                    "is_active": is_active,
                    "minutes_since_activity": int((now - last_activity).total_seconds() / 60),
                    "created_at": doc.metadata.get("created_at")
                })

        # Sort by last activity
        session_list.sort(key=lambda x: x["last_activity"], reverse=True)

        return {
            "userid": userid,
            "sessions": session_list,
            "total_sessions": len(session_list),
            "active_sessions": sum(1 for s in session_list if s["is_active"])
        }
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}", exc_info=True)
        return {"error": str(e)}


@api.get("/state/{session_id}")
async def check_state(session_id: str):
    """Check the conversation state for a specific session."""
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = graph.get_state(config)

        if state:
            return {
                "session_id": session_id,
                "state": state.values,
                "next_node": state.next,
                "config": state.config
            }
        else:
            return {"error": f"Session {session_id} not found"}
    except Exception as e:
        return {"error": str(e)}


########################################################
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting React Agent server on http://0.0.0.0:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)
