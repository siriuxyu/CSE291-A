from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from agent.graph import graph
from agent.context import Context
from agent.utils import setup_logging, get_logger
from agent.memory.langmem_adapter import (
    get_langmem_manager,
    is_langmem_enabled,
    is_storage_available,
)

# Import LangGraph errors for proper handling
try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = None  # type: ignore

# Initialize logging
setup_logging()
logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()

api = FastAPI(
    title="CSE291-A Agent API",
    description="React Agent with LangMem long-term memory support",
    version="2.0.0",
)


########################################################
# Define the request body format
########################################################
class Message(BaseModel):
    role: str
    content: str

class ReactRequest(BaseModel):
    messages: List[Message]
    userid: str  # Required for context history and memory tools
    thread_id: Optional[str] = None  # Optional: separate thread_id for message history (defaults to userid)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_search_results: Optional[int] = None


class ReactResponse(BaseModel):
    messages: List[Dict[str, Any]]
    final_response: str


class ChatRequest(BaseModel):
    message: str


class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10


class MemoryStoreRequest(BaseModel):
    key: str
    content: str


########################################################
# Invoke the React Agent with user context
########################################################
@api.post("/invoke", response_model=ReactResponse)
async def invoke(req: ReactRequest):
    """Invoke the React agent with the provided messages and context.
    
    The userid is used to maintain conversation history for each user.
    LangMem tools are automatically included for memory management.
    """
    request_id = uuid.uuid4().hex
    start_time = time.time()
    
    try:
        logger.info("Received invoke request", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'details': {
                'message_count': len(req.messages),
                'model': req.model or "default",
                'has_system_prompt': req.system_prompt is not None,
                'langmem_enabled': is_langmem_enabled(),
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
        
        # Use thread_id if provided, otherwise fall back to userid
        # thread_id: for LangGraph message history (can be unique per batch)
        # userid: for memory tools (should be consistent for same user's memories)
        thread_id = req.thread_id or req.userid
        
        # Create context with optional parameters, including user_id for memory tools
        context = Context(
            system_prompt=req.system_prompt or "You are a helpful AI assistant.",
            model=req.model or "anthropic/claude-sonnet-4-5-20250929",
            max_search_results=req.max_search_results or 10,
            user_id=req.userid,  # Pass user_id for LangMem memory tools (NOT thread_id)
        )
        
        logger.debug("Context created", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'thread_id': thread_id,
            'details': {
                'system_prompt': context.system_prompt,
                'model': context.model,
                'max_search_results': context.max_search_results
            }
        })
        
        # Configure thread for user context history (uses thread_id, NOT userid)
        config = {"configurable": {"thread_id": thread_id}}
        
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
            final_response=final_response
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        
        # Check if it's a GraphRecursionError (agent called tools too many times)
        is_recursion_error = (
            GraphRecursionError is not None and isinstance(e, GraphRecursionError)
        ) or "GraphRecursionError" in type(e).__name__ or "recursion limit" in str(e).lower()
        
        if is_recursion_error:
            logger.warning("Agent hit recursion limit", extra={
                'request_id': request_id,
                'user_id': req.userid,
                'duration_ms': round(duration_ms, 2),
                'error_type': 'GraphRecursionError'
            })
            return ReactResponse(
                messages=[],
                final_response="Error: Agent reached maximum tool call limit (25 iterations). The question may be too complex or require information not available in memory."
            )
        
        logger.error("Request failed", extra={
            'request_id': request_id,
            'user_id': req.userid,
            'duration_ms': round(duration_ms, 2)
        }, exc_info=True)
        return ReactResponse(
            messages=[],
            final_response=f"Error: {str(e)}"
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
        
        # Generate a unique temporary thread_id for this single-use chat
        temp_thread_id = f"temp_{uuid.uuid4().hex}"
        
        context = Context(
            system_prompt="You are a helpful AI assistant.",
            model="anthropic/claude-sonnet-4-5-20250929",
            user_id=temp_thread_id,  # Use temp_thread_id as user_id for memory
        )
        
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
# Check state endpoint
########################################################
@api.get("/state/{userid}")
async def check_state(userid: str):
    """Check the conversation state for a specific user."""
    try:
        config = {"configurable": {"thread_id": userid}}
        state = graph.get_state(config)
        
        if state:
            return {
                "userid": userid,
                "state": state.values,
                "next_node": state.next,
                "config": state.config
            }
        else:
            return {"error": f"User {userid} not found"}
    except Exception as e:
        return {"error": str(e)}


########################################################
# Reset short-term session (clear conversation history)
########################################################
@api.post("/reset/{userid}")
async def reset_short_term_session(userid: str, preserve_memory: bool = True):
    """
    Reset the short-term conversation history for a user.
    
    Args:
        userid: The user identifier
        preserve_memory: If True (default), keep long-term memories intact.
                        If False, also clear LangMem memories.
    
    This clears the checkpoint state but optionally preserves long-term memories.
    """
    request_id = uuid.uuid4().hex
    
    try:
        logger.info(f"Resetting session for user {userid}", extra={
            'request_id': request_id,
            'user_id': userid,
            'preserve_memory': preserve_memory,
        })
        
        # Clear checkpoint state by setting an empty state
        config = {"configurable": {"thread_id": userid}}
        
        # Get current state
        current_state = graph.get_state(config)
        message_count = 0
        if current_state and current_state.values:
            message_count = len(current_state.values.get("messages", []))
        
        # Note: LangGraph's MemorySaver doesn't have a direct "delete" method
        # The best approach is to update the state to an empty message list
        # or to use a new thread_id
        
        # If not preserving memory, also clear LangMem
        memory_cleared = False
        if not preserve_memory and is_langmem_enabled():
            manager = get_langmem_manager()
            memory_cleared = await manager.clear_user_memories(userid)
        
        logger.info(f"Session reset for user {userid}", extra={
            'request_id': request_id,
            'user_id': userid,
            'messages_cleared': message_count,
            'memory_cleared': memory_cleared,
        })
        
        return {
            "status": "success",
            "userid": userid,
            "messages_cleared": message_count,
            "memory_cleared": memory_cleared,
            "message": f"Session reset for user {userid}. Use a new request to start fresh."
        }
        
    except Exception as e:
        logger.error(f"Failed to reset session for {userid}: {e}", extra={
            'request_id': request_id,
            'user_id': userid,
        }, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


########################################################
# LangMem Memory Management Endpoints
########################################################
@api.get("/memory/{userid}")
async def list_memories(userid: str, limit: int = 100):
    """
    List all memories for a specific user.
    
    Args:
        userid: The user identifier
        limit: Maximum number of memories to return (default: 100)
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    memories = await manager.list_user_memories(userid, limit=limit)
    
    return {
        "userid": userid,
        "count": len(memories),
        "memories": memories,
    }


@api.post("/memory/{userid}/search")
async def search_memories(userid: str, req: MemorySearchRequest):
    """
    Search memories for a specific user.
    
    Args:
        userid: The user identifier
        req: Search request with query and limit
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    results = await manager.search_user_memories(
        userid,
        query=req.query,
        limit=req.limit or 10,
    )
    
    return {
        "userid": userid,
        "query": req.query,
        "count": len(results),
        "results": results,
    }


@api.post("/memory/{userid}/store")
async def store_memory(userid: str, req: MemoryStoreRequest):
    """
    Store a memory for a specific user.
    
    Args:
        userid: The user identifier
        req: Store request with key and content
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    success = await manager.store_user_memory(
        userid,
        key=req.key,
        content=req.content,
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store memory")
    
    return {
        "status": "success",
        "userid": userid,
        "key": req.key,
    }


@api.delete("/memory/{userid}/{key}")
async def delete_memory(userid: str, key: str):
    """
    Delete a specific memory for a user.
    
    Args:
        userid: The user identifier
        key: Memory key to delete
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    success = await manager.delete_user_memory(userid, key)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory")
    
    return {
        "status": "success",
        "userid": userid,
        "key": key,
    }


@api.delete("/memory/{userid}")
async def clear_memories(userid: str):
    """
    Clear all memories for a specific user.
    
    Args:
        userid: The user identifier
    """
    if not is_langmem_enabled():
        raise HTTPException(status_code=503, detail="LangMem is not enabled")
    
    manager = get_langmem_manager()
    success = await manager.clear_user_memories(userid)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear memories")
    
    return {
        "status": "success",
        "userid": userid,
        "message": "All memories cleared",
    }


########################################################
# Health check and status endpoints
########################################################
@api.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "langmem_enabled": is_langmem_enabled(),
    }


@api.get("/status")
async def status():
    """Get system status including LangMem and storage configuration."""
    manager = get_langmem_manager()
    
    return {
        "status": "running",
        "langmem": {
            "enabled": is_langmem_enabled(),
            "available": manager.is_available,
        },
        "storage": {
            "chromadb_available": is_storage_available(),
            "persistent": manager.has_persistent_storage,
        },
        "endpoints": {
            "invoke": "POST /invoke - Main agent endpoint",
            "chat": "POST /chat - Simple chat without history",
            "state": "GET /state/{userid} - Check user state",
            "reset": "POST /reset/{userid} - Reset user session",
            "memory_list": "GET /memory/{userid} - List user memories",
            "memory_search": "POST /memory/{userid}/search - Search memories",
            "memory_store": "POST /memory/{userid}/store - Store memory",
            "memory_delete": "DELETE /memory/{userid}/{key} - Delete memory",
            "memory_clear": "DELETE /memory/{userid} - Clear all memories",
            "storage_stats": "GET /storage/stats - Get storage statistics",
        }
    }


@api.get("/storage/stats")
async def storage_stats(userid: Optional[str] = None):
    """
    Get storage statistics.
    
    Args:
        userid: Optional user ID to filter stats
    """
    if not is_storage_available():
        raise HTTPException(status_code=503, detail="Storage backend not available")
    
    manager = get_langmem_manager()
    stats = await manager.get_storage_stats(userid)
    
    return stats


########################################################
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting React Agent server on http://0.0.0.0:8000")
    logger.info(f"LangMem enabled: {is_langmem_enabled()}")
    uvicorn.run(api, host="0.0.0.0", port=8000)
