from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import time
from agent.graph import graph
from agent.context import Context
from agent.utils import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


api = FastAPI()


########################################################
# Define the request body format
########################################################
class Message(BaseModel):
    role: str
    content: str

class ReactRequest(BaseModel):
    messages: List[Message]
    userid: str  # Required for context history
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    max_search_results: Optional[int] = None


class ReactResponse(BaseModel):
    messages: List[Dict[str, Any]]
    final_response: str


class ChatRequest(BaseModel):
    message: str


########################################################
# Invoke the React Agent with user context
########################################################
@api.post("/invoke", response_model=ReactResponse)
async def invoke(req: ReactRequest):
    """Invoke the React agent with the provided messages and context.
    
    The userid is used to maintain conversation history for each user.
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
            'details': {
                'system_prompt': context.system_prompt,
                'model': context.model,
                'max_search_results': context.max_search_results
            }
        })
        
        # Configure thread for user context history
        config = {"configurable": {"thread_id": req.userid}}
        
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
# Main entry point
########################################################
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting React Agent server on http://0.0.0.0:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)
