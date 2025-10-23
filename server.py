from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from agent.graph import graph
from agent.context import Context


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
    try:
        # Convert messages to the format expected by the graph
        messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        
        # Create context with optional parameters
        context = Context(
            system_prompt=req.system_prompt or "You are a helpful AI assistant.",
            model=req.model or "anthropic/claude-sonnet-4-5-20250929",
            max_search_results=req.max_search_results or 10
        )
        
        # Configure thread for user context history
        config = {"configurable": {"thread_id": req.userid}}
        
        # Stream the response from the graph
        response_messages = []
        final_response = ""
        
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            # Debug: print chunk structure
            print(f"Chunk received: {chunk}")
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
            
            for msg in msgs_to_process:
                # Debug: print message structure
                print(f"Message type: {type(msg)}, Message: {msg}")
                
                # Get message content based on message type
                msg_content = ""
                if hasattr(msg, 'content'):
                    msg_content = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content = msg.data['content']
                else:
                    msg_content = str(msg)
                
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
        
        print(f"Final response: {final_response}")
        
        return ReactResponse(
            messages=response_messages,
            final_response=final_response
        )
        
    except Exception as e:
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
    try:
        messages = [{"role": "user", "content": req.message}]
        
        context = Context(
            system_prompt="You are a helpful AI assistant.",
            model="anthropic/claude-sonnet-4-5-20250929"
        )
        
        # Generate a unique temporary thread_id for this single-use chat
        temp_thread_id = f"temp_{uuid.uuid4().hex}"
        config = {"configurable": {"thread_id": temp_thread_id}}
        
        final_response = ""
        async for chunk in graph.astream(
            {"messages": messages},
            config=config,
            context=context
        ):
            # Debug: print chunk structure
            print(f"Chat chunk received: {chunk}")
            
            # Handle different chunk structures
            msgs_to_process = []
            
            if "messages" in chunk:
                msgs_to_process = chunk["messages"]
            elif "call_model" in chunk and "messages" in chunk["call_model"]:
                msgs_to_process = chunk["call_model"]["messages"]
            elif "tools" in chunk and "messages" in chunk["tools"]:
                msgs_to_process = chunk["tools"]["messages"]
            
            for msg in msgs_to_process:
                # Debug: print message structure
                print(f"Chat message type: {type(msg)}, Message: {msg}")
                
                # Get message content based on message type
                msg_content = ""
                if hasattr(msg, 'content'):
                    msg_content = msg.content
                elif hasattr(msg, 'data') and 'content' in msg.data:
                    msg_content = msg.data['content']
                else:
                    msg_content = str(msg)
                
                # Get final response (last non-tool message)
                if msg_content and not getattr(msg, 'tool_calls', None):
                    final_response = msg_content
        
        print(f"Chat final response: {final_response}")
        
        return {"response": final_response}
        
    except Exception as e:
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
    uvicorn.run(api, host="0.0.0.0", port=8000)
