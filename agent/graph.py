"""Define a custom Reasoning and Action agent with LangMem integration.

Works with a chat model with tool calling support.
Includes long-term memory management through LangMem.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, cast, Optional

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

# Python version compatibility for UTC
if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import MemorySaver

from .context import Context
from .state import InputState, State
from .utils import get_logger, load_chat_model
from tools import TOOLS
from .preference import extract_preferences
# Lazy import to avoid circular imports and startup issues
def is_langmem_enabled():
    try:
        from .memory.langmem_adapter import is_langmem_enabled as _is_langmem_enabled
        return _is_langmem_enabled()
    except Exception:
        return False

def is_storage_available():
    try:
        from .memory.langmem_adapter import is_storage_available as _is_storage_available
        return _is_storage_available()
    except Exception:
        return False

def get_langmem_manager():
    try:
        from .memory.langmem_adapter import get_langmem_manager as _get_langmem_manager
        return _get_langmem_manager()
    except Exception:
        return None

# Import checkpointers - skip Redis to avoid connection issues
RedisSaver = None
redis = None

# Initialize logger
logger = get_logger(__name__)


def get_tools_with_memory(user_id: Optional[str] = None) -> List:
    """
    Get the combined list of tools including LangMem tools for a user.
    
    Args:
        user_id: If provided, include user-specific memory tools
        
    Returns:
        List of all available tools
    """
    all_tools = list(TOOLS)  # Start with base tools
    
    if is_langmem_enabled() and user_id:
        manager = get_langmem_manager()
        memory_tools = manager.get_tools_for_user(user_id)
        all_tools.extend(memory_tools)
        logger.debug(
            f"Added {len(memory_tools)} LangMem tools for user {user_id}",
            extra={"function": "get_tools_with_memory", "user_id": user_id}
        )
    
    return all_tools


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.
    Includes LangMem tools based on user_id from config.

    Args:
        state (State): The current state of the conversation.
        runtime (Runtime[Context]): Runtime context for the model.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    start_time = time.time()
    
    # Extract context from runtime
    context = runtime.context
    
    # Get user_id from context (set by server when creating the context)
    user_id = context.user_id if context and context.user_id else None
    
    logger.info("Calling model", extra={
        'function': 'call_model',
        'details': {
            'model': context.model,
            'message_count': len(state.messages),
            'is_last_step': state.is_last_step,
            'user_id': user_id,
            'langmem_enabled': is_langmem_enabled(),
            'storage_available': is_storage_available(),
        }
    })
    
    # Get tools including LangMem tools for this user
    tools = get_tools_with_memory(user_id)
    
    # Initialize the model with tool binding
    model = load_chat_model(context.model).bind_tools(tools)
    
    logger.debug("Model loaded and tools bound", extra={
        'function': 'call_model',
        'details': {
            'model': context.model,
            'available_tools': [getattr(tool, 'name', str(tool)) for tool in tools],
            'total_tools': len(tools),
        }
    })

    # Format the system prompt with memory context
    system_message = context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )
    
    # Enhance system prompt with memory search hint if LangMem is enabled
    if is_langmem_enabled() and user_id:
        system_message += (
            "\n\nYou have access to long-term memory tools. "
            "Use 'store_memory' to save important information the user shares. "
            "Use 'search_memory' to recall previously stored information when relevant."
        )
    
    logger.debug("Prepared system prompt", extra={
        'function': 'call_model',
        'details': {
            'system_prompt_length': len(system_message),
            'system_time': datetime.now(tz=UTC).isoformat()
        }
    })

    # Get the model's response
    logger.debug("Invoking model", extra={
        'function': 'call_model',
        'details': {
            'total_messages': len(state.messages) + 1  # +1 for system message
        }
    })
    
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    
    model_duration_ms = (time.time() - start_time) * 1000
    
    logger.info("Model response received", extra={
        'function': 'call_model',
        'duration_ms': round(model_duration_ms, 2),
        'details': {
            'has_tool_calls': bool(response.tool_calls),
            'tool_calls_count': len(response.tool_calls) if response.tool_calls else 0,
            'response_length': len(response.content) if response.content else 0,
            'response_preview': response.content[:200] if response.content else None
        }
    })
    
    # Log tool calls if present
    if response.tool_calls:
        logger.debug("Model requested tool calls", extra={
            'function': 'call_model',
            'details': {
                'tool_calls': [
                    {
                        'name': tc.get('name', 'unknown'),
                        'args_preview': {k: str(v)[:50] for k, v in tc.get('args', {}).items()}
                    }
                    for tc in response.tool_calls
                ]
            }
        })

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        logger.warning("Model requested tools on last step, returning fallback message", extra={
            'function': 'call_model',
            'details': {
                'tool_calls_count': len(response.tool_calls)
            }
        })
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def execute_tools(
    state: State,
    runtime: Runtime[Context],
) -> Dict[str, Any]:
    """
    Execute tools dynamically, including user-specific memory tools.
    
    This function replaces the static ToolNode to allow dynamic tool selection
    based on the user_id from context.
    """
    # Get user_id from context
    context = runtime.context
    user_id = context.user_id if context and context.user_id else None
    
    # Get all tools including memory tools for this user
    all_tools = get_tools_with_memory(user_id)
    
    tool_names = [getattr(t, 'name', str(t)) for t in all_tools]
    logger.info(f"Executing tools for user {user_id}", extra={
        'function': 'execute_tools',
        'details': {
            'user_id': user_id,
            'available_tools': tool_names,
            'total_tools': len(all_tools),
        }
    })
    
    # Create a ToolNode with all tools and execute
    # ToolNode expects a dict with "messages" key containing the conversation
    tool_node = ToolNode(all_tools)
    
    # Convert state to dict format that ToolNode expects
    state_dict = {"messages": state.messages}
    result = await tool_node.ainvoke(state_dict)
    
    return result


# Define a new graph
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the nodes
builder.add_node(call_model)
builder.add_node("tools", execute_tools)  # Dynamic tools node with memory support
builder.add_node("extract_preferences", extract_preferences)

# Set the entrypoint
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools", "extract_preferences"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
    has_tool_calls = bool(last_message.tool_calls)
    
    logger.debug("Routing model output", extra={
        'function': 'route_model_output',
        'details': {
            'has_tool_calls': has_tool_calls,
            'tool_calls_count': len(last_message.tool_calls) if last_message.tool_calls else 0
        }
    })
    
    # If there is no tool call, then we finish
    if not has_tool_calls:
        logger.info("No tool calls detected, routing to preference extraction", extra={
            'function': 'route_model_output',
            'details': {'next_node': 'extract_preferences'}
        })
        return "extract_preferences"
    
    # Otherwise we execute the requested actions
    logger.info("Tool calls detected, routing to tools", extra={
        'function': 'route_model_output',
        'details': {
            'next_node': 'tools',
            'tool_calls': [
                {'name': tc.get('name', 'unknown')} 
                for tc in last_message.tool_calls[:3]  # Log first 3 tools
            ]
        }
    })
    return "tools"


# Add conditional edge
builder.add_conditional_edges(
    "call_model",
    route_model_output,
    {"tools": "tools", "extract_preferences": "extract_preferences"}
)

# Add edge from tools back to call_model
builder.add_edge("tools", "call_model")
builder.add_edge("extract_preferences", "__end__")

# Set up checkpoint storage - using MemorySaver for simplicity
logger.info("Using in-memory checkpointer")
checkpointer = MemorySaver()

# Log storage configuration (deferred to avoid import issues)
try:
    langmem_manager = get_langmem_manager()
    logger.info(
        "Memory configuration",
        extra={
            "langmem_enabled": is_langmem_enabled(),
            "storage_available": is_storage_available(),
            "has_persistent_storage": langmem_manager.has_persistent_storage if langmem_manager else False,
        }
    )
except Exception as e:
    logger.warning(f"Could not initialize LangMem manager: {e}")

# Compile the builder into an executable graph with checkpointer only
# Note: We use ChromaDB for persistent storage instead of LangMem's InMemoryStore
logger.info("Compiling graph with checkpointer")
graph = builder.compile(
    name="ReAct Agent",
    checkpointer=checkpointer,
)
