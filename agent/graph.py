"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

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

# Import checkpointers
try:
    from langgraph.checkpoint.redis import RedisSaver
    import redis
except ImportError:
    RedisSaver = None
    redis = None

from langgraph.checkpoint.memory import MemorySaver

# Initialize logger
logger = get_logger(__name__)

# ============================================================================
# Memory Manager Initialization
# ============================================================================

# Global memory manager instance (will be initialized lazily)
_memory_manager = None


def get_memory_manager():
    """Get or create the global memory manager instance"""
    global _memory_manager

    if _memory_manager is not None:
        return _memory_manager

    logger.info("Initializing Memory Manager")

    try:
        # Create mock storage backend (replace with real implementation later)
        class MockStorage:
            def __init__(self):
                self.documents = []

            async def store_documents_batch(self, documents):
                self.documents.extend(documents)
                logger.debug(f"Stored {len(documents)} documents to mock storage")
                return True

            async def get_documents_by_session(self, session_id, user_id, limit=None):
                results = [d for d in self.documents if d.session_id == session_id]
                return results[:limit] if limit else results

            async def get_user_contexts(self, user_id, limit=10):
                from agent.interfaces import StorageType
                prefs = [
                    d for d in self.documents
                    if d.user_id == user_id and d.document_type == StorageType.USER_PREFERENCE
                ]
                return prefs[:limit]

        # Create extractor
        from agent.extraction import PatternBasedContextExtractor
        extractor = PatternBasedContextExtractor()

        # Create mock embedding service
        class MockEmbedding:
            async def embed_text(self, text):
                return [0.1] * 384  # Mock 384-dim vector

        # Create memory manager
        from agent.memory.manager import ContextMemoryManager

        _memory_manager = ContextMemoryManager(
            storage=MockStorage(),
            extractor=extractor,
            embedding=MockEmbedding(),
            max_short_term_messages=10,
            max_short_term_tokens=10000,
            enable_long_term=True
        )

        logger.info("Memory Manager initialized successfully")
        return _memory_manager

    except Exception as e:
        logger.error(f"Failed to initialize Memory Manager: {e}", exc_info=True)
        return None

# Define the function that calls the model

async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        runtime (Runtime[Context]): Runtime context for the model.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    start_time = time.time()
    
    logger.info("Calling model", extra={
        'function': 'call_model',
        'details': {
            'model': runtime.context.model,
            'message_count': len(state.messages),
            'is_last_step': state.is_last_step
        }
    })
    
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)
    
    logger.debug("Model loaded and tools bound", extra={
        'function': 'call_model',
        'details': {
            'model': runtime.context.model,
            'available_tools': [tool.name for tool in TOOLS]
        }
    })

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
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
    
    response = cast( # type: ignore[redundant-cast]
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


# Define finalize_session node
async def finalize_session_node(
    state: State, runtime: Runtime[Context]
) -> Dict:
    """Finalize session by extracting preferences (called at conversation end)"""

    # Get memory manager
    memory_mgr = get_memory_manager()
    if not memory_mgr:
        logger.debug("Memory Manager not available, skipping finalization")
        return {}

    # Get thread_id from config (used as session_id)
    config = runtime.config
    session_id = config.get("configurable", {}).get("thread_id", "unknown")
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    try:
        logger.info(f"Finalizing session: {session_id} for user: {user_id}", extra={
            'function': 'finalize_session_node',
            'details': {
                'session_id': session_id,
                'user_id': user_id,
                'message_count': len(state.messages)
            }
        })

        # Finalize the session - this will extract preferences
        success = await memory_mgr.finalize_session(session_id, user_id, force=False)

        if success:
            logger.info(f"Session {session_id} finalized successfully", extra={
                'function': 'finalize_session_node',
                'details': {'success': True}
            })
            metrics = memory_mgr.get_memory_metrics(user_id=user_id)
            logger.info(f"Memory metrics - Messages: {metrics.short_term_message_count}, "
                       f"Tokens: {metrics.short_term_token_estimate}")
        else:
            logger.warning(f"Session {session_id} finalization returned False")

    except Exception as e:
        logger.error(f"Error finalizing session {session_id}: {e}", exc_info=True)

    return {}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the three main nodes
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("finalize_session", finalize_session_node)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["finalize_session", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("finalize_session" or "tools").
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

    # If there is no tool call, then we finalize and finish
    if not has_tool_calls:
        logger.info("No tool calls detected, routing to session finalization", extra={
            'function': 'route_model_output',
            'details': {'next_node': 'finalize_session'}
        })
        return "finalize_session"

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


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
    {"tools": "tools", "finalize_session": "finalize_session"}
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# After finalization, end the conversation
builder.add_edge("finalize_session", "__end__")

# Set up checkpoint storage
redis_url = os.environ.get("REDIS_URL")

if redis_url and RedisSaver:
    try:
        checkpointer = RedisSaver.from_url(redis_url)
        logger.info(f"Successfully connected to Redis at {redis_url}")
    except (redis.exceptions.ConnectionError, Exception) as e:
        logger.warning(f"Could not connect to Redis at {redis_url}. Error: {e}")
        logger.warning("Falling back to in-memory saver. THIS IS NOT PERSISTENT.")
        checkpointer = MemorySaver()
else:
    logger.warning("REDIS_URL environment variable not set.")
    logger.warning("Using in-memory saver. User sessions will NOT be isolated or persistent.")
    checkpointer = MemorySaver()

# Compile the builder into an executable graph with checkpointer
graph = builder.compile(name="ReAct Agent", checkpointer=checkpointer)
