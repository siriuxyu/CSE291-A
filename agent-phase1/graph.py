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

# Add parent directory to path to import tools from project root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from context import Context
from state import InputState, State
from utils import get_logger, load_chat_model
from tools import TOOLS

# Initialize logger
logger = get_logger(__name__)

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


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
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
        logger.info("No tool calls detected, ending conversation", extra={
            'function': 'route_model_output',
            'details': {'next_node': '__end__'}
        })
        return "__end__"
    
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
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Set up checkpoint storage (memory-only; no disk-backed persistence)
logger.info("Using in-memory checkpointer. Sessions are not persistent across restarts.")
checkpointer = MemorySaver()

# Compile the builder into an executable graph with checkpointer
graph = builder.compile(name="ReAct Agent", checkpointer=checkpointer)
