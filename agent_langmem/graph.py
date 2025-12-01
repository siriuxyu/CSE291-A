"""Constructs a LangGraph graph for an agent with long-term memory.
"""
from langgraph.utils.config import get_store 
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from agent_langmem.react_agent import create_react_agent
from agent_langmem.long_term import create_manage_memory_tool
from tools import TOOLS

TOOLS.append(create_manage_memory_tool(namespace=("memories", "{user_id}")))

def prompt(state):
    """Prepare the messages for the LLM."""
    # Get store from configured contextvar; 
    store = get_store() # Same as that provided to `create_react_agent`
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]

def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]

store = InMemoryStore(
    index={
        "dims": 2,
        "embed": embed,
    }
)
checkpointer = MemorySaver()

graph = create_react_agent( 
    model="anthropic:claude-sonnet-4-5-20250929",
    prompt=prompt,
    tools=TOOLS,
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    # And the graph "state" will be checkpointed after each node
    # completes executing for tracking the chat history and durable execution
    checkpointer=checkpointer, 
)