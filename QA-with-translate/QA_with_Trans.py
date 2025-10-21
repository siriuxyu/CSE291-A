import os
import redis
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic

class State(TypedDict):
    # Input
    input_text: str
    language: str

    # Intermediate
    step1_output: str

    # Output
    output_text: str

llm = ChatAnthropic(model="claude-3-5-haiku-20241022")

########################################################
# Define the LLM nodes
########################################################
def llm_1(state: State) -> State:
    prompt = f"Please answer the following question with your: {state['input_text']}.  Requirements: you should be honest if you don't know the answer, and please don't guess the solution."
    llm_response = llm.invoke(
        [{"role": "user", "content": prompt}]
    )
    # clear the output of the previous round to avoid confusion
    return {"step1_output": llm_response.content, "output_text": ""}

def llm_2(state: State) -> State:
    prompt = f"Please translate the following text to {state['language']}: {state['step1_output']}.  Requirements: you should follow the original text and don't add additional information."
    llm_response = llm.invoke(
        [{"role": "user", "content": prompt}]
    )
    return {"output_text": llm_response.content}


########################################################
# Build the graph
########################################################
def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("llm_1", llm_1)
    graph_builder.add_node("llm_2", llm_2)

    graph_builder.add_edge(START, "llm_1")
    graph_builder.add_edge("llm_1", "llm_2")

    # create a loop
    # When llm_2 runs, the graph will be ready for the next run from llm_1.
    graph_builder.add_edge("llm_2", "llm_1")


    graph = graph_builder.compile(
        # Pass the checkpointer to the graph
        checkpointer=checkpointer,
        # Interrupt after 'llm_2' node runs
        # Force langgraph to save the current state and stop execution,
        # Wait for the next call of the same thread_id.
        interrupt_after=["llm_2"], 
    )
    return graph


########################################################
# DEBUG FUNCTION: view all saved states
########################################################
def view_all_states():
    print("=== View all saved states ===")
    try:
        # Get all thread states
        if hasattr(checkpointer, 'storage') and hasattr(checkpointer.storage, 'storage'):
            # MemorySaver's internal storage
            storage = checkpointer.storage.storage
            if hasattr(storage, 'items'):
                for thread_id, state_data in storage.items():
                    print(f"Thread ID: {thread_id}")
                    print(f"State data: {state_data}")
                    print("-" * 30)
            else:
                print("Cannot access the internal structure of the memory storage")
        else:
            print("Currently using MemorySaver, but cannot directly access the internal state")
            print("Suggest using graph.get_state(config) to view the state of a specific thread")
    except Exception as e:
        print(f"Error viewing states: {e}")
    print("=" * 50)


########################################################
# DEBUG FUNCTION: view the state of a specific thread
########################################################
def view_thread_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        print(f"=== Thread {thread_id} state ===")
        print(f"State values: {state.values}")
        print(f"Next node: {state.next}")
        print(f"Config: {state.config}")
        print(f"State ID: {state.config}")
        print("=" * 50)
        return state
    except Exception as e:
        print(f"Failed to get the state of thread {thread_id}: {e}")
        return None

########################################################
# Main function
########################################################
def app(input_question: str, thread_id: str, language: str = "French") -> str:
    # Define the thread configuration (Config)
    # This will tell LangGraph to load/save the state for which thread_id
    # This is the core of user isolation
    config = {"configurable": {"thread_id": thread_id}}

    # This is the *current* input for this round
    # When the checkpointer loads the state, this input will be merged with the existing state
    # And continue from the last interrupted place (after llm_2),
    # Pass the new input_text to llm_1 along the edge we added (llm_2 -> llm_1)
    current_input = {
        "input_text": input_question,
        "language": language,
    }

    # Debug: view the current state (if exists)
    try:
        current_state = graph.get_state(config)
        print(f"=== Current state (Thread: {thread_id}) ===")
        print(f"State values: {current_state.values}")
        print(f"Next node: {current_state.next}")
        print(f"State ID: {current_state.config}")
        print("=" * 50)
    except Exception as e:
        print(f"Failed to get the state (maybe a new user): {e}")

    # .stream() will automatically handle the loading of the state (if exists) and saving (when interrupted)
    events = graph.stream(current_input, config=config, stream_mode="values")

    final_output = ""
    for event in events:
        # When the graph runs to llm_2 and interrupts,
        # The last event will contain 'output_text'
        if "output_text" in event and event["output_text"]:
            final_output = event["output_text"]
    
    # Debug: view the final saved state
    try:
        final_state = graph.get_state(config)
        print(f"=== Final saved state (Thread: {thread_id}) ===")
        print(f"State values: {final_state.values}")
        print(f"Next node: {final_state.next}")
        print("=" * 50)
    except Exception as e:
        print(f"Failed to get the final state: {e}")
            
    # Return the last valid output
    return final_output


########################################################
# CHECKPOINT SETUP
########################################################
# Get Redis URL from the environment variable
redis_url = os.environ.get("REDIS_URL")

if redis_url:
    try:
        checkpointer = RedisSaver.from_url(redis_url)
        print(f"Successfully connected to Redis at {redis_url}")
    except redis.exceptions.ConnectionError as e:
        print(f"Warning: Could not connect to Redis at {redis_url}. Error: {e}")
        print("Falling back to in-memory saver. THIS IS NOT PERSISTENT.")
        checkpointer = MemorySaver()
else:
    print("WARNING: REDIS_URL environment variable not set.")
    print("Using in-memory saver. User sessions will NOT be isolated or persistent.")
    checkpointer = MemorySaver()

# Build the graph once at startup to reuse
graph = build_graph()