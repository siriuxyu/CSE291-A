import asyncio
import argparse
from agent_langmem.graph import graph, store
from agent_langmem.state import InputState
from agent_langmem.utils import print_debug, print_simple
from langchain_core.messages import HumanMessage

SESSION_ID = "interactive_session"
DEBUG_MODE = True

async def ask(text: str, config: dict = {}):
    input_state = InputState(messages=[HumanMessage(content=text)])

    async for event in graph.astream(
        input=input_state,
        stream_mode="updates",
        config=config,
    ):
        if DEBUG_MODE:
            print_debug(event)

    if DEBUG_MODE:
        print("=" * 60)
        print("End of Response")
        print("=" * 60 + "\n")
    else:
        print_simple(event)
        print()

async def main():
    print("=" * 60)
    if DEBUG_MODE:
        print("AI Assistant - Interactive Q&A (DEBUG MODE)")
    else:
        print("AI Assistant - Interactive Q&A")
    print("=" * 60)
    print("Type 'quit', 'exit' or 'q' to end the conversation")
    print("Type 'clear' to start a new session")
    print("=" * 60)
    print()

    global SESSION_ID

    while True:
        try:
            # Get user input
            user_input = input("You >>> ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Check for clear command
            if user_input.lower() == 'clear':
                import uuid
                SESSION_ID = f"interactive_session_{uuid.uuid4().hex[:8]}"
                print("Session cleared. Starting a new conversation.\n")
                continue

            # Ignore empty inputs
            if not user_input:
                continue

            # invoke the input
            print()
            await ask(user_input)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


async def test():
    cfg1 = {"configurable": {"thread_id": "thread_1"}, "user_id": "user_123"}
    cfg2 = {"configurable": {"thread_id": "thread_2"}, "user_id": "user_123"}

    await ask("Know which display mode I prefer?", config=cfg1)
    await ask("dark. Remember that.", config=cfg1)
    await ask("Hey there. Do you remember me? What are my preferences?", config=cfg2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Assistant Interactive Q&A')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode to show all event details')
    args = parser.parse_args()

    asyncio.run(test())

