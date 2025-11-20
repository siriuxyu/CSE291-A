import asyncio
import argparse
from graph import graph
from context import Context
from state import InputState
from utils import print_debug, print_simple
from langchain_core.messages import HumanMessage

SESSION_ID = "interactive_session"
DEBUG_MODE = False

async def ask(text: str):
    input_state = InputState(messages=[HumanMessage(content=text)])

    async for event in graph.astream(
        input=input_state,
        context=Context(system_prompt="You are a helpful AI assistant."),
        config={"configurable": {"thread_id": SESSION_ID}},
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Assistant Interactive Q&A')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode to show all event details')
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    asyncio.run(main())