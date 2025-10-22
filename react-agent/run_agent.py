import asyncio
from graph import graph
from context import Context
from langchain_core.messages import AIMessageChunk

async def main_token_stream():

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what is the weather in San Diego?"}]},
        context=Context(system_prompt="You are a helpful AI assistant.")
    ):
        if "call_model" in chunk:
            message_chunk = chunk["call_model"]["messages"][-1]
            print(message_chunk)
            if isinstance(message_chunk, AIMessageChunk):
                print(message_chunk.content, end="", flush=True)
            elif message_chunk.tool_calls:
                print(f"\n--- Calling Tools: {message_chunk.tool_calls} ---")

    print("\n--- End of Stream ---")


if __name__ == "__main__":
    asyncio.run(main_token_stream())