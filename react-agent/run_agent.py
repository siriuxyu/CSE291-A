import asyncio
from graph import graph
from context import Context

async def main_token_stream():

    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what is the weather in San Diego?"}]},
        context=Context(system_prompt="You are a helpful AI assistant.")
    ):
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main_token_stream())