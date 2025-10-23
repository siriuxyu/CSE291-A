"""
Test the translation tool.
"""

import asyncio
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph import graph
from context import Context

async def test_translation():
    """Test the translation tool"""
    
    # Test message - request translation
    test_message = "Translate 'Hello, how are you?' to Chinese"
    
    print("=" * 60)
    print("Test translation tool")
    print("=" * 60)
    print(f"User message: {test_message}")
    print()
    
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": test_message}]},
        context=Context(system_prompt="You are a helpful AI assistant.")
    ):
        print("Chunk:", chunk)
        print()

if __name__ == "__main__":
    asyncio.run(test_translation())

