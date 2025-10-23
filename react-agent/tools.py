"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

# from langchain_tavily import TavilySearch
# from langgraph.runtime import get_runtime

# from context import Context


# async def search(query: str) -> Optional[dict[str, Any]]:
#     """Search for general web results.

#     This function performs a search using the Tavily search engine, which is designed
#     to provide comprehensive, accurate, and trusted results. It's particularly useful
#     for answering questions about current events.
#     """
#     pass
    # runtime = get_runtime(Context)
    # wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    # return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from context import Context
from utils import load_chat_model


@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Query weather for a city."""
    return "Sunny, 75°F"


@tool("translate_text", return_direct=False)
async def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language.
    
    This function translates the provided text to the specified language
    while preserving the original meaning and structure.
    
    Args:
        text: The text to translate
        target_language: The target language for translation (e.g., "French", "Spanish", "Chinese")
        
    Returns:
        The translated text
    """
    try:
        # Initialize the LLM using a default model
        llm = load_chat_model("anthropic/claude-sonnet-4-5-20250929")
        
        # Create a prompt for translation
        prompt = f"""Please translate the following text to {target_language}: {text}

Requirements:
- Follow the original text structure and meaning
- Don't add additional information
- Maintain the tone and style of the original
- If the text is already in {target_language}, return it as is
"""
        
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        return response.content
    except Exception as e:
        print(f"Error translating text: {e}")
        return f"Sorry, I encountered an error while translating: {str(e)}"


TOOLS: List[Callable[..., Any]] = [get_weather, translate_text]
