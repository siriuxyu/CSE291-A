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

@tool("get_weather", return_direct=False)
def get_weather(city: str) -> str:
    """Query weather for a city."""
    return "Sunny, 75°F"


TOOLS: List[Callable[..., Any]] = [get_weather]
