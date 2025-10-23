import os
import fnmatch
from langchain_core.tools import tool

@tool("file_system_search", return_direct=False)
def file_system_search(query: str, max_results: int = 5) -> str:
    """
    Minimal file system search tool.
    Searches for files matching the query and returns their paths.
    """

    return "/User/example/documents/report.pdf"