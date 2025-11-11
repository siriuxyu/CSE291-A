import os
import re
import fnmatch
from typing import List, Tuple
from langchain_core.tools import tool


def _parse_query(query: str) -> Tuple[str, List[str]]:
    """
    Extract a search root and one or more glob patterns from `query`.

    Supports:
      - Inline root override:  root=/path/to/start
      - Multiple patterns separated by commas/semicolons.
    """
    root = "."
    m = re.search(r'(?:^|[\s,;])root\s*=\s*([^,;]+)', query)
    if m:
        root = os.path.expanduser(m.group(1).strip())
        # Remove the root=... clause from the query string
        query = (query[:m.start()] + query[m.end():]).strip()

    # Split remaining query into patterns
    if query.strip():
        patterns = [p.strip() for p in re.split(r"[;,]", query) if p.strip()]
    else:
        patterns = ["*"]  # match everything if no pattern provided

    return os.path.abspath(root), patterns


@tool("file_system_search", return_direct=False)
def file_system_search(query: str, max_results: int = 5) -> str:
    """
    Minimal file system search tool.

    Searches for files matching shell-style glob pattern(s) in `query` and returns
    their absolute paths (newline-separated).

    `query` supports:
      • Comma/semicolon separated patterns (e.g., "*.py, README*")
      • Optional root override anywhere: root=/path/to/start
    Patterns are matched against both filenames and the path relative to the root.

    Args:
        query: Pattern(s) and optional root (see above).
        max_results: Maximum number of paths to return.

    Returns:
        Newline-separated absolute paths (string). If none found, returns a short note.
    """
    root, patterns = _parse_query(query)

    if not os.path.isdir(root):
        return f"[file_system_search] Root directory does not exist: {root}"

    # Common noisy folders to skip
    PRUNE_DIRS = {
        ".git", ".hg", ".svn",
        "__pycache__", ".mypy_cache",
        "node_modules", ".venv", "venv",
        "dist", "build"
    }

    results: List[str] = []

    def _onerror(_err):
        # Ignore unreadable directories
        return

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_onerror, followlinks=False):
        # Prune for speed/noise
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]

        # Check each file against all patterns
        for name in filenames:
            full_path = os.path.join(dirpath, name)
            rel_path = os.path.relpath(full_path, root)

            for pat in patterns:
                # Match either the bare filename or the relative path
                if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
                    results.append(os.path.abspath(full_path))
                    break  # avoid duplicates if multiple patterns match

            if len(results) >= max_results:
                break

        if len(results) >= max_results:
            break

    if not results:
        return f"[file_system_search] No matches for {patterns} under {root}"

    leftover = ""
    if len(results) > max_results:
        leftover = f"\n[file_system_search] ... (+{len(results) - max_results} more not shown)"

    return "\n".join(results[:max_results]) + leftover
