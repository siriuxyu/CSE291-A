# tools/web_reader_min.py
from __future__ import annotations
import re
import html
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from langchain_core.tools import tool
import requests


class WebReaderInput(BaseModel):
    """Arguments for the web_reader tool."""
    url: HttpUrl = Field(..., description="HTTP/HTTPS URL to fetch.")
    max_chars: int = Field(
        3000,
        ge=200,
        le=20000,
        description="Maximum number of characters to return after cleaning.",
    )
    user_agent: Optional[str] = Field(
        None,
        description="Optional custom User-Agent header.",
    )


def _extract_title(html_text: str) -> Optional[str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    # Collapse whitespace and unescape entities
    title = " ".join(m.group(1).split())
    return html.unescape(title)


def _html_to_text(html_text: str) -> str:
    # Remove scripts/styles
    html_text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text)
    # Remove all tags
    text = re.sub(r"(?s)<[^>]+>", " ", html_text)
    # Unescape entities and collapse whitespace
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


@tool("web_reader", args_schema=WebReaderInput, return_direct=False)
def web_reader(url: str, max_chars: int = 2000, user_agent: Optional[str] = None) -> str:
    """
    Minimal web reader that fetches a URL and returns a cleaned text snippet.
    - Returns a JSON-like plain string: title + first N characters of body text.
    - No external parsing libraries required.
    """
    headers = {"User-Agent": user_agent or "Mozilla/5.0 (web_reader_min)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Web reader error: request failed ({e})"

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" not in ctype.lower():
        # For non-HTML, return first bytes of the response text
        snippet = resp.text[:max_chars]
        return f'{{"title": null, "content": "{snippet}"}}'

    # Decode and clean
    html_text = resp.text
    title = _extract_title(html_text)
    body = _html_to_text(html_text)

    # Build snippet
    snippet = body[:max_chars]
    # Escape quotes/newlines for a compact JSON-like string (still plain text)
    snippet_escaped = snippet.replace("\\", "\\\\").replace('"', '\\"')
    title_escaped = (title or "").replace("\\", "\\\\").replace('"', '\\"') if title else None

    if title_escaped is None:
        return f'{{"title": null, "content": "{snippet_escaped}"}}'
    return f'{{"title": "{title_escaped}", "content": "{snippet_escaped}"}}'
