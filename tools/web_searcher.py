# tools/web_searcher.py
"""
DuckDuckGo-based web search tool (fallback).

NOTE: This tool is primarily used as a fallback when:
1. Using non-Anthropic models (Claude's built-in web_search is not available)
2. Claude's web search is disabled (enable_web_search=False)

When using Anthropic models with enable_web_search=True (default), Claude uses
its own built-in web_search tool (web_search_20250305) which provides better
integration, encrypted results, and automatic citations.

This tool uses DuckDuckGo to search the web and returns plain text results
that can be used with the web_reader tool to fetch page content.
"""
from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests
import json
import re
import urllib.parse


class WebSearcherInput(BaseModel):
    """Arguments for the web_searcher tool."""
    query: str = Field(..., description="Search query string to find relevant web pages.")
    max_results: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of search results to return (default: 5, max: 20).",
    )


def _search_duckduckgo_v2(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Improved DuckDuckGo search using multiple methods for reliability.
    Returns a list of results with 'title', 'url', and 'snippet' keys.
    """
    results = []
    found_urls = set()
    
    # Method 1: Try DuckDuckGo Instant Answer API
    try:
        ia_url = "https://api.duckduckgo.com/"
        ia_params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        ia_resp = requests.get(ia_url, params=ia_params, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        if ia_resp.status_code == 200:
            ia_data = ia_resp.json()
            
            # Add instant answer if available
            if ia_data.get("AbstractText") and ia_data.get("AbstractURL"):
                url = ia_data.get("AbstractURL", "")
                if url and url not in found_urls:
                    found_urls.add(url)
                    results.append({
                        "title": ia_data.get("Heading", query),
                        "url": url,
                        "snippet": ia_data.get("AbstractText", "")
                    })
            
            # Add related topics
            for topic in ia_data.get("RelatedTopics", []):
                if len(results) >= max_results:
                    break
                if isinstance(topic, dict):
                    url = topic.get("FirstURL", "")
                    text = topic.get("Text", "")
                    if url and url not in found_urls and text:
                        found_urls.add(url)
                        title = text.split(" - ")[0] if " - " in text else text[:100]
                        results.append({
                            "title": title[:150],
                            "url": url,
                            "snippet": text[:300]
                        })
    except Exception as e:
        pass  # Continue to other methods
    
    # Method 2: Use DuckDuckGo Lite (more reliable HTML structure)
    if len(results) < max_results:
        try:
            # Use the lite version which has simpler HTML
            lite_url = "https://lite.duckduckgo.com/lite/"
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            resp = requests.get(lite_url, params=params, headers=headers, timeout=12)
            if resp.status_code == 200:
                html = resp.text
                
                # DuckDuckGo Lite uses simpler HTML structure
                # Look for result links in table format
                # Pattern: <a class="result-link" href="URL">Title</a>
                link_patterns = [
                    r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
                    r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*result-link[^"]*"[^>]*>([^<]+)</a>',
                    r'<a[^>]*href="([^"]+)"[^>]*rel="nofollow"[^>]*>([^<]+)</a>',
                ]
                
                for pattern in link_patterns:
                    for match in re.finditer(pattern, html, re.IGNORECASE):
                        if len(results) >= max_results:
                            break
                        url = match.group(1).strip()
                        title = match.group(2).strip()
                        
                        # Skip invalid URLs
                        if not url or url.startswith("/") or "duckduckgo.com" in url.lower():
                            continue
                        
                        # Handle relative URLs
                        if url.startswith("//"):
                            url = "https:" + url
                        elif not url.startswith("http"):
                            continue
                        
                        # Clean up redirect URLs
                        if "/l/?kh=" in url or "uddg=" in url:
                            url_match = re.search(r'uddg=([^&]+)', url)
                            if url_match:
                                url = urllib.parse.unquote(url_match.group(1))
                        
                        if url not in found_urls and url.startswith("http"):
                            found_urls.add(url)
                            results.append({
                                "title": title[:200] if title else "Untitled",
                                "url": url,
                                "snippet": ""  # Lite version doesn't have easy snippet extraction
                            })
                    
                    if len(results) >= max_results:
                        break
        except Exception as e:
            pass  # Continue to fallback
    
    # Method 3: Fallback - try regular HTML search with better parsing
    if len(results) < max_results:
        try:
            html_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            resp = requests.get(html_url, params=params, headers=headers, timeout=12)
            if resp.status_code == 200:
                html = resp.text
                
                # More comprehensive pattern matching
                # Look for any links that look like search results
                all_links = re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html)
                
                for url, title in all_links:
                    if len(results) >= max_results:
                        break
                    
                    url = url.strip()
                    title = title.strip()
                    
                    # Skip invalid URLs
                    if not url or url.startswith("/") or "duckduckgo.com" in url.lower():
                        continue
                    
                    # Handle redirect URLs
                    if "/l/?kh=" in url or "uddg=" in url:
                        url_match = re.search(r'uddg=([^&]+)', url)
                        if url_match:
                            url = urllib.parse.unquote(url_match.group(1))
                    
                    # Only accept http/https URLs
                    if url not in found_urls and (url.startswith("http://") or url.startswith("https://")):
                        # Basic validation: URL should look like a real website
                        if "." in url and len(url) > 10 and len(title) > 3:
                            found_urls.add(url)
                            results.append({
                                "title": title[:200],
                                "url": url,
                                "snippet": ""
                            })
        except Exception:
            pass  # Return what we have
    
    return results[:max_results]


@tool("web_searcher", args_schema=WebSearcherInput, return_direct=False)
def web_searcher(query: str, max_results: int = 5) -> str:
    """
    Search the web for information and return relevant URLs and snippets.
    
    Use this tool when you need to:
    - Find current news, articles, or information on the web
    - Search for specific topics, headlines, or recent events
    - Discover URLs related to a search query
    - Look up information that requires web search
    
    After getting search results, you can use the web_reader tool to read the content
    from the URLs returned by this search.
    
    Args:
        query: The search query string (e.g., "electric vehicle charging news", 
               "today's headlines about X", "recent articles on Y")
        max_results: Maximum number of results to return (default: 5, max: 20)
    
    Returns:
        A JSON string with search results containing:
        - query: The original search query
        - results: List of results, each with:
          - title: Page title
          - url: Page URL (use this with web_reader to read the page)
          - snippet: Brief description/snippet (if available)
        - count: Number of results returned
    
    Example:
        User: "Can you scan the web for today's top headlines about electric vehicles?"
        → Use web_searcher(query="electric vehicle news today", max_results=5)
        → Then use web_reader with the URLs from the results
    """
    try:
        results = _search_duckduckgo_v2(query, max_results)
        
        # Filter out invalid results
        valid_results = []
        for r in results:
            url = r.get("url", "").strip()
            title = r.get("title", "").strip()
            # Only include results with valid URL and title
            if url and url.startswith("http") and title:
                valid_results.append({
                    "title": title,
                    "url": url,
                    "snippet": r.get("snippet", "").strip()
                })
        
        if not valid_results:
            return json.dumps({
                "query": query,
                "results": [],
                "message": "No search results found. The search may have failed or the query returned no matches. Try rephrasing your query or using more specific keywords.",
                "count": 0
            }, ensure_ascii=False)
        
        return json.dumps({
            "query": query,
            "results": valid_results,
            "count": len(valid_results)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        # Return a helpful error message
        error_msg = str(e)
        return json.dumps({
            "query": query,
            "error": f"Web search encountered an error: {error_msg}",
            "results": [],
            "count": 0,
            "message": "The search service may be temporarily unavailable. Please try again later or rephrase your query."
        }, ensure_ascii=False)

