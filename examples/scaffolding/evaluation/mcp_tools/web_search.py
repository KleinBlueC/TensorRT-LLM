# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Web search MCP tool: implementation lives here. Backend can be 'tavily' (default) or 'serper'.

Tavily: uses TavilyClient. Set TAVILY_API_KEY.
Serper: stub (not implemented yet).

Dataset servers (browsecomp, mind2web, webvoyager) call web_search_impl() and do not
implement search themselves.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Optional: add evaluation dir to path for imports when run standalone
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

mcp = FastMCP("web_search")

# Default backend for web_search (used by dataset_servers and standalone server)
DEFAULT_WEB_SEARCH_BACKEND = "tavily"


def _search_tavily(query: str) -> str:
    """Sync Tavily search."""
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set"
    client = TavilyClient(api_key)
    response = client.search(query=query)
    search_result = ""
    for result in response["results"]:
        search_result += f"{result['title']}: {result['content']}\n"
    return search_result


def _search_serper(query: str) -> str:
    """Serper backend: stub (not implemented)."""
    return f"[serper] Search for '{query}' is not implemented yet."


def web_search_impl(query: str, backend: str = DEFAULT_WEB_SEARCH_BACKEND) -> str:
    """
    Run web search. Used by dataset_servers and by the standalone MCP server.

    Args:
        query: Search query string.
        backend: "tavily" (default) or "serper".

    Returns:
        Search result text or error message.
    """
    backend = (backend or DEFAULT_WEB_SEARCH_BACKEND).lower()
    if backend == "tavily":
        return _search_tavily(query)
    if backend == "serper":
        return _search_serper(query)
    return f"Error: unknown backend '{backend}' (use tavily or serper)."


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web. Backend is chosen at server start (default: tavily)."""
    backend = os.environ.get("WEB_SEARCH_BACKEND", DEFAULT_WEB_SEARCH_BACKEND).lower()
    return await asyncio.to_thread(web_search_impl, query, backend)


def main():
    parser = argparse.ArgumentParser(description="Web search MCP server (tavily or serper)")
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("WEB_SEARCH_BACKEND", DEFAULT_WEB_SEARCH_BACKEND),
        choices=["tavily", "serper"],
        help="Search backend (default: tavily)",
    )
    args = parser.parse_args()
    os.environ["WEB_SEARCH_BACKEND"] = args.backend
    print(f"Starting web_search MCP server with backend: {args.backend}")
    mcp.run()


if __name__ == "__main__":
    main()
