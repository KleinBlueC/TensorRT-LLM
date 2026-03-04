# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse

import uvicorn
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from tools import python_run as python_impl
from tools import scholar as scholar_impl
from tools import search as search_impl
from tools import visit as visit_impl

load_dotenv()

mcp = FastMCP("agent_tools")


@mcp.tool()
async def search(query: str) -> str:
    """Web search: fetch information from the internet.

    Args:
        query: Search query string.
    """
    return await search_impl.run(query)


@mcp.tool()
async def scholar(query: str, limit: int = 5) -> str:
    """Academic / scholarly search: find papers and citations.

    Args:
        query: Search query string.
        limit: Maximum number of results (default 5).
    """
    return await scholar_impl.run(query, limit=limit)


@mcp.tool()
async def visit(url: str, max_chars: int = 50000) -> str:
    """Fetch a URL and return its text content (e.g. for reading a web page).

    Args:
        url: Full URL to fetch.
        max_chars: Maximum characters of content to return (default 50000).
    """
    return await visit_impl.run(url, max_chars=max_chars)


@mcp.tool()
async def python(code: str) -> str:
    """Run Python code in a secure sandbox. Returns stdout, stderr, and result.

    Args:
        code: Python code in Jupyter-style (e.g. single statements or blocks).
    """
    return await python_impl.run(code)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that serves the MCP server over SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: SLF001

    parser = argparse.ArgumentParser(description="Run Agent Tools MCP server (SSE)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8083, help="Port to listen on")
    args = parser.parse_args()

    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port)
