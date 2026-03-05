import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

# Allow importing agent_tools from sibling path (mcp/agent_tools)
_AGENT_TOOLS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "tools")
)
if _AGENT_TOOLS_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_TOOLS_ROOT)

from tools import python_run as python_run_impl
from tools import scholar as scholar_impl
from tools import search as search_impl
from tools import visit as visit_impl

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Initialize FastMCP server for Agent tools (SSE)
mcp = FastMCP("research_synthesis_mcp")


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
async def python_run(code: str, timeout_sec: float = 60.0) -> str:
    """Run Python code in a secure sandbox. Returns stdout, stderr, and result.

    Args:
        code: Python code in Jupyter-style (e.g. single statements or blocks).
        timeout_sec: Maximum execution time in seconds (default 60).
    """
    return await python_run_impl.run(code, timeout_sec=timeout_sec)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can server the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
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
        return Response()

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server  # noqa: WPS437

    parser = argparse.ArgumentParser(description="Run MCP SSE-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on")
    args = parser.parse_args()

    # Bind SSE request handling to MCP server
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
