# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
BrowseComp dataset MCP server (Layer 1). Does not implement tools; uses mcp_tools.

Exposes web_search (delegates to mcp_tools.web_search) and internal state tools
(reset_state, get_answer, set_answer) for evaluation. Supports HTTP/SSE transport
(like ParallelThinking); use launch_mcp_server.py or run this module with --host/--port.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# Ensure evaluation dir is on path so mcp_tools is importable
_EVAL_DIR = Path(__file__).resolve().parent.parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mcp_tools.web_search import web_search_impl


class BrowseCompSearchServer:
    """
    Search server for BrowseComp. Web search is implemented in mcp_tools; this
    server only wires the tool and manages state.
    """

    def __init__(self, search_engine: str = "tavily"):
        """
        Args:
            search_engine: Backend for web search ("tavily" or "serper"), passed to mcp_tools.
        """
        self.name = "browsecomp_search"
        self.search_engine = search_engine
        self.mcp = FastMCP(self.name)

        self.search_count = 0
        self.current_answer = None
        self.task_completed = False
        self.sources = []
        self.source_contents = {}

        self._register_tools()

    def reset_state(self) -> None:
        """Reset server state (called before each task)."""
        self.search_count = 0
        self.current_answer = None
        self.task_completed = False
        self.sources = []
        self.source_contents = {}

    def get_answer(self) -> Optional[str]:
        """Get the current submitted answer."""
        return self.current_answer

    def set_answer(self, answer: str) -> None:
        """Set answer (called by external agent)."""
        self.current_answer = answer
        self.task_completed = True

    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.task_completed

    def _extract_urls(self, content: str) -> list:
        """Extract URLs from content (for attribution)."""
        import re
        url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+[^\s<>"\'`|(){}[\].,;:]'
        return re.findall(url_pattern, content)

    def _register_tools(self) -> None:
        """Register tools (web_search delegates to mcp_tools)."""
        server_ref = self

        @self.mcp.tool(name="reset_state")
        def reset_state() -> str:
            """[Internal] Reset the server state."""
            server_ref.reset_state()
            return json.dumps({"status": "success", "message": "State reset"})

        @self.mcp.tool(name="get_answer")
        def get_answer() -> str:
            """[Internal] Get the submitted answer and statistics."""
            return json.dumps({
                "status": "success",
                "answer": server_ref.current_answer,
                "completed": server_ref.task_completed,
                "search_count": server_ref.search_count,
                "sources": server_ref.sources,
                "source_contents": server_ref.source_contents
            })

        @self.mcp.tool(name="set_answer")
        def set_answer(answer: str) -> str:
            """[Internal] Set the answer (called by agent after parsing <answer> tag)."""
            server_ref.current_answer = answer
            server_ref.task_completed = True
            return json.dumps({
                "status": "success",
                "message": "Answer set successfully",
                "answer": answer
            })

        def _do_web_search(query: str) -> str:
            try:
                content = web_search_impl(query, backend=server_ref.search_engine)
                server_ref.search_count += 1
                urls = server_ref._extract_urls(content)
                for url in urls:
                    if url not in server_ref.source_contents:
                        server_ref.sources.append(url)
                        server_ref.source_contents[url] = content[:1000]
                return content
            except Exception as e:
                return json.dumps({"error": "Search failed", "message": str(e)})

        @self.mcp.tool(name="web_search")
        def web_search(query: str) -> str:
            """Search the web for information related to your query."""
            return _do_web_search(query)

        @self.mcp.tool(name="search")
        def search(query: str) -> str:
            """Web search to find information (alias for ParallelThinking iterative researcher)."""
            return _do_web_search(query)


def create_server(search_engine: str = "tavily") -> BrowseCompSearchServer:
    """Create BrowseComp Search Server (uses mcp_tools for web_search)."""
    return BrowseCompSearchServer(search_engine=search_engine)


def create_starlette_app(
    server: BrowseCompSearchServer,
    *,
    debug: bool = False,
) -> Starlette:
    """Create a Starlette app that serves the BrowseComp MCP server over SSE (HTTP)."""
    mcp_server = server.mcp._mcp_server  # noqa: SLF001
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
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Run BrowseComp Search MCP Server over HTTP/SSE (uses mcp_tools)"
    )
    parser.add_argument(
        "--search-engine",
        type=str,
        default=os.environ.get("SEARCH_ENGINE", "tavily"),
        choices=["tavily", "serper"],
        help="Web search backend via mcp_tools (default: tavily)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on")
    args = parser.parse_args()
    print(
        f"Starting BrowseComp Search Server over HTTP/SSE "
        f"(web_search from mcp_tools, backend: {args.search_engine})"
    )
    server = BrowseCompSearchServer(search_engine=args.search_engine)
    print("Registered tools: web_search, [Internal] reset_state, get_answer, set_answer")
    app = create_starlette_app(server, debug=True)
    uvicorn.run(app, host=args.host, port=args.port)
