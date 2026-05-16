# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SWE-Bench MCP Server over HTTP/SSE; tool logic in mcp_tools.swebench."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mcp_tools.swebench import (
    SWEBenchState,
    execute_bash_impl,
    str_replace_editor_impl,
    finish_impl,
    switch_container_impl,
    run_tests_impl,
    get_patch_impl,
    cleanup_impl,
    get_dataset_path,
)

EXPOSED_TOOLS = [
    "swebench_execute_bash",
    "swebench_str_replace_editor",
    "swebench_finish",
]
INTERNAL_TOOLS = [
    "__swebench_switch_container",
    "__swebench_run_tests",
    "__swebench_get_patch",
    "__swebench_cleanup",
]
ALL_TOOLS = EXPOSED_TOOLS + INTERNAL_TOOLS


class SWEBenchDockerServer:
    """SWE-Bench MCP server over HTTP/SSE; state held here, impl in mcp_tools.swebench."""

    def __init__(self) -> None:
        self.name = "swebench"
        self.state = SWEBenchState()
        self.mcp = FastMCP(self.name)
        self._register_tools()

    def _register_tools(self) -> None:
        state = self.state

        @self.mcp.tool(name="swebench_execute_bash")
        def execute_bash(
            command: str,
            is_input: str = "false",
            timeout: Optional[float] = None,
        ) -> str:
            """Execute a bash command in the SWE-Bench Docker container (runs in /testbed)."""
            return execute_bash_impl(state, command, is_input=is_input, timeout=timeout)

        @self.mcp.tool(name="swebench_str_replace_editor")
        def str_replace_editor(
            command: str,
            path: str,
            file_text: Optional[str] = None,
            old_str: Optional[str] = None,
            new_str: Optional[str] = None,
            insert_line: Optional[int] = None,
            view_range: Optional[list] = None,
        ) -> str:
            """View, create, or edit files (view, create, str_replace, insert, undo_edit)."""
            return str_replace_editor_impl(
                state, command, path,
                file_text=file_text, old_str=old_str, new_str=new_str,
                insert_line=insert_line, view_range=view_range,
            )

        @self.mcp.tool(name="swebench_finish")
        def finish(message: str) -> str:
            """Signal task completion and generate git diff patch."""
            return finish_impl(state, message)

        @self.mcp.tool(name="__swebench_switch_container")
        def switch_container(
            task_id: str,
            output_dir: Optional[str] = None,
            no_rebuild: bool = True,
        ) -> str:
            """[Internal] Switch to a new SWE-Bench task's Docker container."""
            return switch_container_impl(state, task_id, output_dir=output_dir, no_rebuild=no_rebuild)

        @self.mcp.tool(name="__swebench_run_tests")
        def run_tests() -> str:
            """[Internal] Run the SWE-Bench test suite."""
            return run_tests_impl(state)

        @self.mcp.tool(name="__swebench_get_patch")
        def get_patch() -> str:
            """[Internal] Get git diff patch of all changes."""
            return get_patch_impl(state)

        @self.mcp.tool(name="__swebench_cleanup")
        def cleanup() -> str:
            """[Internal] Stop and remove the Docker container."""
            return cleanup_impl(state)


def create_server() -> SWEBenchDockerServer:
    return SWEBenchDockerServer()


def create_starlette_app(server: SWEBenchDockerServer, *, debug: bool = False) -> Starlette:
    """Create Starlette app serving SWE-Bench MCP over SSE (HTTP)."""
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


def main() -> None:
    import argparse
    import uvicorn
    server = create_server()
    app = create_starlette_app(server, debug=True)
    parser = argparse.ArgumentParser(description="Run SWE-Bench MCP Server over HTTP/SSE")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8083, help="Port to listen on")
    args = parser.parse_args()
    print(
        f"[{datetime.now()}] Starting SWEBench MCP Server (HTTP/SSE) at http://{args.host}:{args.port}/sse",
        file=sys.stderr,
    )
    print(f"Exposed tools: {EXPOSED_TOOLS}", file=sys.stderr)
    print(f"Internal tools: {INTERNAL_TOOLS}", file=sys.stderr)
    print(f"SWE-Bench dataset path: {get_dataset_path()}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
