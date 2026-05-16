# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Launch an evaluation MCP server over HTTP/SSE.

Select which dataset server to run via --dataset.
- browsecomp: HTTP/SSE (default port 8082).
- swebench: HTTP/SSE (use --port 8083 to avoid clash with browsecomp).

Usage:
  From examples/scaffolding/evaluation (or with PYTHONPATH including this dir):
    python -m servers.launch_mcp_server --dataset browsecomp --port 8082
    python -m servers.launch_mcp_server --dataset swebench --port 8083
  Or run the SWE-Bench server directly:
    python -m servers.swebench_server --port 8083
"""

import argparse
import os
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


def _launch_browsecomp(host: str, port: int, search_engine: str, debug: bool) -> None:
    from servers.dataset_servers.browsecomp_server import (
        create_server,
        create_starlette_app,
    )

    server = create_server(search_engine=search_engine)
    app = create_starlette_app(server, debug=debug)
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def _launch_swebench_http(host: str, port: int, debug: bool) -> None:
    """Run SWE-Bench MCP server over HTTP/SSE (same transport as BrowseComp)."""
    import uvicorn
    from servers.swebench_server import create_server, create_starlette_app
    server = create_server()
    app = create_starlette_app(server, debug=debug)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch an evaluation MCP server (HTTP/SSE or stdio)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("MCP_DATASET", "browsecomp"),
        choices=["browsecomp", "swebench"],
        help="Dataset server to launch (default: browsecomp)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (browsecomp only)")
    parser.add_argument("--port", type=int, default=8082, help="Port to listen on (browsecomp only)")
    parser.add_argument(
        "--search-engine",
        type=str,
        default=os.environ.get("SEARCH_ENGINE", "tavily"),
        choices=["tavily", "serper"],
        help="Web search backend for browsecomp (default: tavily)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Starlette in debug mode (browsecomp only)",
    )
    args = parser.parse_args()

    if args.dataset == "browsecomp":
        print(
            f"Launching BrowseComp MCP server at http://{args.host}:{args.port}/sse "
            f"(search_engine={args.search_engine})"
        )
        _launch_browsecomp(
            host=args.host,
            port=args.port,
            search_engine=args.search_engine,
            debug=args.debug,
        )
    elif args.dataset == "swebench":
        print(f"Launching SWE-Bench MCP server at http://{args.host}:{args.port}/sse")
        _launch_swebench_http(host=args.host, port=args.port, debug=args.debug)
    else:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
