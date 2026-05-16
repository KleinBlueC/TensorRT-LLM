# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Registry: map benchmark/dataset name -> Layer 1 server + Layer 2 MCP tool matching.

Layer 1: adapter per dataset (BrowseComp, Mind2Web, WebVoyager search servers).
Layer 2: for these datasets the server itself exposes tools (no separate MCP tool list).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# (server_factory, get_mcp_tools_for_task)
# server_factory: () -> server instance (or None if not applicable)
# get_mcp_tools_for_task: (task_entry: dict) -> list of stdio configs for StdioMCPWorker, or None
_DATASET_REGISTRY: Dict[str, Tuple[Optional[Callable[[], Any]], Optional[Callable[[dict], List[dict]]]]] = {}


def register_dataset(
    benchmark: str,
    server_factory: Optional[Callable[[], Any]] = None,
    get_mcp_tools_for_task_fn: Optional[Callable[[dict], List[dict]]] = None,
) -> None:
    """Register a dataset: Layer 1 server factory and optional Layer 2 MCP tool resolver."""
    _DATASET_REGISTRY[benchmark.lower()] = (server_factory, get_mcp_tools_for_task_fn)


def get_dataset_server_factory(benchmark: str) -> Optional[Callable[[], Any]]:
    """Return the Layer 1 server factory for the benchmark, or None."""
    entry = _DATASET_REGISTRY.get(benchmark.lower())
    return entry[0] if entry else None


def get_mcp_tools_for_task(benchmark: str, task_entry: dict) -> Optional[List[dict]]:
    """
    Return Layer 2: list of stdio configs for the task (for mcpbench), or None.
    Used to build StdioMCPWorker when the benchmark uses mcp-bench MCP tools.
    """
    entry = _DATASET_REGISTRY.get(benchmark.lower())
    if not entry or entry[1] is None:
        return None
    return entry[1](task_entry)


def list_dataset_benchmarks() -> List[str]:
    """List all registered dataset/benchmark names."""
    return list(_DATASET_REGISTRY.keys())


def _swebench_get_mcp_tools_for_task(task_entry: dict) -> List[dict]:
    """SWE-Bench uses HTTP/SSE only (no stdio config). Use run_swebench.py with --mcp_url."""
    return []


def _init_default_registry() -> None:
    """Register BrowseComp, Mind2Web, WebVoyager (Layer 1), SWE-Bench (stdio)."""
    from . import browsecomp_server
    from . import mind2web_server
    from . import webvoyager_server

    register_dataset("browsecomp", server_factory=browsecomp_server.create_server, get_mcp_tools_for_task_fn=None)
    register_dataset("mind2web", server_factory=mind2web_server.create_server, get_mcp_tools_for_task_fn=None)
    register_dataset("webvoyager", server_factory=webvoyager_server.create_server, get_mcp_tools_for_task_fn=None)
    register_dataset(
        "swebench",
        server_factory=None,
        get_mcp_tools_for_task_fn=_swebench_get_mcp_tools_for_task,
    )


_init_default_registry()
