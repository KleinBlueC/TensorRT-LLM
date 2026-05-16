# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Evaluation servers: dataset adapters, HTTP/SSE launcher, and optional stdio worker.

Layer 1: dataset_servers (BrowseComp, Mind2Web, WebVoyager). BrowseComp uses HTTP/SSE:
run `python -m servers.launch_mcp_server --dataset browsecomp` then connect with
tensorrt_llm.scaffolding.MCPWorker and --mcp_url in run_browsecomp.py. StdioMCPWorker
remains for other benchmarks that use stdio MCP configs.
"""

from typing import List

from .dataset_servers import (
    get_dataset_server_factory,
    get_mcp_tools_for_task,
    list_dataset_benchmarks,
    register_dataset,
)
from .stdio_mcp_worker import StdioMCPWorker


def get_stdio_configs_for_task(task_entry: dict) -> List[dict]:
    """Return list of stdio configs for the given task (e.g. mcpbench). SWE-Bench uses HTTP only."""
    return []


def get_stdio_configs_for_worker(server_names: List[str]) -> List[dict]:
    """Return list of stdio configs for the given server names. SWE-Bench uses HTTP only."""
    return []


def list_available_servers() -> List[str]:
    """List available MCP server names (e.g. for stdio worker)."""
    return ["swebench"]


__all__ = [
    "get_dataset_server_factory",
    "get_mcp_tools_for_task",
    "get_stdio_configs_for_task",
    "get_stdio_configs_for_worker",
    "list_available_servers",
    "list_dataset_benchmarks",
    "register_dataset",
    "StdioMCPWorker",
]
