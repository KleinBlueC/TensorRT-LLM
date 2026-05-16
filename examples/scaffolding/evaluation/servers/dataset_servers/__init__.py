# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Layer 1: Dataset-level server adapters.

Registered datasets: BrowseComp, Mind2Web, WebVoyager. Each has a search server
adapter that exposes web_search and internal state tools. Paths use env vars (no _tmp).
"""

from .registry import (
    get_dataset_server_factory,
    get_mcp_tools_for_task,
    list_dataset_benchmarks,
    register_dataset,
)

__all__ = [
    "get_dataset_server_factory",
    "get_mcp_tools_for_task",
    "list_dataset_benchmarks",
    "register_dataset",
]
