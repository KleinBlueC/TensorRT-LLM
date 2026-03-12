# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Unified agent multi-task evaluation gateway.

Use GeneralEvalGateway with a ScaffoldingLlm instance and a dataset JSON path
under evaluation/datasets/ to run evaluation and collect results.
"""

from .agent_eval_gateway import GeneralEvalGateway, load_tasks_from_json
from .preprocessors import (
    BENCHMARK_PREPROCESSORS,
    get_preprocessor,
)
from .types import TaskEntry, TaskRunResult

__all__ = [
    "GeneralEvalGateway",
    "load_tasks_from_json",
    "get_preprocessor",
    "BENCHMARK_PREPROCESSORS",
    "TaskEntry",
    "TaskRunResult",
]
