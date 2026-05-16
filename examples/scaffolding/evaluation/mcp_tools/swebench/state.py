# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SWE-Bench server state: container and task info, edit history, paths."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_dataset_path() -> Path:
    """SWE-Bench dataset root (task dirs with docker-compose.yaml). Env SWEBENCH_DATASET_PATH or evaluation/datasets/swebench-verified."""
    _eval_dir = Path(__file__).resolve().parent.parent.parent
    return Path(
        os.environ.get("SWEBENCH_DATASET_PATH", _eval_dir / "datasets" / "swebench-verified")
    )


CONTAINER_TESTBED = "/testbed"
CONTAINER_LOGS = "/logs"
CONTAINER_AGENT_LOGS = "/agent-logs"
CONTAINER_TESTS = "/tests"


class SWEBenchState:
    """Mutable state for one SWE-Bench MCP server instance."""

    __slots__ = (
        "task_id",
        "task_path",
        "container_name",
        "project_name",
        "container_running",
        "edit_history",
        "task_finished",
        "task_result",
        "generated_patch",
        "workspace",
        "logs_path",
        "agent_logs_path",
    )

    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.task_path: Optional[Path] = None
        self.container_name: Optional[str] = None
        self.project_name: Optional[str] = None
        self.container_running = False
        self.edit_history: Dict[str, List[str]] = {}
        self.task_finished = False
        self.task_result: Any = None
        self.generated_patch = ""
        self.workspace = CONTAINER_TESTBED
        self.logs_path: Optional[Path] = None
        self.agent_logs_path: Optional[Path] = None
