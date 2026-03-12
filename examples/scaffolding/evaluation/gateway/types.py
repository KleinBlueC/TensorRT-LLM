# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Type definitions for the general evaluation gateway."""

from dataclasses import dataclass
from typing import Any, Optional

# Task entry format: list of {"benchmark", "domain", "task" [, "dataset"]}
TaskEntry = dict[str, Any]


@dataclass
class TaskRunResult:
    """Result of one task: identity + prompt + output text from ScaffoldingLlm."""

    task_id: str
    benchmark: str
    domain: str
    dataset: str = ""
    prompt: str = ""
    output_text: str = ""
    error: Optional[str] = None
