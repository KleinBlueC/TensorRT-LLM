# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Unified agent multi-task evaluation gateway.

Runs ScaffoldingLlm over task datasets (JSON under evaluation/datasets/).
Supports multiple benchmarks; per-dataset preprocessing is placeholder.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

from tensorrt_llm.scaffolding import ScaffoldingLlm

from .preprocessors import get_preprocessor
from .types import TaskEntry, TaskRunResult


def _extract_task_id(entry: TaskEntry) -> str:
    """Get a stable task id from entry."""
    task = entry.get("task", entry)
    if isinstance(task, dict):
        for key in ("id", "instance_id", "task_id"):
            if key in task and task[key]:
                return str(task[key])
    return entry.get("id", "") or str(hash(json.dumps(entry, sort_keys=True)))


def load_tasks_from_json(path: Union[str, Path]) -> list[TaskEntry]:
    """Load task entries from a JSON file (General-AgentBench task-file format).

    Expected format: list of {"benchmark", "domain", "task" [, "dataset"]}.

    Args:
        path: Path to JSON file (e.g. datasets/swebench_benchmark.json).

    Returns:
        List of task entry dicts.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


class GeneralEvalGateway:
    """Unified interface to run ScaffoldingLlm on multi-benchmark task datasets."""

    def __init__(
        self,
        scaffolding_llm: ScaffoldingLlm,
        preprocessors: Optional[dict[str, Callable[[TaskEntry], str]]] = None,
    ):
        """Initialize the gateway.

        Args:
            scaffolding_llm: ScaffoldingLlm instance (uses generate_async/aresult).
            preprocessors: Optional map benchmark_name -> fn(entry) -> prompt;
                if not provided, uses default preprocessors from preprocessors module.
        """
        self._scaffolding_llm = scaffolding_llm
        self._preprocessors = preprocessors

    def _get_prompt(self, entry: TaskEntry) -> str:
        benchmark = entry.get("benchmark", "").lower()
        if self._preprocessors and benchmark in self._preprocessors:
            fn = self._preprocessors[benchmark]
        else:
            fn = get_preprocessor(benchmark)
        return fn(entry)

    async def _run_agent(self, prompt: str) -> tuple[str, Optional[str]]:
        """Run ScaffoldingLlm on prompt; returns (output_text, error)."""
        try:
            result = self._scaffolding_llm.generate_async(prompt)
            await result.aresult()
            if result.outputs and len(result.outputs) > 0:
                output_text = result.outputs[0].text or ""
            else:
                output_text = ""
            return (output_text, None)
        except Exception as e:
            return ("", str(e))

    async def run_task(self, entry: TaskEntry) -> TaskRunResult:
        """Run the agent on a single task entry."""
        task_id = _extract_task_id(entry)
        benchmark = entry.get("benchmark", "")
        domain = entry.get("domain", "")
        dataset = entry.get("dataset", "")
        prompt = self._get_prompt(entry)
        task = entry.get("task", entry)
        task_dict = task if isinstance(task, dict) else {}
        question = entry.get("question", "") or task_dict.get("question", "")
        golden_answer = entry.get("golden_answer", "") or task_dict.get("golden_answer", "")

        output_text, error = await self._run_agent(prompt)
        return TaskRunResult(
            task_id=task_id,
            benchmark=benchmark,
            domain=domain,
            dataset=dataset,
            prompt=prompt,
            output_text=output_text,
            error=error,
            question=question,
            golden_answer=golden_answer,
        )

    async def run(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        task_entries: Optional[list[TaskEntry]] = None,
        max_tasks: Optional[int] = None,
    ) -> list[TaskRunResult]:
        """Run the agent on a dataset and return results.

        Either dataset_path or task_entries must be provided.

        Args:
            dataset_path: Path to JSON task file (e.g. datasets/swebench_benchmark.json).
            task_entries: In-memory list of task entries (overrides dataset_path if set).
            max_tasks: Optional cap on number of tasks to run.

        Returns:
            List of TaskRunResult, one per task.
        """
        if task_entries is None:
            if dataset_path is None:
                raise ValueError("Provide either dataset_path or task_entries")
            task_entries = load_tasks_from_json(dataset_path)
        if max_tasks is not None:
            task_entries = task_entries[:max_tasks]

        results: list[TaskRunResult] = []
        for entry in task_entries:
            res = await self.run_task(entry)
            results.append(res)
        return results

    def run_sync(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        task_entries: Optional[list[TaskEntry]] = None,
        max_tasks: Optional[int] = None,
    ) -> list[TaskRunResult]:
        """Synchronous wrapper around run()."""
        return asyncio.run(
            self.run(
                dataset_path=dataset_path,
                task_entries=task_entries,
                max_tasks=max_tasks,
            )
        )
