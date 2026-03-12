# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Dataset-specific preprocessing for general evaluation.

Each benchmark turns a task entry (from JSON) into a single prompt string suitable
for ScaffoldingLlm input. Logic aligns with General-AgentBench run.py formatting
so that the same task-file format yields equivalent agent prompts.
"""

from typing import Any

from .types import TaskEntry


def _task_dict(entry: TaskEntry) -> dict[str, Any]:
    """Get the task payload: entry['task'] or entry itself."""
    task = entry.get("task", entry)
    return task if isinstance(task, dict) else {}


def _str(val: Any) -> str:
    """Coerce value to string; empty dict/list becomes empty string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (dict, list)):
        return ""  # Don't str() complex structures
    return str(val)


# -----------------------------------------------------------------------------
# SWE-Bench: instruction from task; wrap with repo context (matches run.py)
# -----------------------------------------------------------------------------

SWEBENCH_PREFIX = """You are a software engineer working on fixing a bug in a Python repository.

The repository is located at /testbed. Please fix the following issue:

"""

SWEBENCH_SUFFIX = """

Use the available tools to:
1. Explore the repository structure
2. Find the relevant files
3. Understand the bug
4. Make the necessary code changes
5. Verify your changes work

When you have completed the fix, use the swebench_finish tool to submit your solution."""


def preprocess_swebench(entry: TaskEntry) -> str:
    """Preprocess SWE-Bench task to prompt (instruction + repo context)."""
    task = _task_dict(entry)
    instruction = _str(task.get("instruction", ""))
    if not instruction:
        instruction = _str(task.get("id", ""))
    return f"{SWEBENCH_PREFIX}{instruction}{SWEBENCH_SUFFIX}"


# -----------------------------------------------------------------------------
# TerminalBench: instruction from task (matches run.py agent.run(instruction=...))
# -----------------------------------------------------------------------------

def preprocess_terminalbench(entry: TaskEntry) -> str:
    """Preprocess TerminalBench task to prompt."""
    task = _task_dict(entry)
    instruction = _str(task.get("instruction", ""))
    if not instruction:
        instruction = _str(task.get("id", ""))
    return instruction


# -----------------------------------------------------------------------------
# MCPBench: fuzzy_description is what the agent sees (matches run.py)
# -----------------------------------------------------------------------------

def preprocess_mcpbench(entry: TaskEntry) -> str:
    """Preprocess MCPBench task to prompt (fuzzy_description)."""
    task = _task_dict(entry)
    prompt = _str(task.get("fuzzy_description", ""))
    if not prompt:
        prompt = _str(task.get("task_description", ""))
    if not prompt:
        prompt = _str(task.get("task_id", task.get("id", "")))
    return prompt


# -----------------------------------------------------------------------------
# Tau2 / Tau2Bench: build from user_scenario.instructions (reason_for_call,
# known_info, task_instructions) so the prompt matches what the User Simulator
# would present as the initial user context.
# -----------------------------------------------------------------------------

def _tau2_instruction_from_task(task: dict[str, Any]) -> str:
    """Build a single instruction string from tau2 task.user_scenario.instructions."""
    scenario = task.get("user_scenario") or {}
    instructions = scenario.get("instructions") or {}
    if not isinstance(instructions, dict):
        return _str(task.get("goal", task.get("id", "")))

    parts = []
    reason = _str(instructions.get("reason_for_call", ""))
    if reason:
        parts.append(f"Reason for call: {reason}")
    known = _str(instructions.get("known_info", ""))
    if known:
        parts.append(f"Known information: {known}")
    task_instr = _str(instructions.get("task_instructions", ""))
    if task_instr:
        parts.append(f"Task instructions: {task_instr}")

    if parts:
        return "\n\n".join(parts)
    # Fallback: goal or id from task (some tau2 tasks may have top-level goal)
    fallback = task.get("goal") or task.get("instruction") or task.get("id") or ""
    return _str(fallback) or ""


def preprocess_tau2(entry: TaskEntry) -> str:
    """Preprocess tau2-bench task to prompt (user_scenario.instructions)."""
    task = _task_dict(entry)
    return _tau2_instruction_from_task(task)


def preprocess_tau2bench(entry: TaskEntry) -> str:
    """Preprocess tau2bench dataset (same format as tau2)."""
    return preprocess_tau2(entry)


# -----------------------------------------------------------------------------
# MathHay: question + JSON output format hint (matches run.py instruction shape)
# -----------------------------------------------------------------------------

MATHHAY_FORMAT = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object should not be wrapped in triple backticks.

Here is the output schema:
```
{"properties": {"reasoning": {"title": "Reasoning", "description": "Solution process.", "type": "string"}, "answer": {"title": "Answer", "description": "The final numerical answer to the question, deduced through reasoning.", "type": "number"}}, "required": ["reasoning", "answer"]}
```"""


def preprocess_mathhay(entry: TaskEntry) -> str:
    """Preprocess MathHay task to prompt (question + JSON format instructions)."""
    task = _task_dict(entry)
    question = _str(task.get("question", ""))
    if not question:
        question = _str(task.get("id", ""))
    return f"Question:\n{question}\n\n{MATHHAY_FORMAT}"


# -----------------------------------------------------------------------------
# Frames: question with <answer></answer> format (same output format as search)
# -----------------------------------------------------------------------------

def preprocess_frames(entry: TaskEntry) -> str:
    """Preprocess Frames task to prompt (question + answer tag format)."""
    task = _task_dict(entry)
    question = _str(task.get("question", ""))
    if not question:
        question = _str(task.get("id", ""))
    return (
        "You should pay attention to the format of your output. "
        "If you want to give the final answer, put the answer between <answer> and </answer>.\n"
        f"Question: {question}"
    )


# -----------------------------------------------------------------------------
# Search benchmarks (BrowseComp, Mind2Web, WebVoyager): same prompt shape as run.py
# -----------------------------------------------------------------------------

SEARCH_PROMPT_PREFIX = (
    "You should pay attention to the format of your output. "
    "If you want to give the final answer, put the answer between <answer> and </answer>.\n"
    "Question: "
)


def preprocess_search(entry: TaskEntry) -> str:
    """Preprocess search-style task (BrowseComp/Mind2Web/WebVoyager) to prompt."""
    # Entry may be full record (benchmark=search, question at top) or nested task
    question = _str(entry.get("question", ""))
    if not question:
        task = _task_dict(entry)
        question = _str(task.get("question", ""))
    if not question:
        question = _str(entry.get("id", "")) or _str(_task_dict(entry).get("id", ""))
    return f"{SEARCH_PROMPT_PREFIX}{question}"


def preprocess_browsecomp(entry: TaskEntry) -> str:
    """Preprocess BrowseComp task to prompt (search-style)."""
    return preprocess_search(entry)


def preprocess_mind2web(entry: TaskEntry) -> str:
    """Preprocess Mind2Web task to prompt (search-style)."""
    return preprocess_search(entry)


def preprocess_webvoyager(entry: TaskEntry) -> str:
    """Preprocess WebVoyager task to prompt (search-style)."""
    return preprocess_search(entry)


# -----------------------------------------------------------------------------
# Default: try common keys from task payload
# -----------------------------------------------------------------------------

def preprocess_default(entry: TaskEntry) -> str:
    """Default preprocessor: use first present of common task keys."""
    task = _task_dict(entry)
    for key in (
        "instruction",
        "task_description",
        "fuzzy_description",
        "question",
        "query",
        "goal",
        "id",
    ):
        val = task.get(key)
        if val is not None and _str(val):
            return _str(val)
    return str(task) if task else str(entry)


# -----------------------------------------------------------------------------
# Registry: benchmark name (as in JSON "benchmark" field) -> preprocess function
# -----------------------------------------------------------------------------

BENCHMARK_PREPROCESSORS: dict[str, Any] = {
    "swebench": preprocess_swebench,
    "terminalbench": preprocess_terminalbench,
    "mcpbench": preprocess_mcpbench,
    "tau2": preprocess_tau2,
    "tau2bench": preprocess_tau2bench,
    "mathhay": preprocess_mathhay,
    "frames": preprocess_frames,
    "search": preprocess_search,
    "browsecomp": preprocess_browsecomp,
    "mind2web": preprocess_mind2web,
    "webvoyager": preprocess_webvoyager,
}


def get_preprocessor(benchmark: str):
    """Return preprocessor for a benchmark; falls back to default."""
    key = (benchmark or "").strip().lower()
    return BENCHMARK_PREPROCESSORS.get(key, preprocess_default)
