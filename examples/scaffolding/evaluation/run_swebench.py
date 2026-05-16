# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Run SWE-Bench: HTTP/SSE MCP server + agent loop with logging and evaluation.

Flow:
  1. Start LLM server (e.g. trtllm-serve on port 8000).
  2. Start SWE-Bench MCP server (HTTP/SSE):
       python -m servers.launch_mcp_server --dataset swebench --port 8083
  3. Run (from examples/scaffolding/evaluation or PYTHONPATH set):
       python run_swebench.py --base_url http://localhost:8000/v1 --mcp_url http://localhost:8083/sse --dataset datasets/swebench_benchmark.json --log_dir log/swebench

Per task: __swebench_switch_container(task_id) -> agent run -> __swebench_run_tests -> log + record.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent.parent.parent
for path in (str(_REPO_ROOT), str(_EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import OpenAIToolDescription, SystemMessage

from gateway import GeneralEvalGateway, load_tasks_from_json
from gateway.types import TaskRunResult

# -----------------------------------------------------------------------------
# SWE-Bench tool descriptions for the LLM (agent-visible tools only)
# -----------------------------------------------------------------------------
TOOL_EXECUTE_BASH = OpenAIToolDescription(
    name="swebench_execute_bash",
    description="Execute a bash command in the SWE-Bench Docker container (runs in /testbed).",
    parameters={
        "command": {"type": "string", "description": "The bash command to execute."},
        "is_input": {"type": "string", "description": "If 'true', command is input to running process. Default 'false'."},
        "timeout": {"type": "number", "description": "Optional timeout in seconds (default 120)."},
    },
)
TOOL_STR_REPLACE_EDITOR = OpenAIToolDescription(
    name="swebench_str_replace_editor",
    description="View, create, or edit files in the repo. Commands: view, create, str_replace, insert, undo_edit.",
    parameters={
        "command": {"type": "string", "description": "One of: view, create, str_replace, insert, undo_edit."},
        "path": {"type": "string", "description": "Absolute path to file or directory, e.g. /testbed/file.py."},
        "file_text": {"type": "string", "description": "Required for create."},
        "old_str": {"type": "string", "description": "Required for str_replace."},
        "new_str": {"type": "string", "description": "For str_replace (optional) and insert (required)."},
        "insert_line": {"type": "integer", "description": "Required for insert."},
        "view_range": {"type": "array", "description": "Optional for view, e.g. [11, 12] for lines 11-12."},
    },
)
TOOL_FINISH = OpenAIToolDescription(
    name="swebench_finish",
    description="Signal task completion and generate git diff patch. Call when the fix is done.",
    parameters={
        "message": {"type": "string", "description": "Final message summarizing the changes made."},
    },
)
SWEBENCH_TOOLS = [TOOL_EXECUTE_BASH, TOOL_STR_REPLACE_EDITOR, TOOL_FINISH]

SYSTEM_PROMPT = (
    "You are a software engineer fixing a bug in a Python repository at /testbed. "
    "Use swebench_execute_bash to run commands, swebench_str_replace_editor to view/edit files, "
    "and swebench_finish when you have completed the fix to submit your solution."
)


async def call_mcp_tool_once(mcp_url: str, tool_name: str, args: dict) -> str:
    """Call a single MCP tool via HTTP/SSE and return the tool result text."""
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    async with sse_client(mcp_url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            if result.content and len(result.content) > 0:
                return result.content[0].text or ""
    return ""


def _extract_task_id(entry: dict) -> str:
    task = entry.get("task", entry)
    if isinstance(task, dict):
        for key in ("id", "instance_id", "task_id"):
            if key in task and task[key]:
                return str(task[key])
    return entry.get("id", "") or ""


def _result_to_record(
    task_id: str,
    benchmark: str,
    run_result: TaskRunResult,
    resolved: Optional[bool],
    test_result_json: Optional[str],
) -> dict:
    return {
        "id": task_id,
        "benchmark": benchmark,
        "question": getattr(run_result, "question", None) or (run_result.prompt or "")[:200],
        "model_output": run_result.output_text or "",
        "error": run_result.error,
        "resolved": resolved,
        "test_result": test_result_json,
    }


def setup_logging(log_dir: Optional[Path] = None, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure root logger: stdout + optional file under log_dir or log_file. Returns logger."""
    log = logging.getLogger("run_swebench")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    out = logging.StreamHandler(sys.stdout)
    out.setLevel(logging.INFO)
    out.setFormatter(fmt)
    log.addHandler(out)
    if log_dir or log_file:
        path = log_file or (log_dir / f"swebench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        log.info("Log file: %s", path)
    return log


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-Bench with HTTP/SSE MCP server, agent loop, and logging."
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://localhost:8083/sse",
        help="SWE-Bench MCP server SSE URL",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_EVAL_DIR / "datasets" / "swebench_benchmark.json",
        help="Path to benchmark JSON (list of {benchmark, task})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: swebench_results.json in cwd)",
    )
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_iterations", type=int, default=20, help="Max tool-call rounds per task")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory for log file (default: no file log)",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Explicit log file path (overrides --log_dir)",
    )
    args = parser.parse_args()

    log = setup_logging(log_dir=args.log_dir, log_file=args.log_file)
    log.info("SWE-Bench run started: mcp_url=%s dataset=%s", args.mcp_url, args.dataset)

    out_path = args.output or Path("swebench_results.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_entries = load_tasks_from_json(args.dataset)
    if args.max_tasks is not None:
        task_entries = task_entries[: args.max_tasks]
    log.info("Loaded %d tasks", len(task_entries))

    mcp_worker = MCPWorker.init_with_urls([args.mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    gen_worker = TRTOpenaiWorker(client, args.model)
    controller = ChatWithMCPController(
        generation_controller=NativeGenerationController(
            sampling_params={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        ),
        system_prompts=[SystemMessage(SYSTEM_PROMPT)],
        max_iterations=args.max_iterations,
        tools=SWEBENCH_TOOLS,
    )
    workers = {
        NativeGenerationController.WorkerTag.GENERATION: gen_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }
    llm = ScaffoldingLlm(controller, workers=workers)
    gateway = GeneralEvalGateway(scaffolding_llm=llm)

    results: List[dict] = []
    try:
        for i, entry in enumerate(task_entries):
            task_id = _extract_task_id(entry)
            benchmark = (entry.get("benchmark") or "swebench").strip().lower()
            log.info("[%d/%d] task_id=%s switch_container ...", i + 1, len(task_entries), task_id)

            switch_out = await call_mcp_tool_once(
                args.mcp_url,
                "__swebench_switch_container",
                {"task_id": task_id, "no_rebuild": True},
            )
            try:
                switch_data = json.loads(switch_out) if switch_out.strip() else {}
            except json.JSONDecodeError:
                switch_data = {}
            if not switch_data.get("success"):
                log.warning("switch_container failed for %s: %s", task_id, switch_out[:500])
                results.append(_result_to_record(task_id, benchmark, TaskRunResult(
                    task_id=task_id, benchmark=benchmark, domain="", dataset="",
                    prompt="", output_text="", error=f"switch_container failed: {switch_out[:200]}",
                    question="", golden_answer="",
                ), None, switch_out))
                continue

            log.info("[%d/%d] task_id=%s running agent ...", i + 1, len(task_entries), task_id)
            run_result = await gateway.run_task(entry)
            log.info(
                "[%d/%d] task_id=%s agent done output_len=%s",
                i + 1, len(task_entries), task_id, len(run_result.output_text or ""),
            )

            tests_out = await call_mcp_tool_once(args.mcp_url, "__swebench_run_tests", {})
            resolved: Optional[bool] = None
            try:
                tests_data = json.loads(tests_out) if tests_out.strip() else {}
                resolved = tests_data.get("resolved") if isinstance(tests_data.get("resolved"), bool) else None
                if resolved is None:
                    resolved = tests_data.get("passed")
                log.info("[%d/%d] task_id=%s run_tests resolved=%s", i + 1, len(task_entries), task_id, resolved)
            except json.JSONDecodeError:
                log.warning("run_tests not JSON for %s: %s", task_id, (tests_out or "")[:200])

            record = _result_to_record(task_id, benchmark, run_result, resolved, tests_out[:2000] if tests_out else None)
            results.append(record)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log.info("[%d/%d] task_id=%s saved to %s", i + 1, len(task_entries), task_id, out_path)
    finally:
        llm.shutdown()
        gen_worker.shutdown()
        await mcp_worker.async_shutdown()

    log.info("Done. %d results written to %s", len(results), out_path)


if __name__ == "__main__":
    asyncio.run(main())
