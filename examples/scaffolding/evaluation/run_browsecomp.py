# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Run BrowseComp benchmark: ParallelThinking iterative web researcher + gateway + MCP over HTTP/SSE.

Full flow:
  1. Start LLM server (e.g. trtllm-serve on port 8000).
  2. Start MCP server (HTTP/SSE):
       python -m servers.launch_mcp_server --dataset browsecomp --port 8082
  3. Run benchmark (from examples/scaffolding/evaluation or PYTHONPATH set):
       python run_browsecomp.py --base_url http://localhost:8000/v1 --mcp_url http://localhost:8082/sse
  4. Results: per-task log on stdout; each task result is written to --output JSON as soon as the task finishes (real-time append).

Data flow: dataset JSON -> GeneralEvalGateway (preprocess_search) -> prompt
-> ScaffoldingLlm (IterativeResearcher Thinker/Reporter/Actor/Extractor + MCPWorker) -> tool_calls (search, answer) to MCP server
-> TaskRunResult -> stdout log + --output file.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EVAL_DIR.parent.parent.parent
for path in (str(_REPO_ROOT), str(_EVAL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.ParallelThinking.iterative_web_researcher import (
    create_iterative_web_researcher_scaffolding_llm,
)

from gateway import GeneralEvalGateway, load_tasks_from_json
from gateway.types import TaskRunResult


def _result_to_record(r: TaskRunResult) -> dict:
    """Convert TaskRunResult to a dataset-aligned dict for JSON output."""
    return {
        "id": r.task_id,
        "benchmark": r.benchmark,
        "domain": r.domain,
        "dataset": r.dataset,
        "question": r.question,
        "golden_answer": r.golden_answer,
        "model_answer": r.output_text or "",
        "error": r.error,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp benchmark with ParallelThinking iterative web researcher and HTTP/SSE MCP server."
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://localhost:8082/sse",
        help="MCP server SSE URL (start with: python -m servers.launch_mcp_server --dataset browsecomp)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_EVAL_DIR / "datasets" / "browsecomp_benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write results JSON (dataset-aligned: id, question, golden_answer, model_answer, error). Default: browsecomp_results.json in cwd",
    )
    parser.add_argument("--max_tasks", type=int, default=None, help="Cap number of tasks")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        out_path = Path("browsecomp_results.json")

    mcp_worker = MCPWorker.init_with_urls([args.mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    llm = create_iterative_web_researcher_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    gateway = GeneralEvalGateway(scaffolding_llm=llm)

    task_entries = load_tasks_from_json(args.dataset)
    if args.max_tasks is not None:
        task_entries = task_entries[: args.max_tasks]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(out_path, "w") as f:
            f.write("[\n")
            for i, entry in enumerate(task_entries):
                r = await gateway.run_task(entry)
                # Per-task log
                q_preview = (r.question or r.prompt or "")[:80].replace("\n", " ")
                out_preview = (r.output_text or r.error or "")[:80].replace("\n", " ")
                print(f"[task_id={r.task_id}] question: {q_preview}... -> {out_preview}")
                # Real-time write: append one record and flush
                record = _result_to_record(r)
                line = json.dumps(record, ensure_ascii=False)
                if i > 0:
                    f.write(",\n")
                f.write("  " + line)
                f.flush()
            f.write("\n]\n")
        print(f"Wrote {len(task_entries)} results to {out_path}")
    finally:
        llm.shutdown()
        generation_worker.shutdown()
        await mcp_worker.async_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
