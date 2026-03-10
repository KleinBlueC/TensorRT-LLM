"""Run web search benchmark in HeavyThinking mode: N parallel research runs + synthesizer, save answers to JSONL.

Reads from benchmarks/datasets (or --dataset path), runs each prompt through the HeavyThinking Web
Researcher (N parallel IterativeResearcher + IntegrativeSynthesizer), and appends id + model_answer
to an output JSONL file for later pass@k evaluation.

Example:
  python -m examples.scaffolding.benchmarks.HeavyThinking.run_web_search_heavy_thinking \\
    --dataset gaia --max_num 5 --n 3 --output ./answers_heavy_thinking.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.HeavyThinking.heavy_thinking_web_researcher import (
    create_heavy_thinking_web_researcher_scaffolding_llm,
)

try:
    from .dataset_utils import load_records, resolve_dataset_path
except ImportError:
    from dataset_utils import load_records, resolve_dataset_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="HeavyThinking web search benchmark (N parallel runs + synthesizer): one synthesis per question, save answers to JSONL.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset short name (gaia, hle, frames, browsecomp_en, browsecomp_zh, xbench_deepsearch) or path to a .jsonl file.",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=None,
        help="Max number of questions to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="answers_heavy_thinking.jsonl",
        help="Output JSONL path for id and model_answer (default: answers_heavy_thinking.jsonl).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        dest="max_parallel_search",
        help="Number of parallel IterativeResearcher runs to synthesize (HeavyThinking N). Default: 3.",
    )
    parser.add_argument(
        "--workspace_log_root",
        type=str,
        default=None,
        help="Optional root directory for workspace logs (reports/tool_calls per run).",
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://0.0.0.0:8082/sse",
        help="MCP server SSE URL for search, scholar, visit, python_run.",
    )
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for generation.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    path = resolve_dataset_path(args.dataset)
    if not path.exists():
        print(f"Error: dataset not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.dataset, max_num=args.max_num)
    if not records:
        print("Error: no records loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {path}")
    print(f"Output: {args.output}, N (max_parallel_search): {args.max_parallel_search}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    mcp_worker = MCPWorker.init_with_urls([args.mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_heavy_thinking_web_researcher_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_parallel_search=args.max_parallel_search,
        workspace_log_root=args.workspace_log_root,
    )

    with open(out_path, "w", encoding="utf-8") as out_file:
        for i, rec in enumerate(records):
            problem_id = rec.get("id", f"unknown_{i}")
            prompt = rec.get("prompt", "")
            if not prompt:
                continue
            print(f"[{i + 1}/{len(records)}] {problem_id} (n={args.max_parallel_search}) ...")
            try:
                future = llm.generate_async(prompt)
                result = await future.aresult()
                text = (result.outputs[0].text or "").strip()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                text = ""
            line = json.dumps({"id": problem_id, "model_answer": text}, ensure_ascii=False) + "\n"
            out_file.write(line)
            out_file.flush()

    llm.shutdown()
    generation_worker.shutdown()
    mcp_worker.shutdown()
    print(f"Done. Wrote {len(records)} answers to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
