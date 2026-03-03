"""
Example: ParallelThinking with NativeGenerationController for both agents.

Flow:
  1. The same prompt is sent to num_parallel SearchAgents (each a NativeGenerationController).
  2. All search results are gathered and printed individually.
  3. A SynthesisAgent (may use a different model) receives the original prompt +
     collected results and produces a single summarized output.

Supports separate models for search and synthesis via --search_model_dir / --synthesis_model_dir.
"""

import argparse
from enum import Enum

from tensorrt_llm.scaffolding import (
    NativeGenerationController,
    ScaffoldingLlm,
    TRTLLMWorker,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking import (
    ParallelThinkingController,
)

class SearchWorkerTag(Enum):
    GENERATION = "search_generation"


class SynthesisWorkerTag(Enum):
    GENERATION = "synthesis_generation"


class SearchGenerationController(NativeGenerationController):
    WorkerTag = SearchWorkerTag


class SynthesisGenerationController(NativeGenerationController):
    WorkerTag = SynthesisWorkerTag


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        "Run ParallelThinking: parallel search + synthesis with NativeGenerationController."
    )
    parser.add_argument(
        "--search_model_dir",
        type=str,
        default="/code/llm_models/gpt_oss/gpt-oss-20b",
        help="Path to the model used for parallel search agents.",
    )
    parser.add_argument(
        "--synthesis_model_dir",
        type=str,
        # default="/code/llm_models/gpt_oss/gpt-oss-20b",
        default="/code/llm_models/Qwen3/Qwen3-30B-A3B",
        help="Path to the model used for the synthesis agent.",
    )
    parser.add_argument(
        "--num_parallel",
        type=int,
        default=3,
        help="Number of parallel SearchAgent runs (default: 3).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens per generation for search and synthesis.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    search_worker = TRTLLMWorker.init_with_new_llm(
        args.search_model_dir,
        backend="pytorch",
        max_batch_size=args.num_parallel,
        max_num_tokens=2048,
        kv_cache_free_gpu_memory_fraction=0.4, 
    )

    synthesis_worker = TRTLLMWorker.init_with_new_llm(
        args.synthesis_model_dir,
        backend="pytorch",
        max_batch_size=args.num_parallel,
        max_num_tokens=4096,
        kv_cache_free_gpu_memory_fraction=0.4, 
    )

    search_agent = SearchGenerationController(
        sampling_params={
            "temperature": 0.5,
            "max_tokens": args.max_tokens,
        },
    )

    synthesis_agent = SynthesisGenerationController(
        sampling_params={
            "temperature": 0.2,
            "max_tokens": args.max_tokens,
        },
    )

    controller = ParallelThinkingController(
        search_agent=search_agent,
        synthesis_agent=synthesis_agent,
        num_parallel=args.num_parallel,
    )

    workers = {
        SearchWorkerTag.GENERATION: search_worker,
        SynthesisWorkerTag.GENERATION: synthesis_worker,
    }

    llm = ScaffoldingLlm(
        prototype_controller=controller,
        workers=workers,
    )

    prompt = (
        "I have a week off and want to visit Tokyo for the first time. "
        "Plan a 2-day itinerary covering culture, food, and sightseeing. "
        "For each day, list the morning, afternoon, and evening activities."
    )

    print("Prompt:", prompt)
    print(f"Search model:    {args.search_model_dir}")
    print(f"Synthesis model: {args.synthesis_model_dir}")
    print("\n--- Running ParallelThinking (parallel search + synthesis) ---\n")

    results = llm.generate(prompt)

    print("\nmain shutting down...")
    llm.shutdown()
    # TODO: same model should share the same worker
    search_worker.shutdown()
    synthesis_worker.shutdown()
    print("main shut down done")


if __name__ == "__main__":
    main()
