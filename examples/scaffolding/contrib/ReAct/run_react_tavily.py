"""Run ReAct with a single MCP tool: Tavily search (web_search).

Requires:
  - An OpenAI-compatible LLM server (e.g. trtllm-serve) for generation.
  - A Tavily MCP server exposing the "web_search" tool (e.g. from
    examples/scaffolding/contrib/open_deep_research/TavilyMCP/travily.py).
  - TAVILY_API_KEY set in the environment where the MCP server runs.

Usage:
  1. Start the Tavily MCP server (in a terminal with TAVILY_API_KEY set):
     cd examples/scaffolding/contrib/open_deep_research/TavilyMCP && python travily.py --port 8082
  2. Start your LLM server (e.g. on port 8000).
  3. Run this script:
     python run_react_tavily.py --base_url http://localhost:8000/v1 --mcp_url http://0.0.0.0:8082/sse
"""
import argparse
import asyncio
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.controller import ChatWithMCPController, NativeGenerationController
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import OpenAIToolDescription

from tensorrt_llm.scaffolding.contrib.ReAct import ReActController
from tensorrt_llm.scaffolding.contrib.ReAct.tools import ANSWER_TOOL_NAME
from tensorrt_llm.scaffolding.contrib.ReAct.sub_agents import (
    ReactActorController,
    ReactReasonerController,
)


# Tool set: only Tavily web_search + answer (matches TavilyMCP server tool name "web_search")
TAVILY_SEARCH_TOOL = OpenAIToolDescription(
    name="web_search",
    description="Web search via Tavily. Use this tool to look up information on the internet.",
    parameters={
        "query": {"type": "string", "description": "Search query string."},
    },
)
ANSWER_TOOL = OpenAIToolDescription(
    name=ANSWER_TOOL_NAME,
    description=(
        "Call this when you have enough information to give the final answer. "
        "Ends the loop and returns your answer to the user."
    ),
    parameters={},
)

TAVILY_ONLY_TOOL_DESCRIPTIONS = [TAVILY_SEARCH_TOOL, ANSWER_TOOL]
TAVILY_ONLY_TOOL_NAMES = {"web_search", ANSWER_TOOL_NAME}


def create_react_tavily_llm(
    generation_worker: TRTOpenaiWorker,
    mcp_worker: MCPWorker,
    max_tokens: int = 4096,
    temperature: float = 0.6,
    max_iteration: int = 10,
) -> ScaffoldingLlm:
    """Build a ScaffoldingLlm with ReAct controller and only Tavily search (+ answer) as tools."""
    sampling_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    reasoner_controller = ReactReasonerController(sampling_params=sampling_params)
    actor_controller = ReactActorController(
        tool_descriptions=TAVILY_ONLY_TOOL_DESCRIPTIONS,
        registered_tool_names=TAVILY_ONLY_TOOL_NAMES,
        sampling_params=sampling_params,
    )
    react_controller = ReActController(
        reasoner_controller=reasoner_controller,
        actor_controller=actor_controller,
        max_iteration=max_iteration,
    )
    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }
    return ScaffoldingLlm(react_controller, workers=workers)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ReAct with a single MCP tool: Tavily search (web_search)."
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://0.0.0.0:8082/sse",
        help="MCP server SSE URL (TavilyMCP: web_search tool)",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_iteration", type=int, default=10)
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the current stock price of NVIDIA? Use the search tool once and then answer.",
    )
    return parser.parse_args()


async def main():
    args = parse_arguments()
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    mcp_worker = MCPWorker.init_with_urls([args.mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_react_tavily_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_iteration=args.max_iteration,
    )

    print("Prompt:", args.prompt)
    print("Running ReAct (Tavily search only)...")
    future = llm.generate_async(args.prompt)
    result = await future.aresult()

    assert result.outputs and result.outputs[0].text is not None
    print("\nFinal output:")
    print(result.outputs[0].text)

    llm.shutdown()
    generation_worker.shutdown()
    await mcp_worker.async_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
