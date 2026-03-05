import argparse
import asyncio
import logging
import os
from typing import Optional

import openai

from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    MCPCallTask,
    OpenAIToolDescription,
    SystemMessage,
    TaskStatus,
)
from tensorrt_llm.scaffolding.worker import MCPWorker, OpenaiWorker

logger = logging.getLogger(__name__)


class LoggingMCPWorker(MCPWorker):
    """MCPWorker that logs each tool request name and success/failure."""

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        tool_name = task.tool_name or "(unknown)"
        logger.info("MCP request: tool=%s", tool_name)
        status = await super().call_handler(task)
        success = task.result_str is not None
        logger.info("MCP request: tool=%s success=%s", tool_name, success)
        return status

# -----------------------------------------------------------------------------
# Four MCP tools (must match agent_tools/server.py: search, scholar, visit, python)
# -----------------------------------------------------------------------------
TOOL_SEARCH = OpenAIToolDescription(
    name="search",
    description="Web search: fetch information from the internet.",
    parameters={
        "query": {"type": "string", "description": "Search query string."},
    },
)
TOOL_SCHOLAR = OpenAIToolDescription(
    name="scholar",
    description="Academic / scholarly search: find papers and citations.",
    parameters={
        "query": {"type": "string", "description": "Search query string."},
        "limit": {"type": "integer", "description": "Maximum number of results (default 5)."},
    },
)
TOOL_VISIT = OpenAIToolDescription(
    name="visit",
    description="Fetch a URL and return its text content (e.g. for reading a web page).",
    parameters={
        "url": {"type": "string", "description": "Full URL to fetch."},
        "max_chars": {"type": "integer", "description": "Max characters to return (default 50000)."},
    },
)
TOOL_PYTHON = OpenAIToolDescription(
    name="python_run",
    description="Run Python code in a secure sandbox. Returns stdout, stderr, and result.",
    parameters={
        "code": {"type": "string", "description": "Python code in Jupyter-style."},
    },
)

ALL_TOOLS = [TOOL_SEARCH, TOOL_SCHOLAR, TOOL_VISIT, TOOL_PYTHON]

# -----------------------------------------------------------------------------
# Prompts that encourage using all four tools
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a research assistant with access to four tools: search (web search), "
    "scholar (academic papers), visit (fetch URL content), and python (run code). "
    "Use them when the user asks for information or computation."
)

# Four prompts, each asking to call exactly one MCP tool.
PROMPT_SEARCH = (
    "Please use the search tool to look up 'TensorRT-LLM NVIDIA'. "
    "Call the search tool once with that query, then reply with a one-sentence summary of what you found."
)
PROMPT_SCHOLAR = (
    "Please use the scholar tool to find academic papers. "
    "Call the scholar tool with query 'large language model inference' and limit 2. "
    "Then reply with a one-sentence summary of the results."
)
PROMPT_VISIT = (
    "Please use the visit tool to fetch a web page. "
    "Call the visit tool with url 'https://www.nvidia.com/en-us/'. "
    "Then reply with a one-sentence summary of what the page is about."
)
PROMPT_PYTHON = (
    "Please use the python_run tool to run some code. "
    "Call the python tool with code 'print(3284 * 1000)'. "
    "Then reply with the result in one sentence. "
)

PROMPTS_FOUR_TOOLS = [
    ("search", PROMPT_SEARCH),
    ("scholar", PROMPT_SCHOLAR),
    ("visit", PROMPT_VISIT),
    ("python", PROMPT_PYTHON),
]


def _create_openai_worker(openai_base: str, model: str, api_key: Optional[str] = None):
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip() or "tensorrt_llm"
    client = openai.AsyncOpenAI(base_url=openai_base, api_key=key)
    return OpenaiWorker(async_client=client, model=model)


async def run_four_tools(
    mcp_url: str,
    openai_base: str,
    model: str,
    openai_api_key: Optional[str] = None,
) -> None:
    """Run 4 user prompts, each asking to call one MCP tool; run 4x generate_async and print each result."""
    mcp_worker = LoggingMCPWorker.init_with_urls([mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    gen_worker = _create_openai_worker(openai_base, model, api_key=openai_api_key)

    controller = ChatWithMCPController(
        generation_controller=NativeGenerationController(
            sampling_params={"temperature": 0.2, "max_tokens": 2048},
        ),
        system_prompts=[SystemMessage(SYSTEM_PROMPT)],
        max_iterations=2,
        tools=ALL_TOOLS,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: gen_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }
    llm = ScaffoldingLlm(controller, workers=workers)

    try:
        for tool_name, prompt in PROMPTS_FOUR_TOOLS:
            print(f"\n--- Tool: {tool_name} ---")
            print("Prompt:", prompt[:80], "...")
            future = llm.generate_async(prompt)
            result = await future.aresult()
            text = result.outputs[0].text if result.outputs else ""
            print(f"Result ({tool_name}):", text or "(empty)")
    finally:
        await mcp_worker.async_shutdown()
        llm.shutdown(shutdown_workers=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-round or multi-round chat with MCP using a real LLM (all four tools: search, scholar, visit, python)."
    )
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://0.0.0.0:8082/sse",
        help="MCP server SSE URL (default: http://0.0.0.0:8082/sse)",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="tensorrt_llm",
        help="OpenAI-compatible API key (default: tensorrt_llm; or set OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3/Qwen3-30B-A3B",
        help="Model name (default: Qwen3/Qwen3-30B-A3B)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    asyncio.run(
        run_four_tools(
            args.mcp_url,
            args.base_url,
            args.model,
            openai_api_key=args.openai_api_key,
        )
    )


if __name__ == "__main__":
    main()
