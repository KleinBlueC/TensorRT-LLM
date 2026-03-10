"""MCP tool registration for ParallelThinking: tool names and OpenAI-style descriptions."""

from enum import Enum

from tensorrt_llm.scaffolding.task import OpenAIToolDescription


class ToolName(Enum):
    """Registered MCP tools for ParallelThinking."""

    SEARCH = "search"
    SCHOLAR = "scholar"
    VISIT = "visit"
    PYTHON_RUN = "python_run"
    ANSWER = "answer"


# Virtual tool: when the model "calls" this, we treat as final answer (no MCP).
ANSWER_TOOL_NAME = "answer"

REGISTERED_MCP_TOOLS = {t.value for t in ToolName}

# OpenAI-style tool descriptions for ChatTask (actor chooses one tool per turn).
# Includes "answer" so the model can end the loop without calling a real tool.
REGISTERED_MCP_TOOL_DESCRIPTIONS = [
    OpenAIToolDescription(
        name="search",
        description="Web search to find information.",
        parameters={
            "query": {"type": "string", "description": "Search query"},
        },
    ),
    OpenAIToolDescription(
        name="scholar",
        description="Academic/scholarly search for papers and citations.",
        parameters={
            "query": {"type": "string", "description": "Search query"},
        },
    ),
    OpenAIToolDescription(
        name="visit",
        description="Fetch and read content from a specific URL.",
        parameters={
            "url": {"type": "string", "description": "URL to fetch"},
        },
    ),
    OpenAIToolDescription(
        name="python_run",
        description="Run Python code (e.g. computations, data processing).",
        parameters={
            "code": {"type": "string", "description": "Python code to run"},
        },
    ),
    OpenAIToolDescription(
        name=ANSWER_TOOL_NAME,
        description=(
            "Call this when you have sufficient evidence to output the final "
            "answer. No further tools needed; ends the research and returns "
            "the current report to the user."
        ),
        parameters={},
    ),
]
