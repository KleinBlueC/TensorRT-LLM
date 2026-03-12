"""ReAct tool registration: answer tool and tool result message. No dependency on ParallelThinking."""
from typing import Any, Dict, List, Optional

from tensorrt_llm.scaffolding.task import OpenAIToolDescription, RoleMessage


# Virtual tool: when the model calls this, we treat as final answer (no MCP).
ANSWER_TOOL_NAME = "answer"

# Default tool set: only answer (callers pass custom tools for search etc.).
DEFAULT_ANSWER_TOOL = OpenAIToolDescription(
    name=ANSWER_TOOL_NAME,
    description=(
        "Call this when you have sufficient information to output the final answer. "
        "No further tools needed; ends the loop and returns your answer to the user."
    ),
    parameters={},
)

SEARCH_TOOL = OpenAIToolDescription(
    name="web_search",
    description="Web search to find information.",
    parameters={
        "query": {"type": "string", "description": "Search query"},
    },
)

DEFAULT_TOOL_DESCRIPTIONS: List[OpenAIToolDescription] = [DEFAULT_ANSWER_TOOL, SEARCH_TOOL]
DEFAULT_REGISTERED_TOOL_NAMES: set = {ANSWER_TOOL_NAME, "search"}


class ToolMessage(RoleMessage):
    """Tool role message (OpenAI API tool result). Used to append tool results to ChatTask.messages."""

    def __init__(
        self,
        content: str,
        tool_call_id: str,
        name: str,
        prefix: Optional[str] = None,
    ):
        super().__init__(role="tool", content=content, prefix=prefix)
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }
