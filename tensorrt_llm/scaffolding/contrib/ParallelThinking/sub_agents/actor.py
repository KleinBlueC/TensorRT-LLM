import copy
import json
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage, UserMessage

from tensorrt_llm.scaffolding.contrib.ParallelThinking.mcp.tool_registration import (
    ANSWER_TOOL_NAME,
    REGISTERED_MCP_TOOLS,
    REGISTERED_MCP_TOOL_DESCRIPTIONS,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.prompts import (
    actor_system_prompt,
)

if TYPE_CHECKING:
    from tensorrt_llm.scaffolding.contrib.ParallelThinking.memory.workspace import (
        Workspace,
    )


class Action(Enum):
    """Actor decision: either answer (done) or call a tool (continue loop)."""

    ANSWER = "Answer"
    TOOL_CALL = "Tool Call"


def _tool_call_name(tc: Any) -> Optional[str]:
    if hasattr(tc, "function") and hasattr(tc.function, "name"):
        return getattr(tc.function, "name", None)
    if isinstance(tc, dict):
        return (tc.get("function") or {}).get("name")
    return None


def _tool_call_arguments(tc: Any) -> dict:
    raw = None
    if hasattr(tc, "function") and hasattr(tc.function, "arguments"):
        raw = getattr(tc.function, "arguments", None)
    elif isinstance(tc, dict):
        raw = (tc.get("function") or {}).get("arguments")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def parse_actor_response(
    message: Any,
) -> Tuple[Action, Optional[List[str]], Optional[List[dict]]]:
    """Parse actor decision from the last response message.

    Parse all tool calls; if any name is **answer**, assert there is exactly
    one tool call (answer only), then return Answer. Otherwise return
    (Action.TOOL_CALL, list of names, list of args) for all tool calls.
    """
    tool_calls = getattr(message, "tool_calls", None)

    # TODO: support more robust tool call parsing logic here
    assert tool_calls is not None and len(tool_calls) >= 1, (
        f"Actor response must contain at least one tool call; "
        f"got tool_calls={tool_calls!r}"
    )
    names = [(_tool_call_name(tc) or "").strip() for tc in tool_calls]
    assert all(n in REGISTERED_MCP_TOOLS for n in names), [
        f"Tool name {n} is not in REGISTERED_MCP_TOOLS" for n in names if n not in REGISTERED_MCP_TOOLS
    ]
    args_list = [_tool_call_arguments(tc) for tc in tool_calls]

    if ANSWER_TOOL_NAME in names:
        assert len(tool_calls) == 1, (
            f"When choosing the answer tool, there must be exactly one tool "
            f"call; got {len(tool_calls)} tool_calls"
        )
        return (Action.ANSWER, None, None)

    return (Action.TOOL_CALL, names, args_list)


# Label for the thinker output in the actor's input (user-visible).
THINKER_INPUT_LABEL = "This is the output of the Thinker sub-agent for reference."


class ActorController(NativeGenerationController):
    """Sub-controller for the Actor stage.

    Input: actor system prompt, workspace.to_messages() (after reporter has
    updated), and this round's thinker output (with a label). Tools are passed
    so the model can choose Tool Call or Final Answer. The actor decides
    whether to call a tool or output the final answer directly.
    """

    def __init__(
        self,
        workspace: Optional["Workspace"] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.workspace = workspace
        self.last_chat_task: Optional[ChatTask] = None

    def clone(self):
        clone = copy.deepcopy(self)
        clone.workspace = self.workspace
        clone.last_chat_task = None
        return clone

    def process(self, tasks: List[Task], **kwargs):
        thinker_output = kwargs.pop("thinker_output", None)
        if thinker_output is None and tasks:
            thinker_output = getattr(tasks[0], "thinker_output", None)
        if self.workspace is None or thinker_output is None:
            yield from super().process(tasks, **kwargs)
            return

        now = datetime.now()
        date_str = f"{now:%a} {now:%b} {now.day}, {now:%Y}"
        system_content = actor_system_prompt.format(date=date_str)
        workspace_messages = self.workspace.to_messages(system_prompt=None)
        thinker_user_message = UserMessage(
            f"[{THINKER_INPUT_LABEL}]\n\n{thinker_output or ''}"
        )
        messages = (
            [SystemMessage(system_content)]
            + workspace_messages
            + [thinker_user_message]
        )
        chat_task = ChatTask.create_from_messages(
            messages,
            tools=REGISTERED_MCP_TOOL_DESCRIPTIONS,
        )
        self.last_chat_task = chat_task
        yield from super().process([chat_task], **kwargs)
