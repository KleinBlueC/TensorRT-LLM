import copy
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage

from tensorrt_llm.scaffolding.task import OpenAIToolDescription

from tensorrt_llm.scaffolding.contrib.ReAct.prompts import actor_system_prompt
from tensorrt_llm.scaffolding.contrib.ReAct.tools import (
    ANSWER_TOOL_NAME,
    DEFAULT_REGISTERED_TOOL_NAMES,
    DEFAULT_TOOL_DESCRIPTIONS,
)


class Action(Enum):
    """Actor decision: answer (done) or tool call (continue)."""
    ANSWER = "Answer"
    TOOL_CALL = "Tool Call"


def _tool_call_name(tc: Any) -> Optional[str]:
    if hasattr(tc, "function") and hasattr(tc.function, "name"):
        return getattr(tc.function, "name", None)
    if isinstance(tc, dict):
        return (tc.get("function") or {}).get("name")
    return None


def _tool_call_id(tc: Any) -> Optional[str]:
    if hasattr(tc, "id"):
        return getattr(tc, "id", None)
    if isinstance(tc, dict):
        return tc.get("id")
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
    registered_tool_names: Optional[set] = None,
) -> Tuple[Action, Optional[List[str]], Optional[List[dict]], Optional[List[str]]]:
    """Parse actor response: (action, tool_names, tool_args_list, tool_call_ids)."""
    allowed = (
        registered_tool_names
        if registered_tool_names is not None
        else DEFAULT_REGISTERED_TOOL_NAMES
    )
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None or len(tool_calls) == 0:
        content = getattr(message, "content", None) or ""
        logger.warning(
            "Actor response has no tool calls; treating as ANSWER. message.content=%s",
            content,
        )
        return (Action.ANSWER, None, None, None)
    names = [(_tool_call_name(tc) or "").strip() for tc in tool_calls]
    ids = [(_tool_call_id(tc) or "") for tc in tool_calls]
    invalid = [n for n in names if n not in allowed]
    if invalid:
        logger.warning(
            "Tool name(s) not in allowed set: %s (allowed=%s)",
            invalid,
            allowed,
        )
    args_list = [_tool_call_arguments(tc) for tc in tool_calls]

    if ANSWER_TOOL_NAME in names:
        if len(tool_calls) != 1:
            logger.warning(
                "When choosing the answer tool, there must be exactly one tool call; got %d",
                len(tool_calls),
            )
        return (Action.ANSWER, None, None, None)
    return (Action.TOOL_CALL, names, args_list, ids)


class ReactActorController(NativeGenerationController):
    """ReAct actor stage: builds ChatTask from current messages + tools, yields generation."""

    def __init__(
        self,
        tool_descriptions: Optional[List[OpenAIToolDescription]] = None,
        registered_tool_names: Optional[set] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tool_descriptions = (
            tool_descriptions
            if tool_descriptions is not None
            else DEFAULT_TOOL_DESCRIPTIONS
        )
        self.registered_tool_names = (
            registered_tool_names
            if registered_tool_names is not None
            else DEFAULT_REGISTERED_TOOL_NAMES
        )
        self.last_chat_task: Optional[ChatTask] = None

    def clone(self):
        clone = copy.deepcopy(self)
        clone.last_chat_task = None
        return clone

    def process(self, tasks: List[Task], **kwargs):
        chat_task = kwargs.pop("chat_task", None)
        if chat_task is None and tasks:
            chat_task = getattr(tasks[0], "chat_task", None)
        if chat_task is None or not chat_task.messages:
            yield from super().process(tasks, **kwargs)
            return

        now = datetime.now()
        date_str = f"{now:%a} {now:%b} {now.day}, {now:%Y}"
        system_content = actor_system_prompt.format(date=date_str)
        messages = [SystemMessage(system_content)] + list(chat_task.messages)
        step_task = ChatTask.create_from_messages(
            messages,
            tools=self.tool_descriptions,
        )
        self.last_chat_task = step_task
        yield from super().process([step_task], **kwargs)
