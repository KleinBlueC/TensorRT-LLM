"""ReAct controller: ReactReasoner -> ReactActor loop, appending each output to ChatTask.messages."""
import json
import logging
from typing import List

from tensorrt_llm.scaffolding import Controller, MCPCallTask, Task
from tensorrt_llm.scaffolding.controller import ChatWithMCPController
from tensorrt_llm.scaffolding.task import ChatTask, UserMessage

from tensorrt_llm.scaffolding.contrib.ReAct.sub_agents.react_actor import (
    Action,
    ReactActorController,
    parse_actor_response,
)
from tensorrt_llm.scaffolding.contrib.ReAct.sub_agents.react_reasoner import (
    ReactReasonerController,
)
from tensorrt_llm.scaffolding.contrib.ReAct.tools import ToolMessage

logger = logging.getLogger(__name__)


class ReActController(Controller):
    """Controller that runs ReactReasoner -> ReactActor in a loop, appending each output to ChatTask.messages.

    Flow: build ChatTask with user question -> while True: ReactReasoner -> append reasoner message ->
    ReactActor -> append actor message. If ReactActor returns Answer: set task.output_str and break.
    If ReactActor returns Tool Call: yield MCP tasks, append tool results to messages, next round.
    """

    def __init__(
        self,
        reasoner_controller: ReactReasonerController,
        actor_controller: ReactActorController,
        max_iteration: int = 10,
    ):
        super().__init__()
        self.reasoner_controller = reasoner_controller
        self.actor_controller = actor_controller
        self.max_iteration = max_iteration

    def clone(self):
        return ReActController(
            reasoner_controller=self.reasoner_controller.clone(),
            actor_controller=self.actor_controller.clone(),
            max_iteration=self.max_iteration,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) >= 1, "ReActController expects at least one task"
        task = tasks[0]
        prompt = getattr(task, "input_str", None) or getattr(task, "user_prompt", None)
        assert prompt is not None, "Task must provide input_str or user_prompt"

        chat_task = ChatTask.create_from_messages([UserMessage(prompt)])
        iteration = 0

        while True:
            if iteration >= self.max_iteration:
                break
            iteration += 1

            # Stage 1: ReactReasoner
            yield from self.reasoner_controller.process(
                [task], chat_task=chat_task, **kwargs
            )
            reasoner_step = getattr(
                self.reasoner_controller,
                "last_chat_task",
                None,
            )
            assert reasoner_step is not None and reasoner_step.messages, (
                "ReactReasoner task must have messages"
            )
            reasoner_message = reasoner_step.messages[-1]
            thinker_content = getattr(reasoner_message, "content", None) or ""
            # Append reasoner output as UserMessage so the Actor request ends with a user turn.
            # This matches ParallelThinking and ensures the backend returns tool_calls (many APIs
            # only do so when the last message is from the user).
            chat_task.add_message(
                UserMessage("[Thinker output for this step]\n\n" + thinker_content)
            )

            # Stage 2: ReactActor
            yield from self.actor_controller.process(
                [task], chat_task=chat_task, **kwargs
            )
            actor_step = getattr(
                self.actor_controller,
                "last_chat_task",
                None,
            )
            assert actor_step is not None and actor_step.messages, (
                "ReactActor task must have messages"
            )
            actor_message = actor_step.messages[-1]
            # Ensure content is not None so messages_to_dict_content() includes this message
            # in the next round (it skips messages where content is None).
            if getattr(actor_message, "content", None) is None:
                actor_message.content = ""
            chat_task.add_message(actor_message)

            action, tool_names, tool_args_list, tool_call_ids = parse_actor_response(
                actor_message,
                registered_tool_names=getattr(
                    self.actor_controller,
                    "registered_tool_names",
                    None,
                ),
            )

            if action == Action.ANSWER:
                task.output_str = (
                    getattr(actor_message, "content", None) or ""
                ).strip()
                if not task.output_str:
                    task.output_str = thinker_content.strip()
                break

            # Build and yield MCP tasks
            assert tool_names is not None and tool_args_list is not None, (
                "Tool names and tool args list must be not None"
            )
            mcp_tasks = []
            for name, args in zip(tool_names, tool_args_list):
                args_str = json.dumps(args) if args else "{}"
                # Print tool and search/content before calling
                if isinstance(args, dict) and "query" in args:
                    logger.info(
                        "ReAct calling tool: name=%s, query=%s",
                        name,
                        args.get("query", ""),
                    )
                else:
                    logger.info(
                        "ReAct calling tool: name=%s, args=%s",
                        name,
                        args_str,
                    )
                mcp_tasks.append(
                    MCPCallTask.create_mcptask(
                        name,
                        args_str,
                        ChatWithMCPController.WorkerTag.TOOLCALL,
                    )
                )
            yield mcp_tasks

            # Append tool results to ChatTask.messages
            ids = tool_call_ids or [f"react-tool-{i}" for i in range(len(tool_names))]
            for i, (name, mcp_task) in enumerate(zip(tool_names, mcp_tasks)):
                tool_call_id = ids[i] if i < len(ids) else f"react-tool-{i}"
                result_str = getattr(mcp_task, "result_str", None) or ""
                chat_task.add_message(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call_id,
                        name=name,
                    )
                )
