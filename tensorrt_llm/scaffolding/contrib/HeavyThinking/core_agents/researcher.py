import json
from dataclasses import dataclass, field
from typing import List

from tensorrt_llm.scaffolding import Controller, MCPCallTask, Task
from tensorrt_llm.scaffolding.controller import ChatWithMCPController

from tensorrt_llm.scaffolding.contrib.HeavyThinking.memory.workspace import Workspace
from tensorrt_llm.scaffolding.contrib.HeavyThinking.mcp.tool_registration import (
    REGISTERED_MCP_TOOLS,
)
from tensorrt_llm.scaffolding.contrib.HeavyThinking.sub_agents.actor import (
    Action,
    ActorController,
    parse_actor_response,
)
from tensorrt_llm.scaffolding.contrib.HeavyThinking.sub_agents.extractor import (
    ExtractorController,
)
from tensorrt_llm.scaffolding.contrib.HeavyThinking.sub_agents.reporter import (
    ReporterController,
)
from tensorrt_llm.scaffolding.contrib.HeavyThinking.sub_agents.thinker import (
    ThinkerController,
)



class IterativeResearcher(Controller):
    """Controller that runs Thinker -> Reporter -> Actor in a loop.

    Flow: set workspace question -> while True: Thinker -> Reporter -> Actor.
    - If Actor returns Answer: break; then run Extractor on workspace question+report
      to get a pure answer and set task.output_str.
    - If Actor returns Tool Call: yield MCP task, update workspace, increment
      iteration, then next round (Thinker -> Reporter -> Actor again).
    - After loop (whether by Answer or max_iteration), Extractor runs and
      task.output_str is set to the extracted answer.
    """

    def __init__(
        self,
        thinker_controller: ThinkerController,
        reporter_controller: ReporterController,
        actor_controller: ActorController,
        extractor_controller: ExtractorController,
        workspace_instance: Workspace,
        max_iteration: int = 10,
    ):
        super().__init__()
        self.workspace = workspace_instance
        self.thinker_controller = thinker_controller
        self.reporter_controller = reporter_controller
        self.actor_controller = actor_controller
        self.extractor_controller = extractor_controller
        self.max_iteration = max_iteration
        # Shared workspace: all sub-controllers read/update the same instance.
        self.thinker_controller.workspace = workspace_instance
        self.reporter_controller.workspace = workspace_instance
        self.actor_controller.workspace = workspace_instance
        self.extractor_controller.workspace = workspace_instance

    def clone(self):
        return IterativeResearcher(
            thinker_controller=self.thinker_controller.clone(),
            reporter_controller=self.reporter_controller.clone(),
            actor_controller=self.actor_controller.clone(),
            extractor_controller=self.extractor_controller.clone(),
            workspace_instance=self.workspace,
            max_iteration=self.max_iteration,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) >= 1, "IterResearcher expects at least one task"
        task = tasks[0]
        prompt = getattr(task, "input_str", None) or getattr(task, "user_prompt", None)
        assert prompt is not None, "Task must provide input_str or user_prompt"

        self.workspace.set_question(prompt)

        while True:
            if self.workspace.get_iteration() >= self.max_iteration:
                break
            # Stage 1: Thinker
            yield from self.thinker_controller.process([task], **kwargs)
            thinker_task = getattr(
                self.thinker_controller,
                "last_chat_task",
                None,
            )
            assert thinker_task is not None and thinker_task.messages, (
                "Thinker task must have messages"
            )
            thinker_output = thinker_task.messages[-1].content

            # Stage 2: Reporter (updates workspace.report)
            yield from self.reporter_controller.process(
                [task], thinker_output=thinker_output, **kwargs
            )

            # Stage 3: Actor (decide Answer or Tool Call)
            yield from self.actor_controller.process(
                [task], thinker_output=thinker_output, **kwargs
            )
            actor_task = getattr(
                self.actor_controller,
                "last_chat_task",
                None,
            )
            assert actor_task is not None and actor_task.messages, (
                "Actor task must have messages"
            )

            action, tool_names, tool_args_list = parse_actor_response(
                actor_task.messages[-1]
            )

            if action == Action.ANSWER:
                break

            # Build MCP tasks (one per tool call). Worker expects task.args as JSON str.
            # Yielding a list runs all tasks concurrently in the framework.
            assert tool_names is not None and tool_args_list is not None, "Tool names and tool args list must be not None"
            mcp_tasks = [
                MCPCallTask.create_mcptask(
                    name,
                    json.dumps(args) if args else "{}",
                    ChatWithMCPController.WorkerTag.TOOLCALL,
                )
                for name, args in zip(tool_names, tool_args_list)
            ]
            yield mcp_tasks

            # only reset the current iteration's actions, tool_args, and tool_responses
            self.workspace.set_tool_calling_result(tool_names, tool_args_list, [mcp_task.result_str for mcp_task in mcp_tasks])
            self.workspace.increment_iteration()
            print("[{}] Iteration incremented to {}".format(self.workspace.get_workspace_id(), self.workspace.get_iteration()))

        # After loop: run Extractor to get pure answer from question + report, write to task
        yield from self.extractor_controller.process([task], **kwargs)
        extractor_task = getattr(
            self.extractor_controller,
            "last_chat_task",
            None,
        )
        assert extractor_task is not None and extractor_task.messages, "Extractor task must have messages"
        task.output_str = (
            getattr(extractor_task.messages[-1], "content", None) or ""
        ).strip()

        print("[{}] Extractor output: {}".format(self.workspace.get_workspace_id(), task.output_str))
        self.workspace.set_answer(task.output_str)
