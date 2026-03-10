"""IntegrativeSynthesizer: runs N IterativeResearcher in parallel, then synthesizes their reports and answers into one final answer."""

import copy
import os
from datetime import datetime
from typing import List, Optional

from tensorrt_llm.scaffolding import Controller, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController, ParallelProcess
from tensorrt_llm.scaffolding.task import ChatTask, GenerationTask, SystemMessage, UserMessage

from tensorrt_llm.scaffolding.contrib.ParallelThinking.core_agents.researcher import (
    IterativeResearcher,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.memory.workspace import Workspace
from tensorrt_llm.scaffolding.contrib.ParallelThinking.prompts import (
    synthesizer_system_prompt,
)


class IntegrativeSynthesizer(Controller):
    """Controller that runs N IterativeResearcher instances in parallel, then synthesizes their final reports and answers into a single final answer.

    Flow:
    1. Create N researcher controllers (each with its own Workspace) and N tasks with the same user prompt.
    2. Yield ParallelProcess so the framework runs all N researchers to completion.
    3. Collect each researcher's workspace.get_report() and workspace.get_answer() (or task.output_str).
    4. Build a synthesizer ChatTask: system prompt + user message containing the question and the N (report, answer) pairs.
    5. Run the synthesizer controller once; set the original task.output_str from the synthesizer's response.
    """

    def __init__(
        self,
        researcher_controller_prototype: IterativeResearcher,
        synthesizer_controller: NativeGenerationController,
        max_parallel_search: int = 3,
        workspace_log_root: Optional[str] = None,
    ):
        super().__init__()
        self.researcher_controller_prototype = researcher_controller_prototype
        self.synthesizer_controller = synthesizer_controller
        self.max_parallel_search = max_parallel_search
        if workspace_log_root is not None:
            self._workspace_log_root = os.path.abspath(workspace_log_root)
        else:
            self._workspace_log_root = os.path.abspath(
                os.path.join(
                    ".",
                    "log",
                    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                )
            )
        os.makedirs(self._workspace_log_root, exist_ok=True)

    def clone(self):
        return IntegrativeSynthesizer(
            researcher_controller_prototype=self.researcher_controller_prototype.clone(),
            synthesizer_controller=self.synthesizer_controller.clone(),
            max_parallel_search=self.max_parallel_search,
            workspace_log_root=self._workspace_log_root,
        )

    def _spawn_researcher_with_workspace(self, index: int) -> IterativeResearcher:
        """Return a clone of the prototype with a fresh Workspace (workspace_id = index, shared log root)."""
        workspace = Workspace(
            workspace_id=str(index),
            workspace_log_root=self._workspace_log_root,
        )
        researcher = self.researcher_controller_prototype.clone()
        researcher.workspace = workspace
        researcher.thinker_controller.workspace = workspace
        researcher.reporter_controller.workspace = workspace
        researcher.actor_controller.workspace = workspace
        researcher.extractor_controller.workspace = workspace
        return researcher

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) >= 1, "IntegrativeSynthesizer expects at least one task"
        task = tasks[0]
        prompt = getattr(task, "input_str", None) or getattr(task, "user_prompt", None)
        assert prompt is not None, "Task must provide input_str or user_prompt"

        n = self.max_parallel_search
        researchers = [self._spawn_researcher_with_workspace(i) for i in range(n)]
        researcher_tasks = [GenerationTask.create_from_prompt(prompt) for _ in range(n)]
        kwargs_list = [copy.deepcopy(kwargs) for _ in range(n)]

        yield ParallelProcess(
            researchers,
            [[t] for t in researcher_tasks],
            kwargs_list,
        )

        reports = [
            r.workspace.get_report() or ""
            for r in researchers
        ]
        answers = [
            (r.workspace.get_answer() or (researcher_tasks[i].output_str or "") or "").strip()
            for i, r in enumerate(researchers)
        ]

        now = datetime.now()
        date_str = f"{now:%a} {now:%b} {now.day}, {now:%Y}"
        system_content = synthesizer_system_prompt.format(date=date_str)

        parts = [
            "[Original question]\n\n",
            prompt,
            "\n\n",
        ]
        for i in range(n):
            parts.append(f"--- Research run {i + 1} ---\n")
            parts.append("Report:\n")
            parts.append(reports[i] or "(empty)")
            parts.append("\n\nExtracted answer: ")
            parts.append(answers[i] or "(none)")
            parts.append("\n\n")

        user_content = "".join(parts)
        messages = [
            SystemMessage(system_content),
            UserMessage(user_content.strip()),
        ]
        synth_task = ChatTask.create_from_messages(messages)

        yield from self.synthesizer_controller.process([synth_task], **kwargs)

        if synth_task.messages:
            last_content = getattr(synth_task.messages[-1], "content", None) or ""
            task.output_str = last_content.strip()
        else:
            task.output_str = getattr(synth_task, "output_str", None) or ""
