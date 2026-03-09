import copy
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage, UserMessage

from tensorrt_llm.scaffolding.contrib.HeavyThinking.prompts import (
    reporter_system_prompt,
)

if TYPE_CHECKING:
    from tensorrt_llm.scaffolding.contrib.HeavyThinking.memory.workspace import (
        Workspace,
    )


class ReporterController(NativeGenerationController):
    """Sub-controller for the Reporter stage.

    Given thinker_output (LLM thinking for this round) and the current workspace,
    builds a ChatTask from workspace.to_messages() plus the thinker output,
    runs generation to produce a new report, then overwrites workspace.report
    with the result.
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
        system_content = reporter_system_prompt.format(date=date_str)
        workspace_messages = self.workspace.to_messages(system_prompt=None)
        thinker_user_message = UserMessage(
            "[Thinker's analysis for this round]\n\n" + (thinker_output or "")
        )
        messages = (
            [SystemMessage(system_content)]
            + workspace_messages
            + [thinker_user_message]
        )
        chat_task = ChatTask.create_from_messages(messages)
        self.last_chat_task = chat_task

        yield from super().process([chat_task], **kwargs)

        report_content = ""
        if chat_task.messages:
            last = chat_task.messages[-1]
            report_content = getattr(last, "content", None) or ""
        if not report_content and getattr(chat_task, "output_str", None):
            report_content = chat_task.output_str or ""
        self.workspace.set_report(report_content)
