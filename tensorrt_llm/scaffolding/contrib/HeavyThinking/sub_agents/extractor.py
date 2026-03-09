import copy
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage, UserMessage

from tensorrt_llm.scaffolding.contrib.HeavyThinking.prompts import (
    extractor_system_prompt,
)

if TYPE_CHECKING:
    from tensorrt_llm.scaffolding.contrib.HeavyThinking.memory.workspace import (
        Workspace,
    )


class ExtractorController(NativeGenerationController):
    """Sub-controller for the Extractor stage.

    After the Thinker–Reporter–Actor loop ends, the Extractor takes the
    workspace's question and report, runs one generation step to extract
    a pure answer (e.g., multiple-choice letter or fill-in value), and
    exposes it via last_chat_task for the caller to write to task.output_str.
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
        if self.workspace is None:
            yield from super().process(tasks, **kwargs)
            return

        question = self.workspace.get_question() or ""
        report = self.workspace.get_report() or ""

        now = datetime.now()
        date_str = f"{now:%a} {now:%b} {now.day}, {now:%Y}"
        system_content = extractor_system_prompt.format(date=date_str)
        user_content = (
            "[Question]\n\n"
            + question
            + "\n\n[Report]\n\n"
            + report
        )
        messages = [
            SystemMessage(system_content),
            UserMessage(user_content),
        ]
        chat_task = ChatTask.create_from_messages(messages)
        self.last_chat_task = chat_task

        yield from super().process([chat_task], **kwargs)
