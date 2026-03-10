import copy
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage

from tensorrt_llm.scaffolding.contrib.ParallelThinking.prompts import (
    thinker_system_prompt,
)

if TYPE_CHECKING:
    from tensorrt_llm.scaffolding.contrib.ParallelThinking.memory.workspace import (
        Workspace,
    )


class ThinkerController(NativeGenerationController):
    """Sub-controller for the Thinker stage. Builds ChatTask from workspace and yields it."""

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

        now = datetime.now()
        date_str = f"{now:%a} {now:%b} {now.day}, {now:%Y}"

        system_content = thinker_system_prompt.format(date=date_str)
        workspace_messages = self.workspace.to_messages(system_prompt=None)
        messages = [SystemMessage(system_content)] + workspace_messages

        # do not pass tools to the chat task
        chat_task = ChatTask.create_from_messages(
            messages
        )
        self.last_chat_task = chat_task
        yield from super().process([chat_task], **kwargs)
