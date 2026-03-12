import copy
from datetime import datetime
from typing import List, Optional

from tensorrt_llm.scaffolding import ChatTask, Task
from tensorrt_llm.scaffolding.controller import NativeGenerationController
from tensorrt_llm.scaffolding.task import SystemMessage

from tensorrt_llm.scaffolding.contrib.ReAct.prompts import thinker_system_prompt


class ReactReasonerController(NativeGenerationController):
    """ReAct reasoner stage: builds ChatTask from current messages (no workspace), yields generation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        system_content = thinker_system_prompt.format(date=date_str)
        messages = [SystemMessage(system_content)] + list(chat_task.messages)
        step_task = ChatTask.create_from_messages(messages)
        self.last_chat_task = step_task
        yield from super().process([step_task], **kwargs)
