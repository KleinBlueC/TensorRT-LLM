import copy
from typing import Any, Callable, Dict, List, Optional

from tensorrt_llm.scaffolding.controller import Controller, ParallelProcess
from tensorrt_llm.scaffolding.task import GenerationTask, Task

DEFAULT_SEARCH_SYSTEM_PROMPT = (
    "Answer the question below directly and completely.\n"
    "Rules:\n"
    "- Do NOT repeat the question.\n"
    "- Do NOT add commentary, meta-discussion, or filler text.\n"
    "- Use clear structure (numbered lists, headings, etc.) when appropriate.\n"
    "- Finish the entire answer; do not stop mid-sentence.\n"
    "\n"
    "Q: ")

DEFAULT_SYNTHESIS_SYSTEM_PROMPT = (
    "You are given several draft answers to the same question.\n"
    "Combine the best parts into a single, complete answer.\n"
    "Rules (in priority order):\n"
    "- **MOST IMPORTANT**: You MUST synthesize, NOT simply repeat or copy any single draft. "
    "Merge, restructure, and improve upon the drafts to produce a genuinely unified answer.\n"
    "- Output ONLY the final answer, no commentary or meta-discussion.\n"
    "- Do NOT say things like 'based on the drafts' or 'combining the above'.\n"
    "- Keep the structure clear (headings, lists, paragraphs as needed).\n"
    "- Finish the entire answer; do not stop mid-sentence.\n")


def _format_search_results_for_synthesis(prompt: str,
                                         results_dict: Dict[int,
                                                            str]) -> str:
    lines = [f"Question: {prompt}", ""]
    for idx, text in sorted(results_dict.items()):
        clean = (str(text) if text else '').strip()
        if clean:
            lines.append(f"Draft {idx + 1}:")
            lines.append(clean)
            lines.append("")
    lines.append("Final answer:")
    return "\n".join(lines)


class ParallelThinkingController(Controller):

    def __init__(
        self,
        search_agent: Controller,
        synthesis_agent: Controller,
        num_parallel: int = 3,
        search_system_prompt: Optional[str] = DEFAULT_SEARCH_SYSTEM_PROMPT,
        synthesis_system_prompt: Optional[str] = DEFAULT_SYNTHESIS_SYSTEM_PROMPT,
        results_formatter: Optional[Callable[[str, Dict[int, str]],
                                             str]] = None,
    ):
        super().__init__()
        self.search_agent = search_agent
        self.synthesis_agent = synthesis_agent
        self.num_parallel = num_parallel
        self.search_system_prompt = search_system_prompt
        self.synthesis_system_prompt = synthesis_system_prompt
        self.results_formatter = results_formatter or _format_search_results_for_synthesis

    def clone(self):
        search_agent = self.search_agent.clone()
        synthesis_agent = self.synthesis_agent.clone()
        return ParallelThinkingController(
            search_agent,
            synthesis_agent,
            num_parallel=self.num_parallel,
            search_system_prompt=self.search_system_prompt,
            synthesis_system_prompt=self.synthesis_system_prompt,
            results_formatter=self.results_formatter,
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) >= 1, "ParallelThinkingController expects at least one task"
        task = tasks[0]
        prompt = getattr(task, "input_str", None)
        if prompt is None:
            raise ValueError(
                "Task must have input_str (prompt) for ParallelThinkingController"
            )

        num_parallel = self.num_parallel
        search_controllers = [
            self.search_agent.clone() for _ in range(num_parallel)
        ]

        if self.search_system_prompt:
            search_prompt = f"{self.search_system_prompt}{prompt}\n\nA:\n"
        else:
            search_prompt = prompt

        search_tasks_list = [[GenerationTask.create_from_prompt(search_prompt)]
                             for _ in range(num_parallel)]
        kwargs_list = [copy.deepcopy(kwargs) for _ in range(num_parallel)]

        yield ParallelProcess(search_controllers, search_tasks_list,
                              kwargs_list)

        results_dict: Dict[int, str] = {}
        for i in range(num_parallel):
            t = search_tasks_list[i][0]
            result_text = getattr(t, "output_str", None) or ""
            results_dict[i] = result_text
            print(f"\n[SearchAgent {i + 1}/{num_parallel} Result]")
            print(result_text)

        print("\n----------\n\n")

        synthesis_input = self.results_formatter(prompt, results_dict)
        if self.synthesis_system_prompt:
            synthesis_input = f"{self.synthesis_system_prompt}\n\n{synthesis_input}"
        synthesis_task = GenerationTask.create_from_prompt(synthesis_input)

        yield from self.synthesis_agent.process([synthesis_task], **kwargs)

        print("[Synthesis Result]")
        print(synthesis_task.output_str)

        task.output_str = synthesis_task.output_str
        task.output_tokens = getattr(synthesis_task, "output_tokens", None)
        if hasattr(synthesis_task, "result"):
            task.result = synthesis_task.result
