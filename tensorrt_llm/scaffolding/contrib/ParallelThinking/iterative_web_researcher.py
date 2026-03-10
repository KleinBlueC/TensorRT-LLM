from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import Worker

from tensorrt_llm.scaffolding.contrib.ParallelThinking.core_agents.researcher import (
    IterativeResearcher,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.memory.workspace import Workspace
from tensorrt_llm.scaffolding.contrib.ParallelThinking.sub_agents.actor import (
    ActorController,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.sub_agents.extractor import (
    ExtractorController,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.sub_agents.reporter import (
    ReporterController,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.sub_agents.thinker import (
    ThinkerController,
)


def create_iterative_web_researcher_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 16384,
    max_parallel_requests: int = 1024,
    temperature: float = 0.2,
) -> ScaffoldingLlm:
    """Build a ScaffoldingLlm whose controller is IterResearcher and workers are generation + MCP.

    Mirrors create_open_deep_research_scaffolding_llm in open_deep_research/supervisor.py:
    returns a ScaffoldingLlm usable for iterative web research (Thinker -> Reporter/Actor loop
    with MCP tools: search, scholar, visit, python_run).
    """
    sampling_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    thinker_controller = ThinkerController(sampling_params=sampling_params)
    reporter_controller = ReporterController(sampling_params=sampling_params)
    actor_controller = ActorController(sampling_params=sampling_params)
    extractor_controller = ExtractorController(sampling_params=sampling_params)

    workspace_instance = Workspace()

    iter_researcher_controller = IterativeResearcher(
        thinker_controller=thinker_controller,
        reporter_controller=reporter_controller,
        actor_controller=actor_controller,
        extractor_controller=extractor_controller,
        workspace_instance=workspace_instance,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }

    return ScaffoldingLlm(
        iter_researcher_controller,
        workers,
        max_parallel_requests=max_parallel_requests,
    )


class IterativeWebResearcher(ScaffoldingLlm):
    """ScaffoldingLlm for iterative web research: IterResearcher controller + generation and MCP workers.

    Use this class when you want a single scaffolding_llm instance configured with
    core_agents.IterResearcher and workers GENERATION and TOOLCALL (same as
    create_iterative_web_researcher_scaffolding_llm).
    """

    def __init__(
        self,
        generation_worker: Worker,
        mcp_worker: Worker,
        max_tokens: int = 16384,
        max_parallel_requests: int = 1024,
        temperature: float = 0.2,
    ):
        sampling_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        thinker_controller = ThinkerController(sampling_params=sampling_params)
        reporter_controller = ReporterController(sampling_params=sampling_params)
        actor_controller = ActorController(sampling_params=sampling_params)
        extractor_controller = ExtractorController(sampling_params=sampling_params)

        workspace_instance = Workspace()

        iter_researcher_controller = IterativeResearcher(
            thinker_controller=thinker_controller,
            reporter_controller=reporter_controller,
            actor_controller=actor_controller,
            extractor_controller=extractor_controller,
            workspace_instance=workspace_instance,
        )

        workers = {
            NativeGenerationController.WorkerTag.GENERATION: generation_worker,
            ActorController.WorkerTag.TOOLCALL: mcp_worker,
        }

        super().__init__(
            iter_researcher_controller,
            workers,
            max_parallel_requests=max_parallel_requests,
        )
