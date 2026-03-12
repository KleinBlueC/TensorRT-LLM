"""ParallelThinking web researcher: IntegrativeSynthesizer controller (N parallel IterativeResearcher + synthesizer)."""
from typing import Optional

from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    NativeGenerationController,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import Worker

from tensorrt_llm.scaffolding.contrib.ParallelThinking.core_agents.researcher import (
    IterativeResearcher,
)
from tensorrt_llm.scaffolding.contrib.ParallelThinking.core_agents.synthesizer import (
    IntegrativeSynthesizer,
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


def create_parallel_thinking_web_researcher_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 16384,
    max_parallel_requests: int = 1024,
    temperature: float = 0.6,
    top_p: Optional[float] = 0.95,
    max_parallel_search: int = 3,
    workspace_log_root: Optional[str] = None,
) -> ScaffoldingLlm:
    """Build a ScaffoldingLlm whose controller is IntegrativeSynthesizer.

    Runs N IterativeResearcher instances in parallel (Thinker -> Reporter -> Actor -> Extractor),
    then synthesizes their reports and answers into one final answer. Workers are generation + MCP.
    """
    sampling_params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        sampling_params["top_p"] = top_p

    thinker_controller = ThinkerController(sampling_params=sampling_params)
    reporter_controller = ReporterController(sampling_params=sampling_params)
    actor_controller = ActorController(sampling_params=sampling_params)
    extractor_controller = ExtractorController(sampling_params=sampling_params)

    # Dummy workspace for the prototype; each spawned researcher gets its own Workspace.
    workspace_instance = Workspace()

    iter_researcher_controller = IterativeResearcher(
        thinker_controller=thinker_controller,
        reporter_controller=reporter_controller,
        actor_controller=actor_controller,
        extractor_controller=extractor_controller,
        workspace_instance=workspace_instance,
    )

    synthesizer_controller = NativeGenerationController(sampling_params=sampling_params)

    integrative_synthesizer_controller = IntegrativeSynthesizer(
        researcher_controller_prototype=iter_researcher_controller,
        synthesizer_controller=synthesizer_controller,
        max_parallel_search=max_parallel_search,
        workspace_log_root=workspace_log_root,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
    }

    return ScaffoldingLlm(
        integrative_synthesizer_controller,
        workers,
        max_parallel_requests=max_parallel_requests,
    )


class ParallelThinkingWebResearcher(ScaffoldingLlm):
    """ScaffoldingLlm for ParallelThinking web research: IntegrativeSynthesizer controller + generation and MCP workers.

    Runs N IterativeResearcher in parallel, then synthesizes their outputs into a single final answer.
    """

    def __init__(
        self,
        generation_worker: Worker,
        mcp_worker: Worker,
        max_tokens: int = 16384,
        max_parallel_requests: int = 1024,
        temperature: float = 0.6,
        top_p: Optional[float] = 0.95,
        max_parallel_search: int = 3,
        workspace_log_root: Optional[str] = None,
    ):
        sampling_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            sampling_params["top_p"] = top_p
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

        synthesizer_controller = NativeGenerationController(sampling_params=sampling_params)

        integrative_synthesizer_controller = IntegrativeSynthesizer(
            researcher_controller_prototype=iter_researcher_controller,
            synthesizer_controller=synthesizer_controller,
            max_parallel_search=max_parallel_search,
            workspace_log_root=workspace_log_root,
        )

        workers = {
            NativeGenerationController.WorkerTag.GENERATION: generation_worker,
            ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
        }

        super().__init__(
            integrative_synthesizer_controller,
            workers,
            max_parallel_requests=max_parallel_requests,
        )
