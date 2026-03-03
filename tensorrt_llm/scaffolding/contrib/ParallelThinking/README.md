# ParallelThinking

Controller that runs multiple **SearchAgent** (Controller) instances in parallel on the same prompt, then feeds all results to a **SynthesisAgent** (Controller) to produce a single summarized output.

## Process flow

1. **Parallel search**: `num_parallel` clones of the SearchAgent run with the same prompt (identical task).
2. **Gather**: All SearchAgent results are collected into a dict (keyed by index).
3. **Synthesis**: The dict is formatted and passed to the SynthesisAgent, which summarizes and returns the final output.

## Usage

```python
from tensorrt_llm.scaffolding import Controller
from tensorrt_llm.scaffolding.contrib.ParallelThinking import ParallelThinkingController

# search_agent and synthesis_agent are Controller subclasses
controller = ParallelThinkingController(
    search_agent=my_search_agent,
    synthesis_agent=my_synthesis_agent,
    num_parallel=3,
)
# Use with ScaffoldingLlm as the prototype_controller
```

SearchAgent and SynthesisAgent must be Controller subclasses that accept tasks with `input_str` (prompt). The synthesis agent receives a single text prompt containing all search results (by default, formatted as `[Result 1]`, `[Result 2]`, ...). You can pass a custom `results_formatter(results_dict: Dict[int, str]) -> str` to control this format.
