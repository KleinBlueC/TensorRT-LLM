# HeavyThinking Web Researcher

This module implements a **deep web research agent** on TensorRT-LLM Scaffolding, using a **Think–Report–Action** paradigm and optional **integrative synthesis** over multiple parallel research runs.

---

## Design Overview

**HeavyThinking** is an open-source research agent with two usage modes:

| Mode | Description |
|------|-------------|
| **Iterative Web Researcher** | Single research run: Thinker → Reporter → Actor (with MCP tools) in a loop until the Actor chooses **answer**; then Extractor produces a concise final answer. |
| **HeavyThinking Web Researcher** | Multiple parallel runs: N independent IterativeResearcher instances run in parallel; an **IntegrativeSynthesizer** compares their reports and extracted answers and produces one final answer. |

### Think–Report–Action Paradigm

Each round of research follows three stages:

- **Think**: The **Thinker** sub-agent reasons over the current workspace (question, evolving report, latest tool results) and produces a cognitive scratchpad. This is used only within the round.
- **Report**: The **Reporter** sub-agent synthesizes the Thinker’s analysis and workspace into an **evolving report** (high-density summary), which becomes the main context for the next round.
- **Action**: The **Actor** sub-agent decides whether to call tools (search, scholar, visit, python_run) or to call the **answer** tool to end research. Tool results are written back into the workspace for the next round.

After the loop ends (either by **answer** or `max_iteration`), the **Extractor** sub-agent takes the final report and question and outputs a minimal answer (e.g. multiple-choice letter, fill-in value, or Yes/No).

### Workspace (Memory)

Long-horizon deep research suffers under **mono-contextual, linear accumulation**: the context window gets dominated by history (cognitive workspace suffocation) and early noise persists (irreversible noise contamination). This module uses an **IterResearch**-style design: deep research is a **Markov Decision Process** with periodic state reconstruction. Each round operates on a **focused Workspace**—the original question *q*, the **evolving report** from the previous round (central memory), and the most recent action and tool response—so the state stays compact and the report is iteratively synthesized instead of linearly accumulated.

**Implementation** (`memory/workspace.py`): The Workspace holds question, report, answer, and for the *current* round only the list of (action, tool_args, tool_responses); iteration count is kept for termination. `to_messages(system_prompt)` turns this state into `RoleMessage`s (user/assistant/tool) for the LLM. Thinker/Reporter/Actor/Extractor all share one Workspace instance per research run and update it via `set_question`, `set_report`, `set_tool_calling_result`, etc. Optional logging under `workspace_log_root/workspace_id` writes `reports.txt` and `tool_calls.txt` for debugging.

### Frontend–Backend Decoupling

As with [Open Deep Research](../open_deep_research/README.md), the control flow lives in **Controllers** (frontend) and execution in **Workers** (backend), so you can swap LLM backends or tool servers independently.

---

## Architecture

### Frontend (Control Flow)

Controllers implement the research pipeline:

| Controller | Description |
|------------|-------------|
| **IntegrativeSynthesizer** | Top-level controller for HeavyThinking mode: spawns N `IterativeResearcher` instances (each with its own Workspace), runs them in parallel via `ParallelProcess`, then runs one synthesizer generation over all (report, answer) pairs to produce the final answer. |
| **IterativeResearcher** | Single research run: loop Thinker → Reporter → Actor; on tool calls yields `MCPCallTask`s; on **answer** breaks and runs Extractor; writes `task.output_str` from the extracted answer. |
| **ThinkerController** | Builds a ChatTask from workspace state (question, report, last tool results); no tools; output is reasoning for the round. |
| **ReporterController** | Takes Thinker output + workspace; produces updated report; writes result to `workspace.set_report()`. |
| **ActorController** | Takes Thinker output + workspace; has tools (search, scholar, visit, python_run, answer); decides tool call(s) or answer; parsed by `parse_actor_response()`. |
| **ExtractorController** | After loop: question + report → one generation → minimal answer (e.g. "A", "42", "Yes"); used to set `task.output_str`. |

### Backend (Workers)

| Worker | Description |
|--------|-------------|
| **TRTOpenaiWorker** | Sends LLM generation requests to the TensorRT-LLM OpenAI-compatible endpoint. |
| **MCPWorker** | Executes tool calls via the HeavyThinking MCP server (SSE); tools: search, scholar, visit, python_run. |


---

## Quick Start

### 1. Start TensorRT-LLM server

Create `.extra-llm-api-config.yml`:

```yaml
reorder_policy_config:
  policy_name: "AgentTree"
  policy_args:
    agent_percentage: 0.5
    agent_types: ["agent_deep_research"]
    agent_inflight_seq_num: 8
```

```bash
trtllm-serve serve /path/to/llm_models/Qwen3/Qwen3-30B-A3B \
  --max_num_tokens 32768 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options .extra-llm-api-config.yml \
  --reasoning_parser qwen3 \
  --tool_parser qwen3
```

### 2. Configure environment

Copy and edit `.env` from `mcp/.env.example` (e.g. `TAVILY_API_KEY`, `E2B_API_KEY`). See [mcp/README.md](mcp/README.md).

### 3. Start the MCP server

```bash
cd tensorrt_llm/scaffolding/contrib/HeavyThinking/mcp
uv run launch_mcp_server.py
```

### 4. Run the example

**Iterative** (single research run):

```bash
cd examples/scaffolding/contrib/HeavyThinking
python run_iterative_web_researcher.py
```

**HeavyThinking** (N parallel runs + synthesizer):

```bash
cd examples/scaffolding/contrib/HeavyThinking
python run_heavy_thinking_web_researcher.py
```

---

## Acknowledgments

This module is a **reproduction and adaptation** of the Tongyi DeepResearch team’s **WebResearcher** work on long-horizon deep-research agents, including the IterResearch paradigm (MDP with periodic state reconstruction) and the focused Workspace with evolving report as central memory. See:

- **WebResearcher: Unleashing unbounded reasoning capability in Long-Horizon Agents**  
  Zile Qiao, Guoxin Chen, Xuanzhong Chen, et al.  
  [arXiv:2509.13309](https://arxiv.org/abs/2509.13309) (2025).  
  Blog: [Introducing Tongyi Deep Research](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/).

The implementation is built on TensorRT-LLM [Scaffolding](../open_deep_research/README.md) and follows the [Open Deep Research](https://github.com/langchain-ai/open_deep_research) style architecture for frontend–backend decoupling.
