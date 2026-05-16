# General Evaluation Gateway

Unified interface for running **ScaffoldingLlm** on multi-task evaluation. Supports multiple **benchmarks** and **datasets** (JSON under `evaluation/datasets/`). Agent is **ScaffoldingLlm** only; calls use its public API (`generate_async` / `aresult`).

## Usage

```python
from pathlib import Path
from tensorrt_llm.scaffolding import ScaffoldingLlm
from evaluation.gateway import GeneralEvalGateway, load_tasks_from_json

# 1. Build ScaffoldingLlm (controller + workers)
scaffolding_llm = ScaffoldingLlm(prototype_controller=..., workers={...})

# 2. Create gateway and run on a dataset
gateway = GeneralEvalGateway(scaffolding_llm=scaffolding_llm)
dataset_path = Path("evaluation/datasets/swebench_benchmark.json")

# Async
results = await gateway.run(dataset_path=dataset_path, max_tasks=5)

# Or sync
results = gateway.run_sync(dataset_path=dataset_path, max_tasks=5)

# 3. Use results (task_id, benchmark, domain, prompt, output_text, error)
for r in results:
    print(r.task_id, r.benchmark, r.output_text or r.error)
```

## Data format

- **Input**: JSON file = list of `{"benchmark", "domain", "task" [, "dataset"]}` (same as General-AgentBench task-file format).
- **Output**: List of `TaskRunResult` (task_id, benchmark, domain, prompt, output_text, error).

## Preprocessing

Per-dataset preprocessing is currently **placeholder**: each benchmark has a function that extracts a prompt from the task payload (e.g. `instruction`, `task_description`). Replace or extend `gateway/preprocessors.py` and pass a custom `preprocessors` dict to `GeneralEvalGateway(agent, preprocessors=...)` for full preprocessing.

## MCP (mcpbench, SWE-Bench)

For **mcpbench** (and other MCP-based benchmarks), use **ScaffoldingLlm** with a controller that includes the **StdioMCPWorker** from `evaluation/servers`. Set `MCP_SERVERS_ROOT` to the mcp-bench `mcp_servers` directory (e.g. General-AgentBench `benchmarks/mcp-bench/mcp_servers`). Build stdio configs with `get_stdio_configs_for_task(entry)` or `get_stdio_configs_for_worker(server_names)`, then `StdioMCPWorker.init_with_stdio_configs(configs)` and register the worker in the controller. See `run_eval_with_mcp_servers.py` for a minimal example.

### SWE-Bench

SWE-Bench is replicated from General-AgentBench: Docker-based repo fix tasks. Tool logic lives in **`mcp_tools/swebench/`**; the server in `servers/swebench_server.py` runs over **HTTP/SSE** (same transport as BrowseComp).

- **Tools (agent-visible)**: `swebench_execute_bash`, `swebench_str_replace_editor`, `swebench_finish`.
- **Internal tools** (orchestrator only): `__swebench_switch_container`, `__swebench_run_tests`, `__swebench_get_patch`, `__swebench_cleanup`.

Dataset path: env **SWEBENCH_DATASET_PATH** or `evaluation/datasets/swebench-verified` (each task is a subdir with `docker-compose.yaml`, `task.json`, etc.).

- **Launch server**: `python -m servers.launch_mcp_server --dataset swebench --port 8083` or `python -m servers.swebench_server --port 8083`.
- **Full test flow with logging**: `python run_swebench.py --mcp_url http://localhost:8083/sse --dataset datasets/swebench_benchmark.json --log_dir log/swebench`.

## Datasets

JSON files under `evaluation/datasets/`:

- `swebench_benchmark.json`
- `mcpbench_benchmark.json`
- `tau2bench_benchmark.json`
- `terminalbench_benchmark.json`
- `mathhay_benchmark.json`
- `frames_benchmark.json`
- `browsecomp_benchmark.json`
- `mind2web_benchmark.json`
- `webvoyager_benchmark.json`
- etc.
