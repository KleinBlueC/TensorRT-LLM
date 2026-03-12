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

## Datasets

JSON files under `evaluation/datasets/`:

- `swebench_benchmark.json`
- `mcpbench_benchmark.json`
- `tau2bench_benchmark.json`
- `terminalbench_benchmark.json`
- `mathhay_benchmark.json`
- `frames_benchmark.json`
- `browsecomp_benchmark.json`, `mind2web_benchmark.json`, `webvoyager_benchmark.json`
- etc.
