# Deep-Search Agent Benchmark Datasets

This directory contains **six** unified JSONL benchmark files for evaluating agent deep-search (search + reasoning) capabilities. Each file follows the same schema so you can load any subset and run horizontal evaluation across datasets.

## The Six Dataset Files

| File | Dataset | Description | Approx. size |
|------|---------|-------------|--------------|
| **hle.jsonl** | Humanity's Last Exam | Multi-discipline, text-only subset | 2,154 |
| **gaia.jsonl** | GAIA (General AI Assistants) | Text-only validation | 103 |
| **browsecomp_en.jsonl** | BrowseComp (English) | Hard-to-find web information | 1,266 |
| **browsecomp_zh.jsonl** | BrowseComp-ZH (Chinese) | Chinese web, multi-hop | 289 |
| **xbench_deepsearch.jsonl** | Xbench-DeepSearch | Planning, search, reasoning (Chinese) | 100 |
| **frames.jsonl** | FRAMES | RAG, factuality, multi-hop over Wikipedia | 824 |

**Total:** ~4,736 problems. Preparation scripts (download, normalize from raw sources) live in the sibling folder **`../dataset_preparation/`**.

---

## Unified Schema (JSONL)

Every line is one JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique id (e.g. `hle_0`, `frames_42`) |
| `prompt` | string | **Question text** — use as agent input |
| `reference_answer` | string | Gold answer for evaluation |
| `dataset` | string | One of: `hle`, `gaia`, `browsecomp_en`, `browsecomp_zh`, `xbench_deepsearch`, `frames` |
| `metadata` | object | Optional (e.g. `subject`, `level`, `category`) |

Example:

```json
{"id": "gaia_0", "prompt": "A paper about AI regulation...", "reference_answer": "egalitarian", "dataset": "gaia", "metadata": {"level": "2"}}
```

---

## How to Use the Six Datasets for Testing

### 1. Load prompts and run your agent

Load prompts from one or more JSONL files, send each `prompt` to your agent, and collect the model’s answer for each `id`.

**Option A — Use the preparation loader** (from repo root or with `PYTHONPATH` set):

```python
from examples.scaffolding.benchmarks.dataset_preparation import load_utils
from pathlib import Path

datasets_dir = Path("examples/scaffolding/benchmarks/datasets")

# Load prompts only (e.g. for a single dataset)
prompts = load_utils.load_prompts_from_deepsearch_jsonl(
    datasets_dir / "frames.jsonl", max_num=50
)

# Load full records (id, prompt, reference_answer, dataset, metadata) for evaluation
records = load_utils.load_deepsearch_records(
    datasets_dir / "hle.jsonl", max_num=100
)
for r in records:
    prompt_text = r["prompt"]
    problem_id = r["id"]
    # ... run agent on prompt_text, then store answer with problem_id
```

**Option B — Read JSONL manually:**

```python
import json
from pathlib import Path

path = Path("examples/scaffolding/benchmarks/datasets/frames.jsonl")
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        problem_id = obj["id"]
        prompt = obj["prompt"]
        reference_answer = obj["reference_answer"]
        dataset_name = obj["dataset"]
        # Run agent on prompt; save (problem_id, model_answer) for evaluation
```

### 2. Run across all six datasets

To run a horizontal test over all benchmarks:

1. Concatenate or iterate over the six files: `hle.jsonl`, `gaia.jsonl`, `browsecomp_en.jsonl`, `browsecomp_zh.jsonl`, `xbench_deepsearch.jsonl`, `frames.jsonl`.
2. For each record, use `prompt` as the agent input.
3. Record the model output keyed by `id` (and optionally `dataset`).
4. Save results in the format expected by the pass@k evaluator (see below).

Example: collect answers into a JSONL file where each line has `id` and `model_answer` (or `model_answers` for pass@k):

```json
{"id": "hle_1", "model_answer": "D"}
{"id": "gaia_0", "model_answer": "egalitarian"}
```

### 3. Evaluate with pass@k

Use the **pass@k** script in `../dataset_preparation/evaluate_pass_at_k.py` to compute:

- **pass@1**: fraction of problems solved correctly in a single attempt.
- **pass@k** (k > 1): for each problem, use k independent model answers; the problem is “solved” if **at least one** of the k answers is correct.

Correctness is determined by **LLM-as-a-Judge** (or optional exact/normalized match). See [Evaluation Metrics](#evaluation-metrics-passk) below and the script’s `--help`.

**Example:**

```bash
cd examples/scaffolding/benchmarks/dataset_preparation
python evaluate_pass_at_k.py \
  --datasets-dir ../datasets \
  --answers-file /path/to/model_answers.jsonl \
  --judge llm --judge-model openai:gpt-4o-mini
```

---

## Model answers file format (for pass@k script)

The evaluator expects a JSONL file where each line is a JSON object with:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Must match an `id` in the dataset JSONL files |
| `model_answer` | string | For pass@1 | Single model output for this problem |
| `model_answers` | list of strings | For pass@k | k independent model outputs; problem solved if any is correct |
| `correct` | bool | No | If present, overrides judge (for precomputed labels) |

For **pass@1**, provide `model_answer`. For **pass@k** (e.g. k=5), provide `model_answers` with k strings. The script loads reference answers from the dataset files using `id`.

---

## Evaluation Metrics: pass@k

We adopt the **pass@k** metric (Chen et al., 2021):

- **pass@1**: Percentage of problems solved correctly in a **single** attempt.  
  For a dataset with n problems,  
  **pass@1 = (1/n) ∑ᵢ I(problem i is solved)**  
  where I(·) is the indicator function.

- **pass@k** (k > 1): For each problem we generate **k** independent samples (e.g. with nucleus sampling). The problem is solved if **at least one** of the k answers is correct.  
  **pass@k = (1/n) ∑ᵢ I(∃ correct among k samples for problem i)**.

To decide whether a generated solution is correct we use **LLM-as-a-Judge** (Liu et al., 2024; Wang et al., 2024). For generation we use **nucleus sampling** with **temperature 0.6** and **top-p 0.95**.

The script `evaluate_pass_at_k.py` reports overall and per-dataset pass@1 and pass@k.

---

## File layout (this directory)

```
datasets/
├── README.md              # This file
├── hle.jsonl
├── gaia.jsonl
├── browsecomp_en.jsonl
├── browsecomp_zh.jsonl
├── xbench_deepsearch.jsonl
└── frames.jsonl
```

All preparation and evaluation code (download, normalize, load_utils, pass@k script) is in **`../dataset_preparation/`**.

---

## Licenses and citations

Each dataset has its own license and citation. See the links below and cite the corresponding papers when publishing results:

- **HLE**: [lastexam.ai](https://lastexam.ai) / [Paper](https://lastexam.ai/paper)
- **GAIA**: [Hugging Face GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- **BrowseComp**: [OpenAI BrowseComp](https://openai.com/index/browsecomp/) / [simple-evals](https://github.com/openai/simple-evals) (MIT)
- **BrowseComp-ZH**: [GitHub PALIN2018/BrowseComp-ZH](https://github.com/PALIN2018/BrowseComp-ZH)
- **Xbench-DeepSearch**: [xbench.org](https://xbench.org) / [xbench-evals](https://github.com/xbench-ai/xbench-evals)
- **FRAMES**: [google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) (Apache 2.0)
