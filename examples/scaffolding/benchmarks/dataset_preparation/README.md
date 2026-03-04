# Dataset Preparation Scripts

This folder contains scripts to **download**, **normalize**, and **prepare** the six deep-search benchmark datasets. The normalized JSONL files are written to the sibling directory `../datasets/` (so that `datasets/` holds only the six unified dataset files).

## Contents

- **download_and_normalize.py** — Download all six datasets from their official sources and normalize to unified JSONL. Use `--output-dir ../datasets` to write into the datasets folder.
- **normalize_local_hle_gaia.py** — Normalize HLE and GAIA from local HuggingFace cache (Parquet blobs) when you already have `hf_data/datasets--cais--hle` and `datasets--gaia-benchmark--GAIA`. No Hub login required.
- **load_utils.py** — Utilities to load prompts and full records from the normalized JSONL (for use by benchmarks or evaluation scripts).
- **evaluate_pass_at_k.py** — Compute pass@1 and pass@k metrics using LLM-as-a-Judge; see below and the main [datasets README](../datasets/README.md).
- **test_datasets.py** — Validate that all six JSONL files are readable, schema and fields are correct, and print per-dataset statistics (counts, prompt/answer length, metadata).

## Quick start (regenerate datasets)

```bash
# From this directory
pip install pandas requests datasets openpyxl pyarrow

# Download and normalize all (HLE/GAIA require huggingface-cli login)
python download_and_normalize.py --output-dir ../datasets --combined

# Or only from local cache for HLE/GAIA
python normalize_local_hle_gaia.py --hf-data-dir /path/to/hf_data --output-dir ../datasets
```

## Using load_utils from Python

```python
from examples.scaffolding.benchmarks.dataset_preparation import load_utils

datasets_dir = "examples/scaffolding/benchmarks/datasets"
prompts = load_utils.load_prompts_from_deepsearch_jsonl(f"{datasets_dir}/frames.jsonl", max_num=50)
records = load_utils.load_deepsearch_records(f"{datasets_dir}/all_deepsearch.jsonl", max_num=100)
```

## Evaluation: pass@k

Run the pass@k evaluation script from this directory (or set `--datasets-dir` and `--answers-file` accordingly). See **../datasets/README.md** for full usage and the expected format of the model answers file.

```bash
python evaluate_pass_at_k.py --datasets-dir ../datasets --answers-file /path/to/model_answers.jsonl
```

**Validate datasets and show statistics:**

```bash
python test_datasets.py --datasets-dir ../datasets
```
