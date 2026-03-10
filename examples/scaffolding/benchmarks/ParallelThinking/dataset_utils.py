"""Resolve dataset name/path and load records for HeavyThinking web search benchmarks."""

from pathlib import Path
from typing import List

# Known dataset short names -> filename in benchmarks/datasets
DATASET_FILES = {
    "gaia": "gaia.jsonl",
    "hle": "hle.jsonl",
    "browsecomp_en": "browsecomp_en.jsonl",
    "browsecomp_zh": "browsecomp_zh.jsonl",
    "xbench_deepsearch": "xbench_deepsearch.jsonl",
    "frames": "frames.jsonl",
}


def get_datasets_dir() -> Path:
    """Return benchmarks/datasets directory (sibling of HeavyThinking)."""
    return Path(__file__).resolve().parent.parent / "datasets"


def resolve_dataset_path(dataset: str) -> Path:
    """Resolve --dataset to a JSONL file path.

    Args:
        dataset: Short name (e.g. gaia, hle) or path to a .jsonl file.

    Returns:
        Path to the JSONL file.
    """
    path = Path(dataset)
    if path.suffix == ".jsonl" and path.exists():
        return path.resolve()
    if path.suffix == ".jsonl":
        return path
    # Short name
    name = dataset.strip().lower()
    if name in DATASET_FILES:
        return get_datasets_dir() / DATASET_FILES[name]
    # Try as filename in datasets dir
    candidates = [get_datasets_dir() / f"{name}.jsonl", get_datasets_dir() / name]
    for c in candidates:
        if c.exists():
            return c
    return get_datasets_dir() / f"{name}.jsonl"


def load_records(dataset: str, max_num: int | None = None) -> List[dict]:
    """Load deep-search records from dataset (id, prompt, reference_answer, dataset, metadata)."""
    import json

    path = resolve_dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_num is not None and len(records) >= max_num:
                break
    return records
