# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Utilities to load normalized deep-search benchmark JSONL for scaffolding agent pipelines.

import json
from pathlib import Path
from typing import Iterator, List


def load_prompts_from_deepsearch_jsonl(
    path: str | Path, max_num: int | None = None
) -> List[str]:
    """Load prompt strings from a normalized deep-search JSONL file.

    Args:
        path: Path to a .jsonl file (e.g. frames.jsonl or all_deepsearch.jsonl).
        max_num: Maximum number of prompts to return; None = all.

    Returns:
        List of prompt strings for agent input.
    """
    path = Path(path)
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            if prompt:
                prompts.append(prompt)
            if max_num is not None and len(prompts) >= max_num:
                break
    return prompts


def load_deepsearch_records(
    path: str | Path, max_num: int | None = None
) -> List[dict]:
    """Load full records (id, prompt, reference_answer, dataset, metadata) from JSONL.

    Args:
        path: Path to a .jsonl file.
        max_num: Maximum number of records; None = all.

    Returns:
        List of dicts with keys id, prompt, reference_answer, dataset, metadata.
    """
    path = Path(path)
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


def iter_deepsearch_records(path: str | Path) -> Iterator[dict]:
    """Iterate over records in a normalized deep-search JSONL file (memory-friendly)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
