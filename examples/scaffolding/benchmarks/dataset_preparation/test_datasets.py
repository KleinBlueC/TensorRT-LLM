# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Test script: validate reading of all six benchmark datasets, check schema/fields,
# and print per-dataset statistics. Run: python test_datasets.py [--datasets-dir ../datasets]

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR_DEFAULT = (SCRIPT_DIR / ".." / "datasets").resolve()

REQUIRED_FIELDS = ("id", "prompt", "reference_answer", "dataset", "metadata")
VALID_DATASET_NAMES = {"hle", "gaia", "browsecomp_en", "browsecomp_zh", "xbench_deepsearch", "frames"}


def _load_jsonl(path: Path):
    """Yield one JSON object per non-empty line."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i}: invalid JSON: {e}") from e


def validate_and_stats(datasets_dir: Path, verbose: bool = True) -> dict:
    """
    Validate each dataset JSONL and compute statistics.
    Returns a dict: dataset_name -> { ok, errors[], stats }.
    """
    datasets_dir = datasets_dir.resolve()
    results = {}

    for name in sorted(VALID_DATASET_NAMES):
        path = datasets_dir / f"{name}.jsonl"
        entry = {"ok": False, "errors": [], "stats": {}}
        results[name] = entry

        if not path.exists():
            entry["errors"].append(f"File not found: {path}")
            if verbose:
                print(f"[{name}] SKIP: file not found")
            continue

        records = []
        ids_seen = set()
        prompt_lengths = []
        ref_lengths = []
        metadata_keys = defaultdict(int)

        try:
            for line_no, obj in _load_jsonl(path):
                if not isinstance(obj, dict):
                    entry["errors"].append(f"Line {line_no}: expected JSON object, got {type(obj).__name__}")
                    continue
                # Required fields
                for field in REQUIRED_FIELDS:
                    if field not in obj:
                        entry["errors"].append(f"Line {line_no}: missing field '{field}'")
                if entry["errors"]:
                    continue
                # Types
                if not isinstance(obj["id"], str) or not obj["id"].strip():
                    entry["errors"].append(f"Line {line_no}: 'id' must be non-empty string")
                if not isinstance(obj["prompt"], str):
                    entry["errors"].append(f"Line {line_no}: 'prompt' must be string")
                if not isinstance(obj["reference_answer"], str):
                    entry["errors"].append(f"Line {line_no}: 'reference_answer' must be string")
                if not isinstance(obj["dataset"], str):
                    entry["errors"].append(f"Line {line_no}: 'dataset' must be string")
                if not isinstance(obj["metadata"], dict):
                    entry["errors"].append(f"Line {line_no}: 'metadata' must be object")
                if obj.get("dataset") != name:
                    entry["errors"].append(
                        f"Line {line_no}: dataset field is '{obj.get('dataset')}', expected '{name}'"
                    )
                rid = obj.get("id", "")
                if rid in ids_seen:
                    entry["errors"].append(f"Line {line_no}: duplicate id '{rid}'")
                ids_seen.add(rid)
                # Id prefix should match dataset
                if rid and not rid.startswith(name + "_") and not rid.startswith(name):
                    entry["errors"].append(f"Line {line_no}: id '{rid}' does not match dataset prefix '{name}'")
                records.append(obj)
                prompt_lengths.append(len(obj.get("prompt") or ""))
                ref_lengths.append(len(obj.get("reference_answer") or ""))
                for k in (obj.get("metadata") or {}).keys():
                    metadata_keys[k] += 1
        except (OSError, ValueError) as e:
            entry["errors"].append(str(e))
            if verbose:
                print(f"[{name}] ERROR: {e}")
            continue

        if entry["errors"]:
            entry["ok"] = False
            if verbose:
                for err in entry["errors"][:10]:
                    print(f"  {err}")
                if len(entry["errors"]) > 10:
                    print("  ... and {} more".format(len(entry["errors"]) - 10))
            continue

        n = len(records)
        entry["ok"] = True

        def _stats(arr):
            if not arr:
                return {"min": 0, "max": 0, "mean": 0.0, "median": 0}
            s = sorted(arr)
            return {
                "min": min(arr),
                "max": max(arr),
                "mean": round(sum(arr) / len(arr), 1),
                "median": s[len(s) // 2] if s else 0,
            }

        entry["stats"] = {
            "count": n,
            "prompt_length": _stats(prompt_lengths),
            "reference_answer_length": _stats(ref_lengths),
            "metadata_keys": dict(metadata_keys),
            "sample_ids": [r["id"] for r in records[:3]],
        }
        results[name] = entry
        if verbose:
            _print_stats(name, entry["stats"])
    return results


def _print_stats(name: str, stats: dict) -> None:
    """Print one dataset's statistics."""
    n = stats["count"]
    pl = stats["prompt_length"]
    rl = stats["reference_answer_length"]
    print(f"\n[{name}] OK — {n} records")
    print(f"  prompt length:  min={pl['min']}, max={pl['max']}, mean={pl['mean']}, median={pl['median']}")
    print(f"  ref ans length: min={rl['min']}, max={rl['max']}, mean={rl['mean']}, median={rl['median']}")
    if stats.get("metadata_keys"):
        print(f"  metadata keys:  {dict(stats['metadata_keys'])}")
    print(f"  sample ids:     {stats.get('sample_ids', [])}")


def main():
    parser = argparse.ArgumentParser(description="Validate and report statistics for the six benchmark datasets.")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR_DEFAULT,
        help="Directory containing the six .jsonl files.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print summary and errors, no per-dataset stats.",
    )
    args = parser.parse_args()
    args.datasets_dir = args.datasets_dir.resolve()

    if not args.datasets_dir.is_dir():
        print(f"Datasets dir not found: {args.datasets_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Datasets directory: {args.datasets_dir}")
    print("Validating schema: id, prompt, reference_answer, dataset, metadata")
    results = validate_and_stats(args.datasets_dir, verbose=not args.quiet)

    ok_count = sum(1 for r in results.values() if r["ok"])
    total_records = sum(r["stats"].get("count", 0) for r in results.values() if r["ok"])
    has_errors = [name for name, r in results.items() if r["errors"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Datasets OK:     {ok_count}/{len(VALID_DATASET_NAMES)}")
    print(f"  Total records:   {total_records}")
    if has_errors:
        print(f"  With errors:      {', '.join(has_errors)}")
    for name in sorted(results.keys()):
        r = results[name]
        if r["ok"]:
            print(f"  {name}: {r['stats'].get('count', 0)} records")
        else:
            print(f"  {name}: FAILED ({len(r['errors'])} errors)")
    print("=" * 60)

    if has_errors:
        sys.exit(1)
    print("All datasets valid.\n")


if __name__ == "__main__":
    main()
