# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Download and normalize deep-search agent benchmark datasets into a unified JSONL format.
# Run from dataset_preparation: python download_and_normalize.py [--datasets ...] [--output-dir ../datasets]

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

# Optional deps: pip install datasets pandas requests openpyxl
try:
    import pandas as pd
    import requests
except ImportError as e:
    raise ImportError(
        "Required: pip install pandas requests. For HLE/GAIA/FRAMES: pip install datasets. "
        "For BrowseComp-ZH: pip install openpyxl."
    ) from e

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_DEFAULT = (SCRIPT_DIR / ".." / "datasets").resolve()

# -----------------------------------------------------------------------------
# Unified record schema (one JSON object per line in .jsonl)
# - id: unique string (e.g. "hle_0")
# - prompt: question text (required for scaffolding agent input)
# - reference_answer: gold answer if available (for evaluation)
# - dataset: short name (hle, gaia, browsecomp_en, browsecomp_zh, xbench_deepsearch, frames)
# - metadata: optional dict (subject, level, domain, etc.)
# -----------------------------------------------------------------------------


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt_b64_xor(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode("utf-8")


def _xor_decrypt_key(data: bytes, key: str) -> bytes:
    key_bytes = key.encode("utf-8")
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])


def fetch_browsecomp_en(output_dir: Path) -> list:
    """BrowseComp-en: 1,266 questions from OpenAI simple-evals (encrypted CSV)."""
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(__import__("io").StringIO(resp.text))
    records = []
    for i, row in df.iterrows():
        canary = row.get("canary", "")
        try:
            problem = _decrypt_b64_xor(row.get("problem", ""), canary)
            answer = _decrypt_b64_xor(row.get("answer", ""), canary)
        except Exception:
            continue
        records.append({
            "id": f"browsecomp_en_{i}",
            "prompt": problem,
            "reference_answer": answer,
            "dataset": "browsecomp_en",
            "metadata": {},
        })
    return records


def fetch_frames(output_dir: Path) -> list:
    """FRAMES: 824 RAG questions from HuggingFace google/frames-benchmark."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skip FRAMES: pip install datasets", file=sys.stderr)
        return []
    ds = load_dataset("google/frames-benchmark", split="test", trust_remote_code=True)
    records = []
    for i, row in enumerate(ds):
        question = (
            row.get("Prompt")
            or row.get("question")
            or row.get("Question")
            or ""
        )
        answer = (
            row.get("Answer")
            or row.get("answer")
            or row.get("gold_answer")
            or ""
        )
        records.append({
            "id": f"frames_{i}",
            "prompt": question,
            "reference_answer": answer,
            "dataset": "frames",
            "metadata": {},
        })
    return records


def fetch_hle(output_dir: Path) -> list:
    """HLE: text-only subset from cais/hle (2,154 text-only; full test has 2,500)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skip HLE: pip install datasets", file=sys.stderr)
        return []
    ds = load_dataset("cais/hle", split="test", trust_remote_code=True)
    records = []
    for i, row in enumerate(ds):
        # Keep only text-only: no image or image field empty/None
        img = row.get("image") if hasattr(row, "get") else (row.get("image") if isinstance(row, dict) else None)
        if img is not None and (hasattr(img, "__len__") and len(img) > 0 if not isinstance(img, (str, type(None))) else bool(img)):
            continue
        question = row.get("question") or row.get("Question") or ""
        if not question:
            continue
        answer = row.get("answer") or row.get("Answer") or ""
        meta = {}
        if row.get("subject") is not None:
            meta["subject"] = row.get("subject")
        if row.get("category") is not None:
            meta["category"] = row.get("category")
        records.append({
            "id": f"hle_{i}",
            "prompt": question,
            "reference_answer": answer,
            "dataset": "hle",
            "metadata": meta,
        })
    return records


def fetch_gaia(output_dir: Path, text_only_limit: int | None = 103) -> list:
    """GAIA: validation subset, text-only (optionally limit to 103 as in literature)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skip GAIA: pip install datasets", file=sys.stderr)
        return []
    try:
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("gaia-benchmark/GAIA", split="validation", trust_remote_code=True)
        except Exception as e:
            print(f"Skip GAIA (gated?): {e}", file=sys.stderr)
            return []
    records = []
    for i, row in enumerate(ds):
        # Text-only: no auxiliary file or empty file path
        file_path = row.get("file_path") or row.get("file_name") or ""
        if file_path and str(file_path).strip():
            continue
        question = row.get("Question") or row.get("question") or ""
        if not question:
            continue
        answer = row.get("Final answer") or row.get("Final_answer") or row.get("answer") or ""
        records.append({
            "id": f"gaia_{i}",
            "prompt": question,
            "reference_answer": answer,
            "dataset": "gaia",
            "metadata": {"level": row.get("Level")},
        })
    if text_only_limit is not None and len(records) > text_only_limit:
        records = records[:text_only_limit]
    return records


def fetch_xbench_deepsearch(output_dir: Path) -> list:
    """Xbench-DeepSearch: from HuggingFace xbench/DeepSearch (encrypted)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skip Xbench-DeepSearch: pip install datasets", file=sys.stderr)
        return []
    try:
        ds = load_dataset("xbench/DeepSearch", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Skip Xbench-DeepSearch: {e}", file=sys.stderr)
        return []
    records = []
    for i, row in enumerate(ds):
        canary = row.get("canary", "")
        prompt_enc = row.get("prompt") or row.get("question") or ""
        answer_enc = row.get("answer") or ""
        try:
            if isinstance(prompt_enc, str) and prompt_enc and canary:
                prompt = _xor_decrypt_key(base64.b64decode(prompt_enc), canary).decode("utf-8")
            else:
                prompt = prompt_enc if isinstance(prompt_enc, str) else ""
            if isinstance(answer_enc, str) and answer_enc and canary:
                answer = _xor_decrypt_key(base64.b64decode(answer_enc), canary).decode("utf-8")
            else:
                answer = answer_enc if isinstance(answer_enc, str) else ""
        except Exception:
            prompt = prompt_enc if isinstance(prompt_enc, str) else ""
            answer = answer_enc if isinstance(answer_enc, str) else ""
        records.append({
            "id": f"xbench_deepsearch_{i}",
            "prompt": prompt,
            "reference_answer": answer,
            "dataset": "xbench_deepsearch",
            "metadata": {},
        })
    return records


def fetch_browsecomp_zh(output_dir: Path) -> list:
    """BrowseComp-zh: 289 questions from GitHub PALIN2018/BrowseComp-ZH (encrypted xlsx)."""
    try:
        import pandas as pd_excel
        _ = pd_excel.read_excel
    except ImportError:
        print("Skip BrowseComp-zh: pip install openpyxl pandas", file=sys.stderr)
        return []
    base_url = "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/main/data"
    xlsx_url = f"{base_url}/browsecomp-zh-encrypted.xlsx"
    xlsx_path = output_dir / "browsecomp_zh_encrypted.xlsx"
    if not xlsx_path.exists():
        r = requests.get(xlsx_url, timeout=60)
        r.raise_for_status()
        xlsx_path.write_bytes(r.content)
    df = pd.read_excel(xlsx_path)
    if "canary" not in df.columns:
        print("BrowseComp-zh: missing canary column", file=sys.stderr)
        return []
    records = []
    for index, row in df.iterrows():
        canary = row.get("canary")
        if pd.isna(canary) or not canary:
            continue
        question = ""
        answer = ""
        try:
            for col, key in [("Question", "question"), ("Answer", "answer")]:
                if col not in df.columns or pd.isna(row[col]):
                    continue
                raw = row[col]
                if isinstance(raw, str) and raw.strip():
                    try:
                        dec = _decrypt_b64_xor(raw, str(canary))
                        if key == "question":
                            question = dec
                        else:
                            answer = dec
                    except Exception:
                        if key == "question":
                            question = str(raw)
                        else:
                            answer = str(raw)
        except Exception:
            question = str(row.get("Question") or row.get("question") or "")
            answer = str(row.get("Answer") or row.get("answer") or "")
        if not question:
            continue
        records.append({
            "id": f"browsecomp_zh_{index}",
            "prompt": question,
            "reference_answer": answer,
            "dataset": "browsecomp_zh",
            "metadata": {},
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Download and normalize deep-search benchmark datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["browsecomp_en", "frames", "hle", "gaia", "browsecomp_zh", "xbench_deepsearch"],
        help="Which datasets to fetch (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for normalized JSONL files (default: ../datasets).",
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip datasets that require HuggingFace login (HLE, GAIA).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Write a single combined JSONL with all datasets.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fetchers = {
        "hle": fetch_hle,
        "gaia": fetch_gaia,
        "browsecomp_en": fetch_browsecomp_en,
        "browsecomp_zh": fetch_browsecomp_zh,
        "xbench_deepsearch": fetch_xbench_deepsearch,
        "frames": fetch_frames,
    }
    gated = {"hle", "gaia"}

    all_records = []
    for name in args.datasets:
        if name not in fetchers:
            print(f"Unknown dataset: {name}", file=sys.stderr)
            continue
        if args.skip_gated and name in gated:
            print(f"Skipping gated dataset: {name}")
            continue
        print(f"Fetching {name}...")
        try:
            records = fetchers[name](args.output_dir)
        except Exception as e:
            print(f"Error fetching {name}: {e}", file=sys.stderr)
            if name in gated:
                print("Try logging in: huggingface-cli login", file=sys.stderr)
            continue
        print(f"  {len(records)} records")
        out_path = args.output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_records.extend(records)

    if args.combined and all_records:
        combined_path = args.output_dir / "all_deepsearch.jsonl"
        with open(combined_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote combined {len(all_records)} records to {combined_path}")
    print("Done.")


if __name__ == "__main__":
    main()
