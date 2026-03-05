
import argparse
import json
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("Required: pip install pyarrow") from None

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_DEFAULT = (SCRIPT_DIR / ".." / "datasets").resolve()

# Parquet blob names (from HuggingFace cache layout)
HLE_PARQUET_BLOB = "6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a"


def _find_parquet_blobs(cache_dir: Path, prefix: str) -> list[Path]:
    """Find all Parquet files in cache_dir/prefix/blobs/ (by reading magic bytes)."""
    blobs_dir = cache_dir / prefix / "blobs"
    if not blobs_dir.is_dir():
        return []
    out = []
    for f in blobs_dir.iterdir():
        if f.is_file() and f.stat().st_size > 100:
            try:
                with open(f, "rb") as fp:
                    head = fp.read(4)
                if head == b"PAR1":
                    out.append(f)
            except Exception:
                pass
    return out


def _normalize_hle(cache_dir: Path) -> list[dict]:
    """HLE: text-only subset from cais/hle Parquet (same logic as download_and_normalize.fetch_hle)."""
    parquet_path = cache_dir / "datasets--cais--hle" / "blobs" / HLE_PARQUET_BLOB
    if not parquet_path.is_file():
        blobs = _find_parquet_blobs(cache_dir, "datasets--cais--hle")
        if not blobs:
            raise FileNotFoundError(f"No HLE Parquet found under {cache_dir / 'datasets--cais--hle'}")
        parquet_path = blobs[0]
    table = pq.read_table(parquet_path)
    records = []
    n = table.num_rows
    question_col = table.column("question") if "question" in table.column_names else None
    answer_col = table.column("answer") if "answer" in table.column_names else None
    image_col = table.column("image") if "image" in table.column_names else None
    raw_subject = table.column("raw_subject") if "raw_subject" in table.column_names else None
    category_col = table.column("category") if "category" in table.column_names else None
    if question_col is None:
        raise ValueError("HLE Parquet missing 'question' column")
    for i in range(n):
        if image_col is not None:
            img = image_col[i]
            if img is not None:
                s = str(img) if not isinstance(img, (str, type(None))) else img
                if s and s.strip() and s.strip().lower() != "none":
                    continue
        q = question_col[i]
        question = (q.as_py() if hasattr(q, "as_py") else q) or ""
        if isinstance(question, bytes):
            question = question.decode("utf-8", errors="replace")
        if not str(question).strip():
            continue
        a = answer_col[i] if answer_col is not None else None
        answer = (a.as_py() if a is not None and hasattr(a, "as_py") else a) or ""
        if isinstance(answer, bytes):
            answer = answer.decode("utf-8", errors="replace")
        meta = {}
        if raw_subject is not None and raw_subject[i] is not None:
            s = raw_subject[i]
            meta["subject"] = s.as_py() if hasattr(s, "as_py") else s
        if category_col is not None and category_col[i] is not None:
            c = category_col[i]
            meta["category"] = c.as_py() if hasattr(c, "as_py") else c
        records.append({
            "id": f"hle_{i}",
            "prompt": str(question),
            "reference_answer": str(answer),
            "dataset": "hle",
            "metadata": meta,
        })
    return records


def _normalize_gaia(cache_dir: Path, text_only_limit: int | None = 103) -> list[dict]:
    """GAIA: validation subset, text-only from Parquet blobs (same logic as fetch_gaia)."""
    blobs = _find_parquet_blobs(cache_dir, "datasets--gaia-benchmark--GAIA")
    if not blobs:
        raise FileNotFoundError(f"No GAIA Parquet found under {cache_dir / 'datasets--gaia-benchmark--GAIA'}")
    all_records = []
    global_idx = 0
    for path in sorted(blobs):
        table = pq.read_table(path)
        if "Question" not in table.column_names:
            continue
        question_col = table.column("Question")
        answer_col = table.column("Final answer") if "Final answer" in table.column_names else None
        if answer_col is None and "Final_answer" in table.column_names:
            answer_col = table.column("Final_answer")
        file_path_col = table.column("file_path") if "file_path" in table.column_names else None
        file_name_col = table.column("file_name") if "file_name" in table.column_names else None
        level_col = table.column("Level") if "Level" in table.column_names else None
        n = table.num_rows
        for i in range(n):
            fp_val = None
            if file_path_col is not None:
                fp_val = file_path_col[i]
            if fp_val is None and file_name_col is not None:
                fp_val = file_name_col[i]
            if fp_val is not None:
                s = fp_val.as_py() if hasattr(fp_val, "as_py") else fp_val
                if s and str(s).strip():
                    continue
            q = question_col[i]
            question = (q.as_py() if hasattr(q, "as_py") else q) or ""
            if isinstance(question, bytes):
                question = question.decode("utf-8", errors="replace")
            if not str(question).strip():
                continue
            a = answer_col[i] if answer_col is not None else None
            answer = (a.as_py() if a is not None and hasattr(a, "as_py") else a) or ""
            if isinstance(answer, bytes):
                answer = answer.decode("utf-8", errors="replace")
            level = None
            if level_col is not None and level_col[i] is not None:
                level = level_col[i].as_py() if hasattr(level_col[i], "as_py") else level_col[i]
            all_records.append({
                "id": f"gaia_{global_idx}",
                "prompt": str(question),
                "reference_answer": str(answer),
                "dataset": "gaia",
                "metadata": {"level": level} if level is not None else {},
            })
            global_idx += 1
    if text_only_limit is not None and len(all_records) > text_only_limit:
        all_records = all_records[:text_only_limit]
    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="Normalize HLE and GAIA from local HuggingFace cache (Parquet) to unified JSONL."
    )
    parser.add_argument(
        "--hf-data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "hf_data",
        help="Directory containing datasets--cais--hle and datasets--gaia-benchmark--GAIA.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory for hle.jsonl and gaia.jsonl (default: ../datasets).",
    )
    parser.add_argument(
        "--gaia-limit",
        type=int,
        default=103,
        help="Max number of GAIA text-only records (default: 103). Use 0 for no limit.",
    )
    args = parser.parse_args()
    hf_data = args.hf_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gaia_limit = None if args.gaia_limit == 0 else args.gaia_limit

    if not hf_data.is_dir():
        print(f"HF data dir not found: {hf_data}", file=sys.stderr)
        sys.exit(1)

    for name, normalize_fn, out_file in [
        ("hle", lambda: _normalize_hle(hf_data), "hle.jsonl"),
        ("gaia", lambda: _normalize_gaia(hf_data, text_only_limit=gaia_limit), "gaia.jsonl"),
    ]:
        print(f"Normalizing {name} from {hf_data}...")
        try:
            records = normalize_fn()
        except Exception as e:
            print(f"Error normalizing {name}: {e}", file=sys.stderr)
            continue
        print(f"  {len(records)} records")
        out_path = output_dir / out_file
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
