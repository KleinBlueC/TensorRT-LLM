
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR_DEFAULT = SCRIPT_DIR / ".." / "datasets"
DATASET_NAMES = ["hle", "gaia", "browsecomp_en", "browsecomp_zh", "xbench_deepsearch", "frames"]


def _normalize(s: str) -> str:
    """Normalize string for exact-match comparison: strip, lowercase, collapse whitespace."""
    if not isinstance(s, str):
        s = str(s).strip()
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _exact_match(reference: str, model_answer: str) -> bool:
    """True if normalized model answer equals normalized reference."""
    return _normalize(model_answer) == _normalize(reference)


def _load_dataset_records(datasets_dir: Path) -> dict:
    """Load all records from the six JSONL files; return id -> {prompt, reference_answer, dataset}."""
    datasets_dir = datasets_dir.resolve()
    records = {}
    for name in DATASET_NAMES:
        path = datasets_dir / f"{name}.jsonl"
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id")
                if not rid:
                    continue
                records[rid] = {
                    "prompt": obj.get("prompt", ""),
                    "reference_answer": obj.get("reference_answer", ""),
                    "dataset": obj.get("dataset", name),
                }
    return records


def _load_answers(answers_path: Path) -> dict:
    """Load model answers JSONL; return id -> {model_answer or model_answers, correct (optional)}."""
    answers = {}
    with open(answers_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if not rid:
                continue
            answers[rid] = {
                "model_answer": obj.get("model_answer"),
                "model_answers": obj.get("model_answers"),
                "correct": obj.get("correct"),
            }
    return answers


def _llm_judge_single(prompt: str, reference: str, model_answer: str, model: str) -> bool:
    """Use LLM to judge if model_answer is correct given prompt and reference. Returns True/False."""
    try:
        import openai
    except ImportError:
        raise ImportError("LLM judge requires: pip install openai") from None
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if ":" in model:
        provider, model_id = model.split(":", 1)
    else:
        provider, model_id = "openai", model
    if provider != "openai":
        raise ValueError("Only openai provider is supported for --judge llm")
    user = (
        f"Question: {prompt[:2000]}\n\n"
        f"Reference (gold) answer: {reference[:500]}\n\n"
        f"Model answer: {model_answer[:1000]}\n\n"
        "Is the model answer correct (equivalent or acceptable compared to the reference)? Reply with exactly one word: YES or NO."
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": user}],
        max_tokens=10,
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    return "YES" in text and "NO" not in text[:text.find("YES") + 3]


def _judge_correct(
    reference: str,
    model_answer: str,
    judge: str,
    judge_model: str | None,
    prompt: str,
) -> bool:
    """Determine if model_answer is correct. Uses precomputed 'correct', exact match, or LLM judge."""
    if judge == "exact":
        return _exact_match(reference, model_answer)
    if judge == "llm" and judge_model:
        return _llm_judge_single(prompt, reference, model_answer, judge_model)
    return _exact_match(reference, model_answer)


def compute_pass_at_k(
    dataset_records: dict,
    answers: dict,
    judge: str = "exact",
    judge_model: str | None = None,
) -> tuple[dict[str, list[bool]], dict[str, int], dict[str, float], dict[str, float]]:
    """
    For each id in both dataset and answers, compute whether the problem was solved.
    Returns:
      - results_by_id: id -> list of bools (one per sample; for pass@k we use any True)
      - n_by_dataset: dataset -> count of problems evaluated
      - pass1_by_dataset: dataset -> pass@1
      - passk_by_dataset: dataset -> pass@k (using all available samples per problem)
    """
    results_by_id = {}
    for rid, rec in dataset_records.items():
        if rid not in answers:
            continue
        ans = answers[rid]
        ref = rec["reference_answer"]
        prompt = rec["prompt"]
        ds = rec["dataset"]
        if ans.get("correct") is not None:
            # Precomputed correctness: treat as single sample
            results_by_id[rid] = [bool(ans["correct"])]
            continue
        model_answers = ans.get("model_answers")
        if model_answers is None and ans.get("model_answer") is not None:
            model_answers = [ans["model_answer"]]
        if not model_answers:
            continue
        correct_list = []
        for ma in model_answers:
            c = _judge_correct(ref, ma, judge, judge_model, prompt)
            correct_list.append(c)
        results_by_id[rid] = correct_list
    n_by_dataset = defaultdict(int)
    solved_pass1 = defaultdict(int)
    solved_passk = defaultdict(int)
    for rid, correct_list in results_by_id.items():
        ds = dataset_records[rid]["dataset"]
        n_by_dataset[ds] += 1
        if correct_list[0]:
            solved_pass1[ds] += 1
        if any(correct_list):
            solved_passk[ds] += 1
    pass1_by_dataset = {ds: (solved_pass1[ds] / n if n else 0.0) for ds, n in n_by_dataset.items()}
    passk_by_dataset = {ds: (solved_passk[ds] / n if n else 0.0) for ds, n in n_by_dataset.items()}
    return results_by_id, dict(n_by_dataset), pass1_by_dataset, passk_by_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Compute pass@1 and pass@k for agent answers against the six benchmark datasets."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATASETS_DIR_DEFAULT,
        help="Directory containing the six .jsonl dataset files.",
    )
    parser.add_argument(
        "--answers-file",
        type=Path,
        required=True,
        help="JSONL with id and model_answer (or model_answers for pass@k).",
    )
    parser.add_argument(
        "--judge",
        choices=["exact", "llm"],
        default="exact",
        help="How to determine correctness: exact (normalized string match) or llm (LLM-as-a-Judge).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai:gpt-4o-mini",
        help="Judge model for --judge llm (e.g. openai:gpt-4o-mini). Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional: write per-problem results (id, correct, dataset) to this JSONL.",
    )
    args = parser.parse_args()
    args.datasets_dir = args.datasets_dir.resolve()
    if not args.answers_file.is_file():
        print(f"Answers file not found: {args.answers_file}", file=sys.stderr)
        sys.exit(1)
    if args.judge == "llm" and not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set; LLM judge may fail.", file=sys.stderr)

    dataset_records = _load_dataset_records(args.datasets_dir)
    answers = _load_answers(args.answers_file)
    common = set(dataset_records) & set(answers)
    if not common:
        print("No common ids between dataset and answers file.", file=sys.stderr)
        sys.exit(1)
    print(f"Evaluating {len(common)} problems (judge={args.judge})")

    results_by_id, n_by_dataset, pass1_by_dataset, passk_by_dataset = compute_pass_at_k(
        dataset_records, answers, judge=args.judge, judge_model=args.judge_model
    )
    n_total = sum(n_by_dataset.values())
    pass1_total = sum(1 for r in results_by_id.values() if r[0]) / n_total if n_total else 0.0
    passk_total = sum(1 for r in results_by_id.values() if any(r)) / n_total if n_total else 0.0

    print("\n--- pass@1 ---")
    print(f"Overall: {pass1_total:.4f} ({int(pass1_total * n_total)}/{n_total})")
    for ds in DATASET_NAMES:
        if ds in pass1_by_dataset:
            print(f"  {ds}: {pass1_by_dataset[ds]:.4f} ({n_by_dataset[ds]} problems)")
    print("\n--- pass@k (at least one correct per problem) ---")
    print(f"Overall: {passk_total:.4f} ({int(passk_total * n_total)}/{n_total})")
    for ds in DATASET_NAMES:
        if ds in passk_by_dataset:
            print(f"  {ds}: {passk_by_dataset[ds]:.4f} ({n_by_dataset[ds]} problems)")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for rid in sorted(results_by_id.keys()):
                rec = dataset_records[rid]
                correct_list = results_by_id[rid]
                f.write(
                    json.dumps(
                        {
                            "id": rid,
                            "dataset": rec["dataset"],
                            "correct_pass1": correct_list[0],
                            "correct_passk": any(correct_list),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"\nWrote per-problem results to {args.output_file}")


if __name__ == "__main__":
    main()
