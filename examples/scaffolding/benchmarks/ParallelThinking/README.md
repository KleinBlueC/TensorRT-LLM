# HeavyThinking Web Search Benchmarks

Two benchmark scripts that run the HeavyThinking web researcher on the **deep-search datasets** (`../datasets/`), one question at a time, and write **id** + **model_answer** to a JSONL file for later pass@k evaluation.

## Modes

| Script | Mode | Description |
|--------|------|-------------|
| `run_web_search_iterative.py` | **Iterative** | Single research run per question (Thinker → Reporter → Actor loop with MCP tools until answer). |
| `run_web_search_heavy_thinking.py` | **HeavyThinking** | N parallel IterativeResearcher runs per question, then one synthesizer to produce the final answer. |

## Datasets

Use `--dataset` with a short name or path:

- Short names: `gaia`, `hle`, `frames`, `browsecomp_en`, `browsecomp_zh`, `xbench_deepsearch`
- Or path: `path/to/any.jsonl` (same schema: `id`, `prompt`, `reference_answer`, `dataset`, `metadata`)

Datasets live in `../datasets/` (see `../datasets/README.md`).

## Common Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | **Required.** Dataset name or path to JSONL. |
| `--max_num` | Max number of questions (default: all). |
| `--output` | Output JSONL path (default: `answers_iterative.jsonl` or `answers_heavy_thinking.jsonl`). |
| `--base_url` | OpenAI-compatible API base URL (default: `http://localhost:8000/v1`). |
| `--model` | Model name for the API (default: `Qwen3/Qwen3-30B-A3B`). |
| `--mcp_url` | MCP server SSE URL (default: `http://0.0.0.0:8082/sse`). |
| `--max_tokens` | Max tokens per generation (default: 16384). |
| `--temperature` | Sampling temperature (default: 0.2). |

## HeavyThinking-only

| Argument | Description |
|----------|-------------|
| `--n` | Number of parallel IterativeResearcher runs to synthesize (default: 3). |
| `--workspace_log_root` | Optional directory for workspace logs (reports/tool_calls). |

## Examples

From repo root:

```bash
# Iterative: 5 questions from GAIA, save to ./out/iter.jsonl
python -m examples.scaffolding.benchmarks.HeavyThinking.run_web_search_iterative \
  --dataset gaia --max_num 5 --output ./out/iter.jsonl

# HeavyThinking: N=3 parallel runs, 5 questions from frames
python -m examples.scaffolding.benchmarks.HeavyThinking.run_web_search_heavy_thinking \
  --dataset frames --max_num 5 --n 3 --output ./out/ht.jsonl
```

From this directory (`examples/scaffolding/benchmarks/HeavyThinking`), with `PYTHONPATH` set to repo root:

```bash
export PYTHONPATH=/path/to/TensorRT-LLM
python run_web_search_iterative.py --dataset gaia --max_num 2 --output ./iter.jsonl
python run_web_search_heavy_thinking.py --dataset gaia --max_num 2 --n 2 --output ./ht.jsonl
```

## Output format

Each line of the output JSONL is one JSON object:

```json
{"id": "gaia_0", "model_answer": "..."}
```

Use `../dataset_preparation/evaluate_pass_at_k.py` with `--answers-file` pointing to this file to compute pass@1 (and pass@k) against the dataset’s `reference_answer`.

## Prerequisites

- TensorRT-LLM serve running with agent config and tool parser.
- MCP server running (e.g. `tensorrt_llm/scaffolding/contrib/HeavyThinking/mcp`: `uv run launch_mcp_server.py`).
- API keys in env (e.g. `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` for search) as needed by the MCP tools.
