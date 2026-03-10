# HeavyThinking MCP Server

Model Context Protocol (MCP) server for the HeavyThinking web researcher. It exposes tools (search, scholar, visit, python_run) to LLM backends that support agent/tool calling.

For full **module layout, architecture, and feature overview** of the HeavyThinking project, see [../README.md](../README.md).

---

## Quick Start

| Step | Description |
|------|-------------|
| 1 | Start the TensorRT-LLM backend with agent config |
| 2 | Configure API keys in `.env` |
| 3 | Start the MCP server |
| 4 | Run the chat client to verify tool calls |

---

## Step 1: Start TensorRT-LLM backend

Create `.extra-llm-api-config.yml` in your working directory with:

```yaml
reorder_policy_config:
  policy_name: "AgentTree"
  policy_args:
    agent_percentage: 0.5
    agent_types: ["agent_deep_research"]
    agent_inflight_seq_num: 8
```

Then start the server (adjust model path and options as needed):

```bash
trtllm-serve serve /path/to/llm_models/Qwen3/Qwen3-30B-A3B \
  --max_num_tokens 32768 \
  --kv_cache_free_gpu_memory_fraction 0.8 \
  --extra_llm_api_options .extra-llm-api-config.yml \
  --reasoning_parser qwen3 \
  --tool_parser qwen3
```

---

## Step 2: Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Purpose |
|----------|---------|
| `TAVILY_API_KEY` | Web search (Tavily). Get a key at [tavily.com](https://tavily.com) |
| `E2B_API_KEY` | Python sandbox / code interpreter. Get a key at [e2b.dev](https://e2b.dev) |

---

## Step 3: Start the MCP server

From the repo root:

```bash
cd tensorrt_llm/scaffolding/contrib/HeavyThinking/mcp
uv run launch_mcp_server.py
```

Or from this directory:

```bash
uv run launch_mcp_server.py
```

---

## Step 4: Verify MCP tool calls

In another terminal, from the same `mcp` directory:

```bash
uv run python run_chat_with_mcp.py
```

This runs an interactive chat that uses the MCP tools against your running backend.

---

## Optional

- **Model path**: Replace `/path/to/llm_models/...` with your actual model path (e.g. `$LLM_MODELS_ROOT/Qwen3/Qwen3-30B-A3B`).
- **Backend URL**: If the LLM server is not on `localhost`, set `OPENAI_API_BASE` (or equivalent) in `.env` or your environment.
