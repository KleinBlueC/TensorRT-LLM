# ReAct example with Tavily search only

This example runs the ReAct controller with a single MCP tool: **Tavily search** (`web_search`).

## Prerequisites

1. **LLM server**: An OpenAI-compatible API (e.g. `trtllm-serve`) for generation. The server should support **tool calling** (returning `tool_calls` in the chat completion response) so the actor receives structured tool calls.
2. **Tavily MCP server**: The server must expose the tool `web_search` (e.g. the TavilyMCP in `examples/scaffolding/contrib/open_deep_research/TavilyMCP/`).
3. **TAVILY_API_KEY**: Set in the environment where the MCP server runs.

## Quick start

**Terminal 1 – start Tavily MCP server** (requires `TAVILY_API_KEY` and `pip install tavily-python mcp`):

```bash
cd examples/scaffolding/contrib/open_deep_research/TavilyMCP
export TAVILY_API_KEY=your_key
python travily.py --port 8082
```

**Terminal 2 – start your LLM server** (e.g. on port 8000).

**Terminal 3 – run ReAct**:

```bash
cd examples/scaffolding/contrib/ReAct
python run_react_tavily.py \
  --base_url http://localhost:8000/v1 \
  --mcp_url http://0.0.0.0:8082/sse \
  --prompt "What is the current stock price of NVIDIA? Use the search tool once and then answer."
```

## Options

- `--base_url`: OpenAI-compatible API base URL (default: `http://localhost:8000/v1`).
- `--mcp_url`: MCP server SSE URL (default: `http://0.0.0.0:8082/sse`).
- `--model`: Model name for the generation worker.
- `--max_tokens`, `--temperature`, `--max_iteration`: Sampling and loop limits.
- `--prompt`: User question (default uses a simple NVIDIA stock query).

## Tool set

Only two tools are exposed to the model:

1. **web_search** – Tavily web search (single parameter: `query`).
2. **answer** – Virtual tool to end the loop and return the final answer.

The MCP server must implement `web_search`; `answer` is handled inside the controller and is not sent to MCP.
