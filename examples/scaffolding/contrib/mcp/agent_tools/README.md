# Agent Tools MCP Server

MCP server exposing four tools for agent use: **search**, **scholar**, **visit**, and **python**.

## Tools

| Tool     | Description                              |
|----------|------------------------------------------|
| search   | Web search via Tavily API                |
| scholar  | Academic search via Google Scholar (scholarly package; no API key) |
| visit    | Extract full page content from a URL via Jina Reader (jina.ai); optional API key |
| python   | Run Python code in a sandbox (E2B Code Interpreter; data/plot libs, stdout/stderr/results) |

## Setup

1. Copy environment template and set API keys:

   ```bash
   cp .env.example .env
   # Edit .env: TAVILY_API_KEY (search), SEMANTIC_SCHOLAR_KEY (scholar), etc.
   ```

   - Tavily: [tavily.com](https://tavily.com) (free tier).
   - Scholar uses Google Scholar via the `scholarly` package (no key). Optional backup: `scholar_semantic_scholar.py` uses Semantic Scholar and needs `SEMANTIC_SCHOLAR_KEY`.
   - Visit uses [Jina Reader](https://jina.ai) (no key required; set `JINA_API_KEY` for higher rate limits).
   - Python sandbox uses [E2B Code Interpreter](https://e2b.dev); set `E2B_API_KEY` in `.env` to enable.

2. Install dependencies:

   ```bash
   cd examples/scaffolding/contrib/mcp/agent_tools
   uv sync
   ```

## Run

```bash
uv run server.py
# Optional: --host 0.0.0.0 --port 8083
```

Default: `http://0.0.0.0:8083`. SSE endpoint: `http://<host>:<port>/sse`.

## Testing

### Step 1: Unit tests (no API keys required for key-check tests)

From the `agent_tools` directory:

```bash
cd examples/scaffolding/contrib/mcp/agent_tools
uv sync --extra dev
uv run pytest tests/ -v
```

- `test_search.py`: missing/empty key tests always run; real API tests run when `TAVILY_API_KEY` is set in `.env`.
- `test_scholar.py`: real Google Scholar search tests (no API key); run with `uv run pytest tests/test_scholar.py -v -s` to see printed results.
- `test_python_run.py`: missing-key test always runs; real E2B run when `E2B_API_KEY` is set.

### Step 2: Manual test of the search tool (real API)

1. Set your Tavily key and start the server:

   ```bash
   export TAVILY_API_KEY=your_key_here
   uv run server.py --port 8083
   ```

2. In another terminal, call the MCP server (e.g. with the MCP CLI or a small script that connects to `http://0.0.0.0:8083/sse`, lists tools, then calls `search` with a query). Alternatively, add `http://0.0.0.0:8083/sse` to `mcptest.py` and run mcptest with a prompt that triggers web search.

### Step 3: End-to-end with mcptest

1. Start the agent_tools server (with `TAVILY_API_KEY` set).
2. In `mcptest.py`, set e.g. `urls = ["http://0.0.0.0:8083/sse"]`.
3. Run `python3 mcptest.py --API_KEY <your_llm_api_key>` and use a prompt that requires web search; the agent will receive the `search` tool and can call it.

## Use with mcptest

Add the agent_tools SSE URL to `urls` in `mcptest.py`, e.g.:

```python
urls = ["http://0.0.0.0:8083/sse", "http://0.0.0.0:8082/sse"]
```

Then run mcptest; the agent will see all four tools via `list_tools` and can call them by name.
