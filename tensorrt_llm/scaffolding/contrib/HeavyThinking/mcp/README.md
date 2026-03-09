## To run MCP server and test functionality:

Usages

### Step 1: start TensorRT-LLM backend

Create `.extra-llm-api-config.yml` with the following content:

```yaml
reorder_policy_config:
    policy_name: "AgentTree"
    policy_args:
        agent_percentage: 0.5
        agent_types: ["agent_deep_research"]
        agent_inflight_seq_num: 8
```

Run

```bash

trtllm-serve serve /code/llm_models/Qwen3/Qwen3-30B-A3B \
    --max_num_tokens 32768 \
    --kv_cache_free_gpu_memory_fraction 0.8 \
    --extra_llm_api_options .extra-llm-api-config.yml \
    --reasoning_parser qwen3 \
    --tool_parser qwen3

```

### Step 2: Save api key to `.env`

Create `.env` file similar to the `.env.example`

### Step 3: start MCP server

Run

```bash

cd /code/tensorrt_llm/TensorRT-LLM/tensorrt_llm/scaffolding/contrib/ResearchSynthesis/MCP

uv run launch_mcp_server.py


```

### Step 4: check mcp calling functionality

Run
```bash

cd /code/tensorrt_llm/TensorRT-LLM/tensorrt_llm/scaffolding/contrib/ResearchSynthesis/MCP

python run_chat_with_mcp.py


```