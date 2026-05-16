# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Stdio MCP worker: runs MCP servers as subprocesses and routes tool calls.

Used by the evaluation gateway with ScaffoldingLlm; configs come from
mcpbench_config (General-AgentBench mcp-bench commands.json).
"""

import asyncio
import json
import os
from typing import List, Optional

from tensorrt_llm.scaffolding.task import MCPCallTask, TaskStatus
from tensorrt_llm.scaffolding.worker import Worker

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None


class StdioMCPWorker(Worker):
    """MCP worker that connects to MCP servers via stdio (subprocess)."""

    class ToolCall:
        def __init__(self, tool_name: str, args: dict):
            self.tool_name = tool_name
            self.args = args
            self.ready = asyncio.Event()
            self.result: Optional[str] = None

        def set_result(self, result: Optional[str]) -> None:
            self.result = result
            self.ready.set()

    def __init__(self, configs: List[dict], queues: List[asyncio.Queue]) -> None:
        """Initialize with stdio configs and one queue per server.

        Args:
            configs: List of dicts with name, command, args, env, cwd.
            queues: One asyncio.Queue per config (filled by main, consumed by server loop).
        """
        self._configs = configs
        self._queues = queues
        self._background_tasks: List[asyncio.Task] = []

    @classmethod
    def init_with_stdio_configs(cls, configs: List[dict]) -> "StdioMCPWorker":
        """Build worker from list of stdio configs (e.g. from get_stdio_configs_for_worker)."""
        if not configs:
            return cls(configs=[], queues=[])
        queues = [asyncio.Queue() for _ in configs]
        return cls(configs=configs, queues=queues)

    async def _server_loop(self, index: int) -> None:
        config = self._configs[index]
        queue = self._queues[index]
        env = config.get("env") or {}
        full_env = dict(os.environ)
        full_env.update(env)
        params = StdioServerParameters(
            command=config["command"],
            args=config.get("args") or [],
            env=full_env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                tools = {t.name for t in response.tools}
                while True:
                    try:
                        tool_call = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if tool_call is None:
                        return
                    if tool_call.tool_name in tools:
                        try:
                            result = await session.call_tool(
                                tool_call.tool_name, tool_call.args
                            )
                            text = result.content[0].text if result.content else ""
                            tool_call.set_result(text)
                        except Exception:
                            tool_call.set_result(None)
                    else:
                        tool_call.set_result(None)

    async def init_in_asyncio_event_loop(self) -> None:
        """Start one background task per stdio server."""
        if not StdioServerParameters or not stdio_client or not ClientSession:
            raise RuntimeError(
                "MCP stdio client not available. Install mcp with stdio support."
            )
        for i in range(len(self._configs)):
            t = asyncio.create_task(self._server_loop(i))
            self._background_tasks.append(t)

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        tool_name = task.tool_name
        if isinstance(task.args, dict):
            tool_args = task.args
        else:
            tool_args = json.loads(task.args) if task.args else {}
        for index in range(len(self._queues)):
            tool_call = self.ToolCall(tool_name, tool_args)
            self._queues[index].put_nowait(tool_call)
            await tool_call.ready.wait()
            if tool_call.result is not None:
                task.result_str = tool_call.result
                return TaskStatus.SUCCESS
        return TaskStatus.SUCCESS

    async def async_shutdown(self) -> None:
        for q in self._queues:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

    task_handlers = {MCPCallTask: call_handler}
