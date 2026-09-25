"""
Synchronous MCP client for LangGraph nodes.

The MCP SDK is async and its stdio client must be opened and closed in the
same asyncio task. LangGraph nodes here are plain sync functions, so one
background thread owns an event loop, and a single long-lived task holds the
connection open; call() hands each tool request to that loop and blocks for
the JSON-RPC reply.
"""

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, List

import anyio
from mcp import Client, MCPError, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).with_name("mcp_server.py")
SERVER_LOG = Path(__file__).with_name("mcp_server.log")


class MCPToolError(RuntimeError):
    pass


class MCPConnectionLost(ConnectionError):
    """The server process went away; this client is dead and must be replaced."""


class WorkspaceMCP:
    def __init__(self, workspace: Path, timeout_s: float = 180):
        self.timeout_s = timeout_s
        self.dead = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(SERVER_SCRIPT), "--workspace", str(workspace)],
            env=dict(os.environ),
        )
        # Server stderr (logs, crash tracebacks) goes to a file so it can't scribble over the terminal UI.
        self._errlog = open(SERVER_LOG, "a", encoding="utf-8")
        self._runner = asyncio.run_coroutine_threadsafe(self._hold_open(params), self._loop)
        self._ready.result(timeout=60)

    async def _hold_open(self, params: StdioServerParameters) -> None:
        self._closing = asyncio.Event()
        try:
            async with Client(stdio_client(params, errlog=self._errlog)) as client:
                self._client = client
                self._ready.set_result(True)
                await self._closing.wait()
        except BaseException as e:
            if not self._ready.done():
                self._ready.set_exception(e)
            raise

    def tool_names(self) -> List[str]:
        result = asyncio.run_coroutine_threadsafe(self._client.list_tools(), self._loop).result(self.timeout_s)
        return [t.name for t in result.tools]

    def call(self, tool: str, **arguments: Any) -> Any:
        """Call an MCP tool and return its structured result (dict/list)."""
        if self.dead:
            raise MCPConnectionLost("MCP server is gone")
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(tool, arguments), self._loop)
        try:
            result = fut.result(timeout=self.timeout_s)
        except (MCPError, anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream) as e:
            if isinstance(e, MCPError) and "closed" not in str(e).lower():
                raise
            self.dead = True
            raise MCPConnectionLost(f"MCP server connection lost during {tool}: {e}") from e
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            raise MCPToolError(f"{tool}: {text}")
        data = result.structured_content if result.structured_content is not None else json.loads(text)
        # Non-dict return values (e.g. list_files) come back wrapped as {"result": ...}.
        if isinstance(data, dict) and set(data) == {"result"}:
            return data["result"]
        return data

    def close(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._closing.set)
        try:
            self._runner.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._errlog.close()
