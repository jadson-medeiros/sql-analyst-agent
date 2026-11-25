from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer
from mcp_types import TextContent

from .mcp_server import GUARDRAIL_PREFIX


@dataclass(frozen=True)
class ToolOutcome:
    text: str
    is_error: bool
    guardrail_blocked: bool = False


class McpToolbox:
    def __init__(self, server: MCPServer | StdioServerParameters, tool_timeout_s: float = 10.0):
        self._client = Client(server)
        self.tool_timeout_s = tool_timeout_s
        self.tools: list[dict[str, Any]] = []

    @classmethod
    def stdio(cls, db_path: Path | str, tool_timeout_s: float = 10.0) -> McpToolbox:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "sql_analyst_agent.mcp_server", "--db", str(db_path)],
        )
        return cls(params, tool_timeout_s)

    async def __aenter__(self) -> McpToolbox:
        await self._client.__aenter__()
        listing = await self._client.list_tools()
        # invariant: tools are sorted by name so the tool block is byte-identical across runs; any reordering would invalidate the prompt cache prefix.
        self.tools = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in sorted(listing.tools, key=lambda t: t.name)
        ]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.__aexit__(*exc)

    @property
    def names(self) -> set[str]:
        return {t["name"] for t in self.tools}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if name not in self.names:
            return ToolOutcome(f"unknown tool {name!r}; available: {sorted(self.names)}", is_error=True)
        try:
            result = await asyncio.wait_for(self._client.call_tool(name, arguments), self.tool_timeout_s)
        except TimeoutError:
            return ToolOutcome(f"tool {name} timed out after {self.tool_timeout_s}s", is_error=True)
        except Exception as exc:
            # why: a transport or protocol failure must reach the model as a tool error, not crash the loop; the model can retry or answer with what it has.
            return ToolOutcome(f"tool {name} failed: {type(exc).__name__}: {exc}", is_error=True)
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        return ToolOutcome(
            text=text,
            is_error=result.is_error,
            guardrail_blocked=result.is_error and f"{GUARDRAIL_PREFIX}:" in text,
        )
