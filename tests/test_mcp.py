from __future__ import annotations

import asyncio
import json

from mcp.server.mcpserver import MCPServer

from sql_analyst_agent.mcp_server import build_server
from sql_analyst_agent.toolbox import McpToolbox

EXPECTED_TOOLS = ["describe_table", "list_tables", "run_query", "sample_rows"]


async def test_in_process_discovery_maps_mcp_tools_to_anthropic_definitions(db_path):
    async with McpToolbox(build_server(db_path)) as toolbox:
        assert [t["name"] for t in toolbox.tools] == EXPECTED_TOOLS
        run_query = next(t for t in toolbox.tools if t["name"] == "run_query")
        assert set(run_query) == {"name", "description", "input_schema"}
        assert run_query["input_schema"]["required"] == ["sql"]
        assert "read-only" in run_query["description"]


async def test_stdio_server_is_discovered_and_answers_queries(db_path):
    async with McpToolbox.stdio(db_path) as toolbox:
        assert [t["name"] for t in toolbox.tools] == EXPECTED_TOOLS
        outcome = await toolbox.call("run_query", {"sql": "SELECT COUNT(*) AS n FROM orders"})
    assert not outcome.is_error
    payload = json.loads(outcome.text)
    assert payload["rows"] == [[2500]]
    assert payload["executed_sql"] == "SELECT COUNT(*) AS n FROM orders LIMIT 200"


async def test_guardrail_rejection_comes_back_as_a_flagged_tool_error(db_path):
    async with McpToolbox(build_server(db_path)) as toolbox:
        blocked = await toolbox.call("run_query", {"sql": "DROP TABLE orders"})
        sql_error = await toolbox.call("run_query", {"sql": "SELECT nope FROM orders"})
        still_there = await toolbox.call("run_query", {"sql": "SELECT COUNT(*) FROM orders"})
    assert blocked.is_error and blocked.guardrail_blocked
    assert sql_error.is_error and not sql_error.guardrail_blocked
    assert "no such column: nope" in sql_error.text
    assert json.loads(still_there.text)["rows"] == [[2500]]


async def test_slow_tool_is_cut_off_by_the_client_timeout():
    server = MCPServer(name="slow")

    @server.tool(description="sleeps")
    async def slow() -> str:
        await asyncio.sleep(5)
        return "done"

    async with McpToolbox(server, tool_timeout_s=0.2) as toolbox:
        outcome = await toolbox.call("slow", {})
    assert outcome.is_error
    assert "timed out after 0.2s" in outcome.text


async def test_unknown_tool_is_an_error_not_a_crash(db_path):
    async with McpToolbox(build_server(db_path)) as toolbox:
        outcome = await toolbox.call("delete_everything", {})
    assert outcome.is_error and "unknown tool" in outcome.text
