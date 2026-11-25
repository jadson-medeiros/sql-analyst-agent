from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from .database import QueryError, ReadOnlyDatabase
from .guardrails import GuardrailError
from .seed import DEFAULT_DB

GUARDRAIL_PREFIX = "GUARDRAIL_BLOCKED"


def build_server(db_path: Path | str, query_timeout_s: float = 2.0) -> MCPServer:
    db = ReadOnlyDatabase(db_path, timeout_s=query_timeout_s)
    server = MCPServer(
        name="sql-analyst-db",
        instructions="Read-only access to the store database (customers, products, orders, order_items, refunds).",
    )

    @server.tool(
        description="List every table in the store database with its row count. Call this first.",
        structured_output=False,
    )
    def list_tables() -> str:
        return _dump(db.list_tables())

    @server.tool(
        description="Show the columns (name, type, primary key) and foreign keys of one table.",
        structured_output=False,
    )
    def describe_table(table: Annotated[str, Field(description="Exact table name from list_tables")]) -> str:
        return _dump(_guarded(lambda: db.describe_table(table)))

    @server.tool(
        description="Return up to n example rows from a table, to see real value formats (dates, statuses).",
        structured_output=False,
    )
    def sample_rows(
        table: Annotated[str, Field(description="Exact table name from list_tables")],
        n: Annotated[int, Field(description="Number of rows, 1-20", ge=1, le=20)] = 5,
    ) -> str:
        return _dump(_guarded(lambda: db.sample_rows(table, n).as_dict()))

    @server.tool(
        description=(
            "Run ONE read-only SQLite SELECT and return columns and rows. Writes, PRAGMA, ATTACH and "
            "multiple statements are rejected. A LIMIT of at most 200 is enforced, so aggregate in SQL "
            "(COUNT, SUM, AVG, GROUP BY) instead of fetching raw rows. The response includes executed_sql, "
            "the exact statement that ran."
        ),
        structured_output=False,
    )
    def run_query(sql: Annotated[str, Field(description="A single SQLite SELECT statement")]) -> str:
        return _dump(_guarded(lambda: db.run_query(sql).as_dict()))

    return server


def _guarded(fn):
    try:
        return fn()
    except GuardrailError as exc:
        # invariant: guardrail rejections carry a fixed prefix so the client can count them separately from ordinary SQL errors.
        raise ToolError(f"{GUARDRAIL_PREFIX}: {exc}") from exc
    except QueryError as exc:
        raise ToolError(f"SQL error: {exc}") from exc


def _dump(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server (stdio) exposing the store database")
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("SQL_AGENT_DB", DEFAULT_DB)))
    parser.add_argument("--query-timeout", type=float, default=2.0)
    args = parser.parse_args()
    build_server(args.db, args.query_timeout).run("stdio")


if __name__ == "__main__":
    main()
