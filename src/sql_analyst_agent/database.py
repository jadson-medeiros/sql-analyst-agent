from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .guardrails import DEFAULT_ROW_LIMIT, GuardrailError, validate_sql

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_CELL_CHARS = 200

# why: the authorizer is the last line of defense under the parser. If a statement ever slips past validate_sql (a sqlglot gap, a dialect quirk), SQLite itself refuses every action except reads.
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _ALLOWED_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


class QueryError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "executed_sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": len(self.rows),
            "truncated": self.truncated,
        }


class ReadOnlyDatabase:
    def __init__(self, path: Path | str, timeout_s: float = 2.0, max_rows: int = DEFAULT_ROW_LIMIT):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"database not found: {self.path} (run `uv run sql-analyst seed`)")
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def list_tables(self) -> list[dict[str, Any]]:
        with self._connect(authorize=False) as conn:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )]
            return [{"table": n, "rows": conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]} for n in names]

    def describe_table(self, table: str) -> dict[str, Any]:
        self._check_table(table)
        with self._connect(authorize=False) as conn:
            columns = [
                {"name": r[1], "type": r[2], "not_null": bool(r[3]), "primary_key": bool(r[5])}
                for r in conn.execute(f'PRAGMA table_info("{table}")')
            ]
            foreign_keys = [
                {"column": r[3], "references": f"{r[2]}.{r[4]}"}
                for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')
            ]
        return {"table": table, "columns": columns, "foreign_keys": foreign_keys}

    def sample_rows(self, table: str, n: int = 5) -> QueryResult:
        self._check_table(table)
        return self.run_query(f'SELECT * FROM "{table}" LIMIT {max(1, min(n, 20))}')

    def run_query(self, sql: str) -> QueryResult:
        safe_sql = validate_sql(sql, self.max_rows)
        with self._connect(authorize=True) as conn:
            deadline = time.monotonic() + self.timeout_s
            # why: SQLite has no statement timeout; the progress handler runs every N VM steps and a non-zero return aborts the query with "interrupted".
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            try:
                cursor = conn.execute(safe_sql)
                rows = cursor.fetchmany(self.max_rows)
            except sqlite3.OperationalError as exc:
                if "interrupted" in str(exc):
                    raise QueryError(f"query exceeded the {self.timeout_s}s timeout") from exc
                raise QueryError(str(exc)) from exc
            except sqlite3.DatabaseError as exc:
                raise QueryError(str(exc)) from exc
            columns = [d[0] for d in cursor.description or []]
        return QueryResult(
            sql=safe_sql,
            columns=columns,
            rows=[[_clip(v) for v in row] for row in rows],
            truncated=len(rows) >= self.max_rows,
        )

    def _connect(self, authorize: bool) -> _ClosingConnection:
        # invariant: mode=ro makes SQLite refuse writes at the file level, independent of any SQL validation; query_only blocks writes even to the temp schema.
        conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=self.timeout_s)
        conn.execute("PRAGMA query_only = ON")
        if authorize:
            conn.set_authorizer(_authorizer)
        return _ClosingConnection(conn)

    def _check_table(self, table: str) -> None:
        if not IDENTIFIER.match(table):
            raise GuardrailError(f"invalid table name: {table!r}")
        known = {t["table"] for t in self.list_tables()}
        if table not in known:
            raise QueryError(f"no such table: {table}. Known tables: {', '.join(sorted(known))}")


class _ClosingConnection:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, *exc: object) -> None:
        self._conn.close()


def _authorizer(action: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL_CHARS:
        return value[:MAX_CELL_CHARS] + "..."
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return value
