from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DEFAULT_ROW_LIMIT = 200

FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

# why: SQLite functions that touch the filesystem or load native code. load_extension is already disabled on Python's sqlite3 connections, but a parse-level block gives the model a readable error.
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class GuardrailError(ValueError):
    pass


def validate_sql(sql: str, max_rows: int = DEFAULT_ROW_LIMIT) -> str:
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except ParseError as exc:
        raise GuardrailError(f"could not parse SQL: {_first_line(str(exc))}") from exc

    if len(statements) != 1:
        raise GuardrailError(f"exactly one statement is allowed, got {len(statements)}")
    statement = statements[0]

    if not isinstance(statement, (exp.Select, exp.SetOperation)):
        raise GuardrailError(f"only SELECT queries are allowed, got {statement.key.upper()}")

    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise GuardrailError(f"{node.key.upper()} is not allowed inside a query")
        if isinstance(node, (exp.Anonymous, exp.Func)) and _function_name(node) in FORBIDDEN_FUNCTIONS:
            raise GuardrailError(f"function {_function_name(node)}() is not allowed")

    return _enforce_limit(statement, max_rows).sql(dialect="sqlite")


def _enforce_limit(statement: exp.Query, max_rows: int) -> exp.Query:
    limit = statement.args.get("limit")
    if limit is None:
        return statement.limit(max_rows)
    value = limit.expression
    # invariant: a LIMIT that is not a plain integer literal (a subquery, an expression, -1) is replaced, so the cap cannot be bypassed with LIMIT (SELECT 1e9) or LIMIT -1.
    if isinstance(value, exp.Literal) and value.is_int and 0 <= int(value.this) <= max_rows:
        return statement
    return statement.limit(max_rows)


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return node.sql_name().lower()


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else text
