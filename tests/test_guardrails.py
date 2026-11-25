from __future__ import annotations

import sqlite3

import pytest

from sql_analyst_agent import database
from sql_analyst_agent.database import QueryError, ReadOnlyDatabase
from sql_analyst_agent.guardrails import GuardrailError, validate_sql


def test_select_gets_a_forced_limit():
    assert validate_sql("SELECT id FROM orders", max_rows=50) == "SELECT id FROM orders LIMIT 50"


def test_cte_and_union_are_allowed():
    assert validate_sql("WITH t AS (SELECT 1 AS x) SELECT x FROM t").endswith("LIMIT 200")
    assert validate_sql("SELECT 1 UNION SELECT 2").endswith("LIMIT 200")


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT * FROM orders LIMIT 10", "LIMIT 10"),
        ("SELECT * FROM orders LIMIT 100000", "LIMIT 200"),
        ("SELECT * FROM orders LIMIT -1", "LIMIT 200"),
        ("SELECT * FROM orders LIMIT (SELECT 1000000)", "LIMIT 200"),
    ],
)
def test_limit_cannot_exceed_the_cap(sql, expected):
    assert validate_sql(sql).endswith(expected)


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders VALUES (9999, 1, '2025-01-01', 'completed')",
        "UPDATE orders SET status = 'cancelled'",
        "DELETE FROM refunds",
        "DROP TABLE orders",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE orders ADD COLUMN y INT",
        "PRAGMA table_info(orders)",
        "PRAGMA writable_schema = ON",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "VACUUM",
        "BEGIN TRANSACTION",
    ],
)
def test_non_select_statements_are_blocked(sql):
    with pytest.raises(GuardrailError):
        validate_sql(sql)


def test_multiple_statements_are_blocked():
    with pytest.raises(GuardrailError, match="exactly one statement"):
        validate_sql("SELECT 1; DROP TABLE orders")


def test_write_hidden_in_a_cte_is_blocked():
    with pytest.raises(GuardrailError, match="DELETE"):
        validate_sql("WITH d AS (DELETE FROM refunds RETURNING *) SELECT * FROM d")


def test_dangerous_functions_are_blocked():
    with pytest.raises(GuardrailError, match="load_extension"):
        validate_sql("SELECT load_extension('/tmp/x.so')")


def test_unparseable_sql_is_rejected_with_a_readable_error():
    with pytest.raises(GuardrailError, match="could not parse"):
        validate_sql("SELEC FROM WHERE (")


def test_connection_is_read_only_at_the_file_level(db_path):
    db = ReadOnlyDatabase(db_path)
    with db._connect(authorize=False) as conn, pytest.raises(sqlite3.OperationalError, match="readonly|query_only"):
        conn.execute("DELETE FROM refunds")


def test_authorizer_denies_writes_even_if_the_parser_is_bypassed(db_path, monkeypatch):
    monkeypatch.setattr(database, "validate_sql", lambda sql, max_rows: sql)
    db = ReadOnlyDatabase(db_path)
    with pytest.raises(QueryError, match="not authorized"):
        db.run_query("DELETE FROM refunds")
    with pytest.raises(QueryError, match="not authorized"):
        db.run_query("PRAGMA query_only = OFF")
    assert db.run_query("SELECT COUNT(*) FROM refunds").rows[0][0] > 0


def test_runaway_query_hits_the_timeout(db_path):
    db = ReadOnlyDatabase(db_path, timeout_s=0.2)
    with pytest.raises(QueryError, match="timeout"):
        db.run_query("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c")


def test_table_name_is_validated_before_interpolation(db_path):
    db = ReadOnlyDatabase(db_path)
    with pytest.raises(GuardrailError):
        db.describe_table('orders"; DROP TABLE orders; --')
    with pytest.raises(QueryError, match="no such table"):
        db.describe_table("payments")


def test_results_are_capped_and_marked_truncated(db_path):
    result = ReadOnlyDatabase(db_path, max_rows=25).run_query("SELECT * FROM orders")
    assert len(result.rows) == 25
    assert result.truncated
    assert result.sql.endswith("LIMIT 25")
