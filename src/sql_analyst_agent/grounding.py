from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot.errors import SqlglotError

SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
ANY_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
DATE_LIKE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
NUMBER = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?")
# why: small integers in prose are usually structure ("top 3", "2 categories"), not data claims; checking them produces false flags. A known gap: a wrong small count goes unflagged.
IGNORE_UP_TO = 10


@dataclass
class GroundingReport:
    cited_sql: list[str] = field(default_factory=list)
    cited_sql_matches_executed: bool = False
    numbers_checked: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.cited_sql_matches_executed and not self.unsupported_numbers

    def flags(self) -> list[str]:
        out = []
        if not self.cited_sql:
            out.append("answer does not cite any SQL")
        elif not self.cited_sql_matches_executed:
            out.append("cited SQL does not match a query that ran successfully")
        if self.unsupported_numbers:
            out.append(f"numbers not found in tool results: {', '.join(self.unsupported_numbers)}")
        return out


def check_grounding(answer: str, question: str, executed_sql: list[str], tool_outputs: list[str]) -> GroundingReport:
    report = GroundingReport(cited_sql=[s.strip() for s in SQL_BLOCK.findall(answer)])
    executed = {_normalize_sql(s) for s in executed_sql}
    report.cited_sql_matches_executed = bool(report.cited_sql) and all(
        _normalize_sql(s) in executed for s in report.cited_sql
    )

    observed = _numbers_in_outputs(tool_outputs)
    context = " ".join([question, *executed_sql])
    excluded = {_parse_number(m)[0] for m in NUMBER.findall(DATE_LIKE.sub(" ", context))}
    prose = DATE_LIKE.sub(" ", ANY_CODE_BLOCK.sub(" ", answer))
    for token in NUMBER.findall(prose):
        value, decimals, is_percent = _parse_number(token)
        if value in excluded or (decimals == 0 and abs(value) <= IGNORE_UP_TO):
            continue
        report.numbers_checked.append(token)
        if not any(_matches(value, decimals, is_percent, o) for o in observed):
            report.unsupported_numbers.append(token)
    return report


def _matches(value: float, decimals: int, is_percent: bool, observed: float) -> bool:
    candidates = [observed, observed * 100] if is_percent else [observed]
    # invariant: the answer may round a result, never invent precision; 1,234.6 matches 1234.5678 but 1,234.57 does not match 1234.5.
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(c - value) <= tolerance for c in candidates)


def _parse_number(token: str) -> tuple[float, int, bool]:
    is_percent = token.endswith("%")
    raw = token.replace("$", "").replace(",", "").rstrip("%")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return float(raw), decimals, is_percent


def _numbers_in_outputs(outputs: list[str]) -> list[float]:
    numbers: list[float] = []
    for text in outputs:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        _collect(payload, numbers)
    return numbers


def _collect(node: object, into: list[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        into.append(float(node))
    elif isinstance(node, str):
        # why: numbers inside returned text values ("Protein Bars 12-Pack") are grounded too.
        into.extend(_parse_number(t)[0] for t in NUMBER.findall(DATE_LIKE.sub(" ", node)))
    elif isinstance(node, dict):
        for key, value in node.items():
            if key != "executed_sql":
                _collect(value, into)
    elif isinstance(node, list):
        for value in node:
            _collect(value, into)


def _normalize_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite").lower()
    except SqlglotError:
        return " ".join(sql.lower().rstrip(";").split())
