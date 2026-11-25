from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import AgentConfig, AgentResult, Toolbox, run_agent
from .grounding import NUMBER, _matches, _parse_number
from .model import ModelClient


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    reference_sql: str
    match: str
    adversarial: bool = False


@dataclass
class CaseResult:
    case: EvalCase
    expected: list[Any]
    correct: bool
    result: AgentResult


def load_dataset(path: Path) -> list[EvalCase]:
    return [EvalCase(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def expected_values(db_path: Path, case: EvalCase) -> list[Any]:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(case.reference_sql).fetchall()
    finally:
        conn.close()
    if case.match == "number_all":
        return list(rows[0])
    if case.match == "number":
        return [rows[0][0]]
    return [r[0] for r in rows]


def is_correct(answer: str | None, case: EvalCase, expected: list[Any]) -> bool:
    if not answer:
        return False
    prose = re.sub(r"```.*?```", " ", answer, flags=re.DOTALL)
    if case.match in ("number", "number_all"):
        numbers = [_parse_number(t) for t in NUMBER.findall(prose)]
        return all(
            any(_matches(value, decimals, pct, float(e)) for value, decimals, pct in numbers) for e in expected
        )
    return all(_contains_term(prose, str(e)) for e in expected)


def _contains_term(text: str, term: str) -> bool:
    # why: short codes like "US" must match as a word and case-sensitively, or "us" in any sentence would count as the right country.
    flags = 0 if len(term) <= 3 else re.IGNORECASE
    variants = {term, term.replace("_", " ")}
    return any(re.search(rf"(?<![\w-]){re.escape(v)}(?![\w-])", text, flags) for v in variants)


async def run_eval(
    cases: list[EvalCase],
    db_path: Path,
    model_for: Callable[[EvalCase], ModelClient],
    toolbox: Toolbox,
    config: AgentConfig = AgentConfig(),
    on_case: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    out = []
    for case in cases:
        expected = expected_values(db_path, case)
        result = await run_agent(case.question, model_for(case), toolbox, config)
        case_result = CaseResult(case, expected, is_correct(result.answer, case, expected), result)
        out.append(case_result)
        if on_case is not None:
            on_case(case_result)
    return out


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results) or 1
    model = results[0].result.model if results else ""
    costs = [r.result.usage.cost_usd(r.result.model) for r in results]
    return {
        "model": model,
        "cases": len(results),
        "accuracy": round(sum(r.correct for r in results) / n, 3),
        "correct": sum(r.correct for r in results),
        "avg_steps": round(sum(r.result.steps for r in results) / n, 2),
        "avg_tool_calls": round(sum(len(r.result.tool_calls) for r in results) / n, 2),
        "avg_total_tokens": round(sum(r.result.usage.total_tokens for r in results) / n, 1),
        "cache_read_tokens": sum(r.result.usage.cache_read_input_tokens for r in results),
        "cache_write_tokens": sum(r.result.usage.cache_creation_input_tokens for r in results),
        "total_cost_usd": round(sum(c for c in costs if c is not None), 6) if any(costs) else None,
        "guardrail_blocks": sum(r.result.guardrail_blocks for r in results),
        "tool_errors": sum(r.result.tool_errors for r in results),
        "grounding_flagged": sum(1 for r in results if r.result.grounding and not r.result.grounding.ok),
        "no_answer": sum(1 for r in results if not r.result.answer),
        "stop_reasons": _count(r.result.stop_reason for r in results),
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
