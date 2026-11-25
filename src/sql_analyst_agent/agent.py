from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from anthropic.types import Message

from .grounding import GroundingReport, check_grounding
from .model import ModelClient, TextCallback
from .toolbox import ToolOutcome
from .usage import UsageTotals

SYSTEM_PROMPT = """You are a data analyst for an outdoor-gear online store. You answer business \
questions by querying the store's SQLite database through the tools you are given.

How to work:
- Discover the schema before writing SQL: list_tables, then describe_table for the tables you need. \
Use sample_rows when you need to see value formats (dates are ISO 'YYYY-MM-DD' text, statuses are \
lowercase strings).
- Compute every number in SQL (COUNT, SUM, AVG, ROUND, GROUP BY, ORDER BY). Do not do arithmetic \
yourself and do not estimate. If you need a ratio or a percentage, compute it in the query.
- Prefer one query that returns exactly the figures the question asks for. Results are capped at 200 \
rows, so aggregate instead of fetching raw rows.
- Revenue means SUM(order_items.quantity * order_items.unit_price) over orders with status \
'completed', unless the question says otherwise. Refund amounts live in refunds.amount.
- If a query fails, read the error, fix the SQL and try again. Do not repeat a query that already \
succeeded; its result is in the conversation.
- The database is read-only. If asked to modify data, say you cannot and answer any read-only part \
of the request.

How to answer:
- Start with the direct answer in one to three sentences, using the numbers exactly as the query \
returned them (you may round, but say the unit: dollars, orders, percent).
- Then cite the SQL that produced those numbers in a ```sql code block, copied exactly as you sent \
it to run_query. If several queries contributed, include each one in its own block.
- Do not mention numbers that did not come from a query result."""

StopReason = Literal["end_turn", "step_cap", "token_budget", "max_tokens", "refusal"]


class Toolbox(Protocol):
    tools: list[dict[str, Any]]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome: ...


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    max_total_tokens: int = 60_000
    max_tokens_per_call: int = 4_096
    max_duplicate_calls: int = 1


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    output: str
    is_error: bool
    guardrail_blocked: bool
    skipped_duplicate: bool = False


@dataclass
class AgentResult:
    question: str
    answer: str | None
    stop_reason: StopReason
    steps: int
    usage: UsageTotals
    model: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    grounding: GroundingReport | None = None
    forced_final_answer: bool = False

    @property
    def guardrail_blocks(self) -> int:
        return sum(1 for c in self.tool_calls if c.guardrail_blocked)

    @property
    def tool_errors(self) -> int:
        return sum(1 for c in self.tool_calls if c.is_error)


ToolEventCallback = Callable[[ToolCallRecord], None]


async def run_agent(
    question: str,
    model: ModelClient,
    toolbox: Toolbox,
    config: AgentConfig = AgentConfig(),
    on_text: TextCallback | None = None,
    on_tool: ToolEventCallback | None = None,
) -> AgentResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    usage = UsageTotals()
    records: list[ToolCallRecord] = []
    seen_calls: dict[str, int] = {}
    answer: str | None = None
    stop: StopReason = "step_cap"
    step = 0
    final_step = False

    while step < config.max_steps:
        if usage.total_tokens >= config.max_total_tokens:
            stop = "token_budget"
            break
        step += 1
        final_step = step == config.max_steps
        params = _request(toolbox.tools, messages, config, force_answer=final_step)
        response: Message = await model.create(on_text=on_text, **params)
        usage.add(response.usage)
        # invariant: the assistant turn is appended verbatim (thinking blocks and signatures included); editing earlier turns invalidates replayed thinking and the cache prefix.
        messages.append({"role": "assistant", "content": [_block_param(b) for b in response.content]})

        if response.stop_reason == "pause_turn":
            continue
        if response.stop_reason in ("end_turn", "stop_sequence"):
            answer, stop = _text_of(response), "end_turn"
            break
        if response.stop_reason in ("max_tokens", "refusal"):
            answer, stop = _text_of(response) or None, response.stop_reason
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            answer, stop = _text_of(response), "end_turn"
            break
        if final_step:
            break

        outcomes = await asyncio.gather(*(_execute(toolbox, b, seen_calls, config) for b in tool_uses))
        results = []
        for block, (outcome, duplicate) in zip(tool_uses, outcomes):
            record = ToolCallRecord(
                step=step,
                name=block.name,
                arguments=dict(block.input),
                output=outcome.text,
                is_error=outcome.is_error,
                guardrail_blocked=outcome.guardrail_blocked,
                skipped_duplicate=duplicate,
            )
            records.append(record)
            if on_tool is not None:
                on_tool(record)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": outcome.text,
                "is_error": outcome.is_error,
            })
        # invariant: every tool_result for one assistant turn goes back in a single user message, in tool_use order; splitting them teaches the model to stop calling tools in parallel.
        messages.append({"role": "user", "content": results})

    result = AgentResult(
        question=question,
        answer=answer,
        stop_reason=stop,
        steps=step,
        usage=usage,
        model=model.model,
        tool_calls=records,
        forced_final_answer=final_step,
    )
    if answer:
        result.grounding = check_grounding(answer, question, *_evidence(records))
    return result


def _request(
    tools: list[dict[str, Any]], messages: list[dict[str, Any]], config: AgentConfig, force_answer: bool
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_tokens": config.max_tokens_per_call,
        # why: tools render before system, so one breakpoint on the system block caches tools + system.
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "tools": tools,
        "messages": messages,
        # why: top-level cache_control moves a second breakpoint to the end of the conversation on every call, so step N reads step N-1's history from cache instead of paying for it again.
        "cache_control": {"type": "ephemeral"},
    }
    if force_answer:
        # why: on the last allowed step the model must answer from what it already has; tool_choice none keeps the tools block (and its cache) but forbids new calls.
        params["tool_choice"] = {"type": "none"}
    return params


async def _execute(
    toolbox: Toolbox, block: Any, seen: dict[str, int], config: AgentConfig
) -> tuple[ToolOutcome, bool]:
    key = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
    seen[key] = seen.get(key, 0) + 1
    if seen[key] > config.max_duplicate_calls:
        return ToolOutcome(
            "This exact call already ran earlier in this conversation; reuse that result instead of "
            "calling it again.",
            is_error=True,
        ), True
    return await toolbox.call(block.name, dict(block.input)), False


def _evidence(records: list[ToolCallRecord]) -> tuple[list[str], list[str]]:
    sql: list[str] = []
    outputs: list[str] = []
    for r in records:
        if r.is_error:
            continue
        outputs.append(r.output)
        if r.name == "run_query":
            sql.append(str(r.arguments.get("sql", "")))
            try:
                sql.append(json.loads(r.output)["executed_sql"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    return sql, outputs


def _text_of(response: Message) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _block_param(block: Any) -> dict[str, Any]:
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if block.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    return block.to_dict()
