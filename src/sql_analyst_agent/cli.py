from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .agent import AgentConfig, ToolCallRecord, run_agent
from .model import AnthropicModel
from .seed import DEFAULT_DB, generate
from .toolbox import McpToolbox


def main() -> None:
    parser = argparse.ArgumentParser(prog="sql-analyst")
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser("seed", help="generate the sample store database")
    seed.add_argument("--db", type=Path, default=DEFAULT_DB)
    seed.add_argument("--seed", type=int, default=42)
    ask = sub.add_parser("ask", help="ask a business question")
    ask.add_argument("question")
    ask.add_argument("--db", type=Path, default=Path(os.environ.get("SQL_AGENT_DB", DEFAULT_DB)))
    ask.add_argument("--max-steps", type=int, default=AgentConfig.max_steps)
    ask.add_argument("--max-total-tokens", type=int, default=AgentConfig.max_total_tokens)
    ask.add_argument("--tool-timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.command == "seed":
        print(f"wrote {generate(args.db, seed=args.seed)}")
        return
    if not args.db.exists():
        generate(args.db)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    try:
        asyncio.run(_ask(args))
    except Exception as exc:
        root = _root_cause(exc)
        sys.exit(f"error: {type(root).__name__}: {root}")


async def _ask(args: argparse.Namespace) -> None:
    model = AnthropicModel()
    config = AgentConfig(max_steps=args.max_steps, max_total_tokens=args.max_total_tokens)
    async with McpToolbox.stdio(args.db, tool_timeout_s=args.tool_timeout) as toolbox:
        print(f"[mcp] discovered tools: {', '.join(t['name'] for t in toolbox.tools)}", file=sys.stderr)
        result = await run_agent(
            args.question,
            model,
            toolbox,
            config,
            on_text=lambda chunk: print(chunk, end="", flush=True),
            on_tool=_print_tool,
        )
    print()
    print("-" * 60, file=sys.stderr)
    print(f"stop={result.stop_reason} steps={result.steps} tool_calls={len(result.tool_calls)} "
          f"tool_errors={result.tool_errors} guardrail_blocks={result.guardrail_blocks}", file=sys.stderr)
    print(f"model={result.model} {result.usage.summary(result.model)}", file=sys.stderr)
    if result.grounding is not None:
        flags = result.grounding.flags()
        status = "OK" if not flags else "FLAGGED: " + "; ".join(flags)
        print(f"grounding: {status} (checked {len(result.grounding.numbers_checked)} numbers)", file=sys.stderr)


def _root_cause(exc: BaseException) -> BaseException:
    # why: failures inside the MCP client's task group arrive wrapped in ExceptionGroups.
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _print_tool(record: ToolCallRecord) -> None:
    args = record.arguments.get("sql") or record.arguments.get("table") or ""
    status = "BLOCKED" if record.guardrail_blocked else ("ERROR" if record.is_error else "ok")
    preview = record.output if record.is_error else record.output[:160]
    print(f"\n[tool step={record.step}] {record.name}({args}) -> {status}: {preview}", file=sys.stderr)


if __name__ == "__main__":
    main()
