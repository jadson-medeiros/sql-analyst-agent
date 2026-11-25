from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from sql_analyst_agent.agent import AgentConfig
from sql_analyst_agent.evaluation import CaseResult, load_dataset, run_eval, summarize
from sql_analyst_agent.fakes import ScriptedModel, oracle_responder
from sql_analyst_agent.seed import DEFAULT_DB, generate
from sql_analyst_agent.toolbox import McpToolbox

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SQL analyst eval set")
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--max-steps", type=int, default=AgentConfig.max_steps)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="no API call: a scripted model replays the reference SQL (checks the harness, not the model)",
    )
    args = parser.parse_args()

    if not args.oracle and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Export it, or pass --oracle to check the harness offline.")
    if not args.db.exists():
        generate(args.db)
    logging.getLogger("mcp").setLevel(logging.WARNING)

    cases = load_dataset(args.dataset)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    asyncio.run(_run(args, cases))


async def _run(args: argparse.Namespace, cases) -> None:
    if args.oracle:
        def model_for(case):
            return ScriptedModel(oracle_responder(case.reference_sql), model="oracle")
    else:
        from sql_analyst_agent.model import AnthropicModel

        shared = AnthropicModel()

        def model_for(case):
            return shared

    config = AgentConfig(max_steps=args.max_steps)
    async with McpToolbox.stdio(args.db) as toolbox:
        results = await run_eval(cases, args.db, model_for, toolbox, config, on_case=_print_case)

    summary = summarize(results)
    print("\n" + json.dumps(summary, indent=2))
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{summary['model']}.json"
    out.write_text(json.dumps({"summary": summary, "cases": [_case_json(r) for r in results]}, indent=2, default=str))
    print(f"wrote {out}")


def _print_case(r: CaseResult) -> None:
    res = r.result
    flags = "; ".join(res.grounding.flags()) if res.grounding else "no answer"
    print(
        f"{'PASS' if r.correct else 'FAIL'} {r.case.id} steps={res.steps} tools={len(res.tool_calls)} "
        f"tokens={res.usage.total_tokens} blocks={res.guardrail_blocks} stop={res.stop_reason} "
        f"expected={r.expected} grounding={flags or 'ok'}"
    )


def _case_json(r: CaseResult) -> dict:
    res = r.result
    return {
        "id": r.case.id,
        "question": r.case.question,
        "expected": r.expected,
        "correct": r.correct,
        "answer": res.answer,
        "stop_reason": res.stop_reason,
        "steps": res.steps,
        "usage": res.usage.per_call,
        "tool_calls": [
            {"step": c.step, "name": c.name, "arguments": c.arguments, "is_error": c.is_error,
             "guardrail_blocked": c.guardrail_blocked, "output": c.output[:500]}
            for c in res.tool_calls
        ],
        "grounding_flags": res.grounding.flags() if res.grounding else None,
    }


if __name__ == "__main__":
    main()
