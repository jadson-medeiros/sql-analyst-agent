from __future__ import annotations

from pathlib import Path

import pytest

from sql_analyst_agent.evaluation import EvalCase, expected_values, is_correct, load_dataset, run_eval, summarize
from sql_analyst_agent.fakes import ScriptedModel, oracle_responder, text_reply
from sql_analyst_agent.mcp_server import build_server
from sql_analyst_agent.toolbox import McpToolbox

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"


def case(match: str) -> EvalCase:
    return EvalCase(id="t", question="q", reference_sql="SELECT 1", match=match)


@pytest.mark.parametrize(
    "answer, ok",
    [
        ("Revenue was $221,312.24.", True),
        ("Revenue was about $221,312.", True),
        ("Revenue was $221,312.25.", False),
        ("Revenue was $221,000.", False),
        ("No idea.", False),
    ],
)
def test_number_matching_respects_displayed_precision(answer, ok):
    assert is_correct(answer, case("number"), [221312.24]) is ok


def test_number_inside_the_sql_block_does_not_count():
    assert not is_correct("I could not tell.\n```sql\nSELECT 191\n```", case("number"), [191])


def test_short_codes_match_as_whole_words_only():
    assert is_correct("The US has the most customers.", case("text_all"), ["US"])
    assert not is_correct("Most customers trust us.", case("text_all"), ["US"])
    assert is_correct("Most refunds were for wrong size.", case("text_all"), ["wrong_size"])


def test_dataset_has_enough_cases_and_every_reference_query_returns_a_value(db_path):
    cases = load_dataset(DATASET)
    assert len(cases) >= 15
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        values = expected_values(db_path, c)
        assert values and all(v is not None for v in values), c.id


async def test_eval_runner_scores_an_oracle_and_a_wrong_model(db_path):
    cases = load_dataset(DATASET)[:3]
    async with McpToolbox(build_server(db_path)) as toolbox:
        oracle = await run_eval(cases, db_path, lambda c: ScriptedModel(oracle_responder(c.reference_sql)), toolbox)
        wrong = await run_eval(cases, db_path, lambda c: ScriptedModel([text_reply("It is 42.")]), toolbox)

    good, bad = summarize(oracle), summarize(wrong)
    assert good["accuracy"] == 1.0 and good["avg_steps"] == 2.0 and good["grounding_flagged"] == 0
    assert bad["accuracy"] == 0.0 and bad["grounding_flagged"] == 3
