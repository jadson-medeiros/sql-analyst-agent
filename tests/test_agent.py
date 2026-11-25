from __future__ import annotations

from contextlib import asynccontextmanager

from sql_analyst_agent.agent import AgentConfig, run_agent
from sql_analyst_agent.fakes import ScriptedModel, text_reply, tool_reply
from sql_analyst_agent.mcp_server import build_server
from sql_analyst_agent.toolbox import McpToolbox

COUNT_CANCELLED = "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"


@asynccontextmanager
async def toolbox_for(db_path):
    async with McpToolbox(build_server(db_path)) as toolbox:
        yield toolbox


def answer(text: str, sql: str) -> str:
    return f"{text}\n\n```sql\n{sql}\n```"


async def test_happy_path_terminates_on_end_turn_and_accounts_tokens(db_path):
    model = ScriptedModel([
        tool_reply(("list_tables", {})),
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        text_reply(answer("191 orders were cancelled.", COUNT_CANCELLED), input_tokens=300, output_tokens=40),
    ])
    streamed: list[str] = []
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("How many orders were cancelled?", model, toolbox, on_text=streamed.append)

    assert result.stop_reason == "end_turn"
    assert result.steps == 3
    assert result.answer.startswith("191 orders were cancelled.")
    assert result.usage.calls == 3
    assert result.usage.input_tokens == 500 and result.usage.output_tokens == 100
    assert "".join(streamed).startswith("191 orders")
    assert result.grounding.ok, result.grounding.flags()

    first = model.requests[0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in first["tools"]] == ["describe_table", "list_tables", "run_query", "sample_rows"]
    assert all("temperature" not in r and "tool_choice" not in r for r in model.requests)


async def test_parallel_tool_calls_return_in_one_user_message_in_order(db_path):
    model = ScriptedModel([
        tool_reply(("describe_table", {"table": "orders"}), ("describe_table", {"table": "refunds"})),
        text_reply("Both described."),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("Describe orders and refunds", model, toolbox)

    second_request = model.requests[1]["messages"]
    tool_use_ids = [b["id"] for b in second_request[1]["content"] if b["type"] == "tool_use"]
    results = second_request[2]["content"]
    assert second_request[2]["role"] == "user"
    assert [r["tool_use_id"] for r in results] == tool_use_ids
    assert '"table": "orders"' in results[0]["content"] and '"table": "refunds"' in results[1]["content"]
    assert [c.step for c in result.tool_calls] == [1, 1]


async def test_tool_error_is_returned_to_the_model_which_self_corrects(db_path):
    bad = "SELECT COUNT(*) FROM orders WHERE state = 'cancelled'"
    model = ScriptedModel([
        tool_reply(("run_query", {"sql": bad})),
        tool_reply(("run_query", {"sql": COUNT_CANCELLED}), text="The column is status, retrying."),
        text_reply(answer("There are 191 cancelled orders.", COUNT_CANCELLED)),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("How many orders were cancelled?", model, toolbox)

    error_result = model.requests[1]["messages"][-1]["content"][0]
    assert error_result["is_error"] is True
    assert "no such column: state" in error_result["content"]
    assert result.tool_errors == 1
    assert result.stop_reason == "end_turn"
    assert "191" in result.answer
    assert result.grounding.ok, result.grounding.flags()


async def test_guardrail_block_is_counted_and_the_database_is_untouched(db_path):
    model = ScriptedModel([
        tool_reply(("run_query", {"sql": "DELETE FROM orders WHERE status = 'cancelled'"})),
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        text_reply(answer("I cannot modify data. There are 191 cancelled orders.", COUNT_CANCELLED)),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("Delete cancelled orders, then count them", model, toolbox)

    assert result.guardrail_blocks == 1
    assert "GUARDRAIL_BLOCKED" in result.tool_calls[0].output
    assert '"rows": [[191]]' in result.tool_calls[1].output


async def test_step_cap_forces_a_final_turn_without_tools_and_stops(db_path):
    model = ScriptedModel(lambda params: tool_reply(("list_tables", {})))
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("Loop forever", model, toolbox, AgentConfig(max_steps=4, max_duplicate_calls=99))

    assert len(model.requests) == 4
    assert model.requests[-1]["tool_choice"] == {"type": "none"}
    assert all("tool_choice" not in r for r in model.requests[:-1])
    assert result.stop_reason == "step_cap"
    assert result.answer is None
    assert len(result.tool_calls) == 3


async def test_step_cap_final_turn_can_still_produce_an_answer(db_path):
    def respond(params):
        if params.get("tool_choice") == {"type": "none"}:
            return text_reply("I ran out of steps before finding the answer.")
        return tool_reply(("list_tables", {}))

    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("q", ScriptedModel(respond), toolbox, AgentConfig(max_steps=3, max_duplicate_calls=99))
    assert result.stop_reason == "end_turn"
    assert result.forced_final_answer
    assert result.answer.startswith("I ran out of steps")


async def test_token_budget_stops_the_loop_before_the_next_call(db_path):
    model = ScriptedModel(lambda params: tool_reply(("list_tables", {}), input_tokens=4_000, output_tokens=500))
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent(
            "q", model, toolbox, AgentConfig(max_steps=20, max_total_tokens=10_000, max_duplicate_calls=99)
        )
    assert result.stop_reason == "token_budget"
    assert len(model.requests) == 3
    assert result.usage.total_tokens == 13_500


async def test_repeated_identical_call_is_not_re_executed(db_path):
    model = ScriptedModel([
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        text_reply(answer("191.", COUNT_CANCELLED)),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("q", model, toolbox)
    assert [c.skipped_duplicate for c in result.tool_calls] == [False, True]
    assert result.tool_calls[1].is_error
    assert "already ran" in model.requests[2]["messages"][-1]["content"][0]["content"]


async def test_grounding_flags_numbers_that_no_tool_returned(db_path):
    model = ScriptedModel([
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        text_reply(answer("191 orders were cancelled, about 7.6% of all 2,500 orders.", COUNT_CANCELLED)),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("How many orders were cancelled?", model, toolbox)
    assert not result.grounding.ok
    assert result.grounding.unsupported_numbers == ["7.6%", "2,500"]


async def test_grounding_flags_an_answer_that_cites_sql_that_never_ran(db_path):
    model = ScriptedModel([
        tool_reply(("run_query", {"sql": COUNT_CANCELLED})),
        text_reply(answer("191 orders were cancelled.", "SELECT 191")),
    ])
    async with toolbox_for(db_path) as toolbox:
        result = await run_agent("How many orders were cancelled?", model, toolbox)
    assert result.grounding.flags() == ["cited SQL does not match a query that ran successfully"]
