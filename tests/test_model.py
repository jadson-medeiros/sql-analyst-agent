from __future__ import annotations

import json

import httpx2
from anthropic import AsyncAnthropic

from sql_analyst_agent.agent import run_agent
from sql_analyst_agent.mcp_server import build_server
from sql_analyst_agent.model import AnthropicModel
from sql_analyst_agent.toolbox import McpToolbox


def sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_events(blocks: list[tuple[str, dict, list[dict]]], stop_reason: str) -> list[dict]:
    events = [{
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 900}},
    }]
    for i, (_, start, deltas) in enumerate(blocks):
        events.append({"type": "content_block_start", "index": i, "content_block": start})
        events += [{"type": "content_block_delta", "index": i, "delta": d} for d in deltas]
        events.append({"type": "content_block_stop", "index": i})
    events.append({"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                   "usage": {"output_tokens": 42}})
    events.append({"type": "message_stop"})
    return events


async def test_real_sdk_streaming_path_replays_thinking_and_sends_no_sampling_params(db_path):
    bodies: list[dict] = []
    turns = [
        message_events([
            ("thinking", {"type": "thinking", "thinking": "", "signature": ""},
             [{"type": "thinking_delta", "thinking": ""}, {"type": "signature_delta", "signature": "sig-abc"}]),
            ("tool_use", {"type": "tool_use", "id": "toolu_1", "name": "run_query", "input": {}},
             [{"type": "input_json_delta", "partial_json": '{"sql": "SELECT COUNT(*) FROM refunds"}'}]),
        ], "tool_use"),
        message_events([
            ("text", {"type": "text", "text": ""},
             [{"type": "text_delta", "text": "There are 175 refunds.\n\n"},
              {"type": "text_delta", "text": "```sql\nSELECT COUNT(*) FROM refunds\n```"}]),
        ], "end_turn"),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse(turns.pop(0)))

    client = AsyncAnthropic(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    streamed: list[str] = []
    async with McpToolbox(build_server(db_path)) as toolbox:
        result = await run_agent("How many refunds?", AnthropicModel("claude-haiku-5-5", client), toolbox,
                                 on_text=streamed.append)

    assert result.answer.startswith("There are 175 refunds.")
    assert "".join(streamed) == result.answer
    assert result.grounding.ok, result.grounding.flags()
    assert result.usage.cache_read_input_tokens == 1800
    assert result.usage.cost_usd("claude-haiku-5-5") > 0

    first, second = bodies
    assert first["model"] == "claude-haiku-5-5" and first["stream"] is True
    assert not {"temperature", "top_p", "top_k"} & set(first)
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    replayed = second["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert replayed[1]["input"] == {"sql": "SELECT COUNT(*) FROM refunds"}
    assert second["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"
