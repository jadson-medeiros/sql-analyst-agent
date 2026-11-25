from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from .model import TextCallback

_ids = itertools.count(1)

Responder = Callable[[dict[str, Any]], Message]


def text_reply(text: str, input_tokens: int = 100, output_tokens: int = 50) -> Message:
    return _message([TextBlock(type="text", text=text)], "end_turn", input_tokens, output_tokens)


def tool_reply(*calls: tuple[str, dict[str, Any]], text: str | None = None, input_tokens: int = 100,
               output_tokens: int = 30) -> Message:
    blocks: list[Any] = [TextBlock(type="text", text=text)] if text else []
    blocks += [ToolUseBlock(type="tool_use", id=f"toolu_{next(_ids):04d}", name=n, input=a) for n, a in calls]
    return _message(blocks, "tool_use", input_tokens, output_tokens)


def _message(content: list[Any], stop_reason: str, input_tokens: int, output_tokens: int) -> Message:
    return Message(
        id=f"msg_{next(_ids):04d}",
        type="message",
        role="assistant",
        model="fake",
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


class ScriptedModel:
    def __init__(self, script: Sequence[Message] | Responder, model: str = "fake-model"):
        self.model = model
        self._script = script if callable(script) else list(script)
        self.requests: list[dict[str, Any]] = []

    async def create(self, *, on_text: TextCallback | None = None, **params: Any) -> Message:
        # why: snapshot the messages list; the agent appends to the same list after this call.
        self.requests.append({**params, "messages": json.loads(json.dumps(params["messages"], default=str))})
        if callable(self._script):
            response = self._script(params)
        else:
            if not self._script:
                raise AssertionError("ScriptedModel ran out of scripted responses")
            response = self._script.pop(0)
        if on_text is not None:
            for block in response.content:
                if block.type == "text":
                    for word in block.text.split(" "):
                        on_text(word + " ")
        return response


def tool_results(params: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for message in params["messages"]:
        if message["role"] == "user" and isinstance(message["content"], list):
            out += [b for b in message["content"] if b.get("type") == "tool_result"]
    return out


def oracle_responder(reference_sql: str) -> Responder:
    def respond(params: dict[str, Any]) -> Message:
        results = tool_results(params)
        if not results:
            return tool_reply(("run_query", {"sql": reference_sql}))
        payload = json.loads(results[-1]["content"])
        values = ", ".join(str(v) for row in payload["rows"] for v in row)
        return text_reply(f"The answer is {values}.\n\n```sql\n{reference_sql}\n```")

    return respond
