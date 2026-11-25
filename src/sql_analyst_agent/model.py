from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol

from anthropic import AsyncAnthropic
from anthropic.types import Message

DEFAULT_MODEL = "claude-haiku-5-5"

TextCallback = Callable[[str], None]


class ModelClient(Protocol):
    model: str

    async def create(self, *, on_text: TextCallback | None = None, **params: Any) -> Message: ...


class AnthropicModel:
    def __init__(self, model: str | None = None, client: AsyncAnthropic | None = None):
        self.model = model or os.environ.get("SQL_AGENT_MODEL", DEFAULT_MODEL)
        self._client = client or AsyncAnthropic()
        self._effort = os.environ.get("SQL_AGENT_EFFORT")

    async def create(self, *, on_text: TextCallback | None = None, **params: Any) -> Message:
        # hazard: Claude 5 models reject temperature/top_p/top_k with HTTP 400; never pass them here.
        if self._effort:
            params["output_config"] = {"effort": self._effort}
        async with self._client.messages.stream(model=self.model, **params) as stream:
            async for event in stream:
                if event.type == "text" and on_text is not None:
                    on_text(event.text)
            return await stream.get_final_message()
