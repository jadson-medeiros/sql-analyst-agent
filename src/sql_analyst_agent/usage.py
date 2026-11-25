from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# why: USD per million tokens, first-party API, prompts up to 100K tokens (claude-api skill table, cached 2026-10-06). Cache writes (5 min TTL) bill at 1.25x input and cache reads at 0.1x input.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass
class UsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0
    per_call: list[dict[str, int]] = field(default_factory=list)

    def add(self, usage: Any) -> None:
        row = {
            "input": usage.input_tokens or 0,
            "output": usage.output_tokens or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", None) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", None) or 0,
        }
        self.input_tokens += row["input"]
        self.output_tokens += row["output"]
        self.cache_creation_input_tokens += row["cache_write"]
        self.cache_read_input_tokens += row["cache_read"]
        self.calls += 1
        self.per_call.append(row)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def cost_usd(self, model: str) -> float | None:
        price = PRICES_PER_MTOK.get(model)
        if price is None:
            return None
        return (
            self.input_tokens * price["input"]
            + self.cache_creation_input_tokens * price["input"] * CACHE_WRITE_MULTIPLIER
            + self.cache_read_input_tokens * price["input"] * CACHE_READ_MULTIPLIER
            + self.output_tokens * price["output"]
        ) / 1_000_000

    def summary(self, model: str) -> str:
        cost = self.cost_usd(model)
        cost_text = f"${cost:.6f}" if cost is not None else "n/a (no price for model)"
        return (
            f"calls={self.calls} input={self.input_tokens} output={self.output_tokens} "
            f"cache_write={self.cache_creation_input_tokens} cache_read={self.cache_read_input_tokens} "
            f"total={self.total_tokens} cost={cost_text}"
        )
