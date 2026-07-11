"""Retrieval budget enforcement helpers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.knowledge.types import RetrievalBudget


class BudgetExceededError(Exception):
    """Retrieval budget limit reached."""


class TurnBudgetTracker:
    """Track latency and adapter call counts for a retrieval turn."""

    def __init__(self, budget: RetrievalBudget) -> None:
        self._budget = budget
        self._t0 = time.perf_counter()
        self._adapter_calls = 0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def check_latency(self) -> None:
        limit = int(self._budget.max_latency_ms or 0)
        if limit > 0 and self.elapsed_ms > limit:
            raise BudgetExceededError(
                f"Retrieval latency budget exceeded ({self.elapsed_ms:.0f}ms > {limit}ms)"
            )

    def record_adapter_call(self) -> None:
        limit = int(self._budget.max_adapter_calls or 0)
        self._adapter_calls += 1
        if limit > 0 and self._adapter_calls > limit:
            raise BudgetExceededError(
                f"Adapter call budget exceeded ({self._adapter_calls} > {limit})"
            )

    def remaining_adapter_calls(self) -> int | None:
        limit = int(self._budget.max_adapter_calls or 0)
        if limit <= 0:
            return None
        return max(0, limit - self._adapter_calls)

    def max_response_bytes(self) -> int:
        return int(self._budget.max_fetch_bytes or 0)
