"""Theme derivation strategy protocol and registry."""

from __future__ import annotations

from typing import Protocol

from core.theme.tokens import CoreTokenSet, ResolvedTheme, ThemeMode


class ThemeStrategy(Protocol):
    def derive(
        self,
        core: CoreTokenSet,
        *,
        scheme_id: str,
        scheme_name: str,
        mode: ThemeMode,
        algorithm: str,
    ) -> ResolvedTheme: ...


_STRATEGIES: dict[str, "ThemeStrategy"] = {}


def register_strategy(strategy_id: str, strategy: ThemeStrategy) -> None:
    _STRATEGIES[strategy_id] = strategy


def get_strategy(strategy_id: str) -> ThemeStrategy:
    try:
        return _STRATEGIES[strategy_id]
    except KeyError as exc:
        raise KeyError(f"Unknown theme derivation strategy: {strategy_id!r}") from exc
