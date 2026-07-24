"""Built-in theme derivation strategies."""

from __future__ import annotations

from core.theme.strategies.base import get_strategy, register_strategy
from core.theme.strategies.catppuccin import CatppuccinThemeStrategy
from core.theme.strategies.default import DefaultThemeStrategy
from core.theme.strategies.nord import NordThemeStrategy

register_strategy("default", DefaultThemeStrategy())
register_strategy("catppuccin", CatppuccinThemeStrategy())
register_strategy("nord", NordThemeStrategy())

__all__ = ["get_strategy"]
