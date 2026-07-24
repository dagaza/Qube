"""Polarity toggle types for family-aware nav theme switching (§14 Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.theme.tokens import ThemeMode


class PolarityToggleAction(str, Enum):
    APPLY_FALLBACK = "apply_fallback"
    CHOOSE_THEME = "choose_theme"
    CANCEL = "cancel"


@dataclass(frozen=True)
class PolarityToggleRequest:
    family: str
    family_display_name: str
    current_scheme_id: str
    current_display_name: str
    target_mode: ThemeMode
    fallback_scheme_id: str
    fallback_display_name: str
    primary_action_label: str

    @property
    def target_polarity_label(self) -> str:
        return "light" if self.target_mode is ThemeMode.LIGHT else "dark"

    @property
    def message(self) -> str:
        return (
            f"{self.current_display_name} has no {self.target_polarity_label} variant."
        )
