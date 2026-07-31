"""Settings typography helpers (L2–L5 hierarchy)."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QSizePolicy


def make_settings_card_title(text: str, *, anchor: str | None = None) -> QLabel:
    """L2 card section title (above the card or in the collapsible header)."""
    lbl = QLabel(text)
    lbl.setObjectName("SettingsSubsectionLabel")
    if anchor:
        lbl.setProperty("settings_anchor", anchor)
    return lbl


def make_settings_group_header(text: str) -> QLabel:
    """L3 in-card group header (control clusters inside a section card)."""
    lbl = QLabel(text)
    lbl.setObjectName("SettingsGroupLabel")
    lbl.setWordWrap(True)
    lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    lbl.setMinimumWidth(0)
    return lbl


def make_settings_hint(text: str) -> QLabel:
    """L5 muted body copy for settings sections."""
    hint = QLabel(text)
    hint.setWordWrap(True)
    hint.setObjectName("SettingsHint")
    hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    hint.setMinimumWidth(0)
    return hint


# Backward-compatible aliases used during migration.
make_subsection_label = make_settings_card_title
make_settings_group_label = make_settings_group_header
