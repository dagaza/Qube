"""Reusable settings UI widgets."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ui.components.toggle import PrestigeToggle


def make_subsection_label(text: str, *, anchor: str | None = None) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SettingsSubsectionLabel")
    if anchor:
        lbl.setProperty("settings_anchor", anchor)
    return lbl


def add_subsection_to_form(
    form: QFormLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    lbl = make_subsection_label(text, anchor=anchor)
    form.addRow(lbl)
    return lbl


def add_subsection_to_layout(
    layout: QVBoxLayout, text: str, *, anchor: str | None = None
) -> QLabel:
    lbl = make_subsection_label(text, anchor=anchor)
    layout.addWidget(lbl)
    return lbl


def wrap_subsection(content: QWidget, *, anchor: str | None = None) -> QWidget:
    """Wrap content in a container tagged for engine-mode visibility toggles."""
    wrapper = QWidget()
    if anchor:
        wrapper.setProperty("settings_anchor", anchor)
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(content)
    return wrapper


def make_disclosure_row(
    host,
    label_text: str,
    tooltip: str,
    *,
    panel_object_name: str = "SettingsDisclosurePanel",
) -> tuple[PrestigeToggle, QLabel, QWidget]:
    """Toggle + label row and an empty panel for progressive disclosure."""
    toggle = PrestigeToggle()
    toggle.setToolTip(tooltip)
    label = QLabel(label_text)
    label.setToolTip(tooltip)
    info_btn = host._make_settings_info_button(tooltip)
    label_cluster = QWidget()
    label_cluster_layout = QHBoxLayout(label_cluster)
    label_cluster_layout.setContentsMargins(0, 0, 0, 0)
    label_cluster_layout.setSpacing(6)
    label_cluster_layout.addWidget(label)
    label_cluster_layout.addWidget(info_btn)
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(toggle, alignment=Qt.AlignmentFlag.AlignLeft)
    row_layout.addWidget(label_cluster)
    row_layout.addStretch(1)
    panel = QWidget()
    panel.setObjectName(panel_object_name)
    panel_layout = QVBoxLayout(panel)
    panel_layout.setContentsMargins(0, 8, 0, 0)
    panel_layout.setSpacing(12)
    return toggle, row, panel


def make_ai_status_strip(host) -> QLabel:
    """Read-only status bar for AI & Models section."""
    strip = QLabel("")
    strip.setObjectName("SettingsStatusStrip")
    strip.setWordWrap(True)
    strip.setProperty("class", "SettingsHint")
    host._ai_status_strip = strip
    return strip


def update_ai_status_strip(host) -> None:
    strip = getattr(host, "_ai_status_strip", None)
    if strip is None:
        return
    from core.app_settings import get_engine_mode

    mode = get_engine_mode()
    engine = "Internal" if mode == "internal" else "External"
    model_lbl = getattr(host, "active_native_model_lbl", None)
    model_text = model_lbl.text() if model_lbl is not None else "—"
    ctx_spin = getattr(host, "llm_ctx_spin", None)
    ctx = str(ctx_spin.value()) if ctx_spin is not None else "—"
    strip.setText(f"Engine: {engine}  ·  Model: {model_text}  ·  Context: {ctx}")


def register_theme_button(host, button) -> None:
    """Track a SelectorButton for theme refresh (also see ``collect_theme_buttons``)."""
    buttons = getattr(host, "_theme_buttons", None)
    if buttons is None:
        host._theme_buttons = []
    if button not in host._theme_buttons:
        host._theme_buttons.append(button)


def collect_theme_buttons(host) -> None:
    """Register all SelectorButton widgets under the settings view."""
    from ui.components.selector_button import SelectorButton

    host._theme_buttons = []
    for btn in host.findChildren(SelectorButton):
        host._theme_buttons.append(btn)


def track_internal_ai_label(host, label: QLabel) -> None:
    labels = getattr(host, "_ai_internal_subsection_labels", None)
    if labels is None:
        host._ai_internal_subsection_labels = []
    host._ai_internal_subsection_labels.append(label)


def make_external_engine_hint(host) -> QLabel:
    hint = QLabel(
        "Local model library, hardware tuning, chat template, and startup loading "
        "apply when the Internal engine is selected. Switch AI Engine to Internal "
        "to configure these options."
    )
    hint.setWordWrap(True)
    hint.setProperty("class", "SettingsHint")
    hint.setObjectName("SettingsExternalEngineHint")
    host._ai_external_engine_hint = hint
    return hint
