"""Nested accent cards and structured info panels for Settings."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core.theme.widget_styles import (
    DISCOVERY_BODY_TEXT,
    DISCOVERY_DIVIDER,
    DISCOVERY_INFO_BULLET,
    DISCOVERY_INFO_CARD,
    DISCOVERY_INFO_HIGHLIGHT,
    DISCOVERY_INFO_KV_KEY,
    DISCOVERY_INFO_KV_VALUE,
    DISCOVERY_INFO_STATUS,
    DISCOVERY_INFO_TITLE,
    DISCOVERY_PROVIDER_CARD,
    DISCOVERY_PROVIDER_NAME,
)
from ui.views.settings.primitives.theme import repolish_widget, settings_theme

_NESTED_CARD_ROLES = {
    "primary": "primary",
    "fallback": "fallback",
    "optional": "optional",
}

DEFAULT_POLICY_KV_KEYS = frozenset(
    {
        "Privacy tier",
        "Primary",
        "Burst",
        "Session",
        "Pacing",
        "On primary failure",
    }
)


def _normalize_nested_role(role: str) -> str:
    return _NESTED_CARD_ROLES.get(role.strip().lower(), "fallback")


def apply_settings_nested_card_theme(
    card: QWidget,
    *,
    accent_role: str,
    is_dark: bool,
) -> None:
    role = _normalize_nested_role(accent_role)
    theme = settings_theme(is_dark=is_dark)
    card.setObjectName("DiscoveryProviderCard")
    card.setProperty("discovery_role", role)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(theme.style(DISCOVERY_PROVIDER_CARD, discovery_role=role))
    repolish_widget(card)


def style_settings_nested_card_title(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryCardProviderName")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_PROVIDER_NAME))


def style_settings_nested_card_body(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryCardBody")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_BODY_TEXT))


def build_settings_divider(*, is_dark: bool) -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    line.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_DIVIDER))
    return line


def apply_settings_info_card_theme(
    card: QWidget,
    *,
    tone: str,
    is_dark: bool,
) -> None:
    variant = tone if tone in ("privacy", "policy", "info") else "policy"
    if variant == "info":
        variant = "policy"
    card.setObjectName("DiscoveryInfoCard")
    card.setProperty("discovery_info_variant", variant)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_CARD, variant=variant))


def style_settings_info_card_title(label: QLabel, *, tone: str, is_dark: bool) -> None:
    variant = tone if tone in ("privacy", "policy", "info") else "policy"
    if variant == "info":
        variant = "policy"
    label.setObjectName("DiscoveryInfoCardTitle")
    label.setStyleSheet(
        settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_TITLE, variant=variant)
    )


def style_settings_info_highlight(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoHighlight")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_HIGHLIGHT))


def style_settings_info_kv_key(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoKvKey")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_KV_KEY))


def style_settings_info_kv_value(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoKvValue")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_KV_VALUE))


def style_settings_info_bullet(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoBullet")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_BULLET))


def style_settings_info_status(label: QLabel, *, is_dark: bool) -> None:
    label.setObjectName("DiscoveryInfoStatus")
    label.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_INFO_STATUS))


def refresh_settings_divider(divider: QWidget, *, is_dark: bool) -> None:
    divider.setStyleSheet(settings_theme(is_dark=is_dark).style(DISCOVERY_DIVIDER))


class SettingsInfoCard(QWidget):
    """Structured info card with title, highlight, and bullet / key-value rows."""

    def __init__(
        self,
        *,
        title: str,
        tone: str = "policy",
        is_dark: bool,
        policy_kv_keys: frozenset[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._tone = tone
        self._is_dark = is_dark
        self._policy_kv_keys = policy_kv_keys or DEFAULT_POLICY_KV_KEYS
        self._title_text = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._title_label = QLabel(title)
        self._title_label.setWordWrap(True)
        layout.addWidget(self._title_label)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)
        self._last_privacy_lines: list[str] | None = None
        self._last_policy_lines: list[str] | None = None
        self._policy_structure: tuple[str, ...] | None = None
        self._policy_value_labels: list[QLabel] = []

        self.refresh_theme(is_dark)

    def refresh_theme(self, is_dark: bool) -> None:
        if is_dark != self._is_dark:
            self._last_privacy_lines = None
            self._last_policy_lines = None
            self._policy_structure = None
            self._policy_value_labels = []
        self._is_dark = is_dark
        apply_settings_info_card_theme(self, tone=self._tone, is_dark=is_dark)
        style_settings_info_card_title(self._title_label, tone=self._tone, is_dark=is_dark)

    @staticmethod
    def _policy_line_structure(lines: list[str], kv_keys: frozenset[str]) -> tuple[str, ...]:
        structure: list[str] = []
        for line in lines:
            if ": " in line:
                key, _value = line.split(": ", 1)
                if key in kv_keys:
                    structure.append(f"kv:{key}")
                    continue
            lower = line.lower()
            if lower.startswith(("brave ", "conservative", "paused")):
                structure.append("status")
                continue
            structure.append("bullet")
        return tuple(structure)

    @staticmethod
    def _policy_line_value_text(line: str, kv_keys: frozenset[str]) -> str:
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in kv_keys:
                return value
        return line if line.startswith("•  ") else f"•  {line}"

    def _swap_content(self, populate) -> None:
        self.setUpdatesEnabled(False)
        try:
            new_content = QWidget()
            new_layout = QVBoxLayout(new_content)
            new_layout.setContentsMargins(0, 0, 0, 0)
            new_layout.setSpacing(6)
            populate(new_layout)

            root = self.layout()
            if root is None:
                return
            root.removeWidget(self._content)
            self._content.deleteLater()
            self._content = new_content
            self._content_layout = new_layout
            root.addWidget(new_content)
        finally:
            self.setUpdatesEnabled(True)
            self.updateGeometry()

    def set_privacy_lines(self, lines: list[str]) -> None:
        normalized = list(lines)
        if normalized == self._last_privacy_lines:
            return
        self._last_privacy_lines = normalized
        is_dark = self._is_dark

        def _populate(content_layout: QVBoxLayout) -> None:
            if not normalized:
                return
            highlight = QLabel(normalized[0])
            highlight.setWordWrap(True)
            style_settings_info_highlight(highlight, is_dark=is_dark)
            content_layout.addWidget(highlight)

            for line in normalized[1:]:
                bullet = QLabel(f"•  {line}")
                bullet.setWordWrap(True)
                style_settings_info_bullet(bullet, is_dark=is_dark)
                content_layout.addWidget(bullet)

        self._swap_content(_populate)

    def set_policy_lines(self, lines: list[str]) -> None:
        normalized = list(lines)
        if normalized == self._last_policy_lines:
            return

        kv_keys = self._policy_kv_keys
        structure = self._policy_line_structure(normalized, kv_keys)
        if (
            structure == self._policy_structure
            and self._policy_value_labels
            and len(self._policy_value_labels) == len(normalized)
        ):
            for label, line in zip(self._policy_value_labels, normalized):
                label.setText(self._policy_line_value_text(line, kv_keys))
            self._last_policy_lines = normalized
            return

        self._last_policy_lines = normalized
        self._policy_structure = structure
        value_labels: list[QLabel] = []
        is_dark = self._is_dark

        def _populate(content_layout: QVBoxLayout) -> None:
            if not normalized:
                return
            for line in normalized:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    if key in kv_keys:
                        row = QWidget()
                        row_layout = QHBoxLayout(row)
                        row_layout.setContentsMargins(0, 0, 0, 0)
                        row_layout.setSpacing(10)

                        key_lbl = QLabel(key)
                        key_lbl.setMinimumWidth(108)
                        style_settings_info_kv_key(key_lbl, is_dark=is_dark)
                        row_layout.addWidget(key_lbl)

                        value_lbl = QLabel(value)
                        value_lbl.setWordWrap(True)
                        style_settings_info_kv_value(value_lbl, is_dark=is_dark)
                        row_layout.addWidget(value_lbl, stretch=1)
                        content_layout.addWidget(row)
                        value_labels.append(value_lbl)
                        continue

                if line.lower().startswith(("brave ", "conservative", "paused")):
                    status = QLabel(line)
                    status.setWordWrap(True)
                    style_settings_info_status(status, is_dark=is_dark)
                    content_layout.addWidget(status)
                    value_labels.append(status)
                    continue

                bullet = QLabel(f"•  {line}")
                bullet.setWordWrap(True)
                style_settings_info_bullet(bullet, is_dark=is_dark)
                content_layout.addWidget(bullet)
                value_labels.append(bullet)

        self._swap_content(_populate)
        self._policy_value_labels = value_labels
