"""Inset callout banners for Settings sections."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core.theme.widget_styles import (
    KNOWLEDGE_ACCESS_HINT,
    KNOWLEDGE_SETUP_CALLOUT,
    KNOWLEDGE_SETUP_CALLOUT_BODY,
    KNOWLEDGE_SETUP_CALLOUT_DISMISS,
    KNOWLEDGE_SETUP_CALLOUT_TITLE,
)
from ui.views.settings.primitives.actions import ACTION_CONTROL_HEIGHT_PX
from ui.views.settings.primitives.theme import repolish_widget, settings_theme


class SettingsCallout(QWidget):
    """Recommended-setup / guidance callout with title, body, and dismiss action."""

    def __init__(
        self,
        *,
        title: str,
        dismiss_text: str = "Dismiss",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.body_label = QLabel()
        self.body_label.setWordWrap(True)
        self.dismiss_btn = QPushButton(dismiss_text)

        self.setMinimumWidth(0)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        content_col = QWidget()
        content_col.setMinimumWidth(0)
        content_col.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        content_layout = QVBoxLayout(content_col)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        content_layout.addWidget(self.title_label)
        content_layout.addWidget(self.body_label)
        layout.addWidget(content_col, stretch=1)
        layout.addWidget(
            self.dismiss_btn,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )


def apply_settings_callout_theme(callout: SettingsCallout, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    callout.setObjectName("KnowledgeSetupCallout")
    callout.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    callout.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT))
    repolish_widget(callout)

    callout.title_label.setObjectName("KnowledgeSetupCalloutTitle")
    callout.title_label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_TITLE))
    repolish_widget(callout.title_label)

    callout.body_label.setObjectName("KnowledgeSetupCalloutBody")
    callout.body_label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_BODY))
    repolish_widget(callout.body_label)

    callout.dismiss_btn.setObjectName("KnowledgeSetupCalloutDismiss")
    callout.dismiss_btn.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    callout.dismiss_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    callout.dismiss_btn.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_DISMISS))
    repolish_widget(callout.dismiss_btn)


def style_settings_access_hint(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    label.setObjectName("KnowledgeAccessHint")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setWordWrap(True)
    label.setStyleSheet(theme.style(KNOWLEDGE_ACCESS_HINT))
    repolish_widget(label)
