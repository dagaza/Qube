"""Isolated theme preview widgets for Settings → Themes (no global apply)."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import QEvent, Qt, QSize, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.brand_identity import BRAND_LOGO_STROKE_HEX
from core.surface_fill.constants import SURFACE_CHAT_TRANSCRIPT, SURFACE_LIBRARY_PREVIEW
from core.surface_fill.models import SurfaceProfile
from core.theme.color_utils import with_alpha
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import (
    AGENT_MESSAGE_FRAME,
    AGENT_MESSAGE_SHELL,
    LIST_SURFACE,
    ACCENT_ICON,
    RAG_INDICATOR_STANDBY,
    SETTINGS_CHECKBOX,
    SETTINGS_FORM_CONTROLS,
    SETTINGS_LINE_EDIT,
    SETTINGS_NAV_ICON,
    SETTINGS_SECTION_CARD,
    STAGE_SURFACE,
    TOGGLE_BUTTON,
    TRANSPARENT_TEXT_PREVIEW,
    USER_BUBBLE_FRAME,
    USER_BUBBLE_LABEL,
    UTILITY_ICON_BUTTON,
    WEB_INDICATOR_STANDBY,
)
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.selector_button import SelectorButton
from core.app_settings import (
    get_ui_assistant_message_background,
    get_ui_library_transcript_background,
)
from ui.components.readability_toolbar_styles import readability_font_pair_stylesheet
from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH
from ui.views.settings.settings_card_style import settings_card_content_horizontal_padding_total
from ui.surface_fill.transcript_host import TranscriptWallpaperHost
from ui.shell_theme import (
    muted_icon_color,
    nav_icon_colors,
    retrieval_indicator_colors,
    sidebar_row_action_icon_color,
)


def _indicator_label_style(color: str) -> str:
    return f"color: {color}; font-weight: bold; font-size: 9px; background: transparent; border: none;"


def _section_header_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_muted}; font-weight: 800; font-size: 8px; "
        f"letter-spacing: 1px; text-transform: uppercase; background: transparent; border: none;"
    )


def _sidebar_title_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_muted}; font-weight: 800; font-size: 8px; "
        f"letter-spacing: 0.8px; text-transform: uppercase; background: transparent; border: none;"
    )


def _history_folder_row_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_primary}; background: transparent;"
        f" border: none; font-size: 9px; font-weight: 700; padding: 0px;"
    )


def _history_session_selected_frame_style(resolved: ResolvedTheme) -> str:
    bg = (
        with_alpha(resolved.text_primary, 0.03)
        if not resolved.is_dark
        else with_alpha(resolved.text_primary, 0.05)
    )
    return (
        f"QFrame#ThemePreviewHistorySessionSelected {{"
        f" background-color: {bg};"
        f" border: 1px solid {resolved.accent};"
        f" border-radius: 6px;"
        f" }}"
    )


def _history_session_selected_label_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.list_row_title_selected}; background: transparent;"
        f" border: none; font-size: 9px; font-weight: 500; padding: 0px;"
    )


def _tools_spin_style(resolved: ResolvedTheme) -> str:
    border = resolved.border_subtle if resolved.is_dark else resolved.border
    return f"""
        QDoubleSpinBox {{
            background-color: {resolved.surface_elevated};
            color: {resolved.text_primary};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 0px 4px;
            font-size: 10px;
            min-height: 0px;
        }}
    """


def _settings_menu_button_style(resolved: ResolvedTheme) -> str:
    bg = resolved.surface_elevated
    border = resolved.border
    text = resolved.text_primary
    return (
        f"text-align: left; padding: 4px 8px; background-color: {bg};"
        f" border: 1px solid {border}; color: {text}; border-radius: 6px; font-size: 9px;"
    )


def _preview_shell_colors(resolved: ResolvedTheme) -> dict[str, str]:
    """Colors that match live Conversations chrome (global QSS + sidebar palette)."""
    border = resolved.border_subtle if resolved.is_dark else resolved.border
    top_bg = (
        with_alpha(resolved.text_primary, 0.08)
        if resolved.is_dark
        else resolved.surface_elevated
    )
    return {
        "main_container": resolved.background,
        "nav_sidebar": resolved.surface,
        "history_sidebar": resolved.color(LIST_SURFACE),
        "tools_pane": resolved.surface,
        "chat_stage": resolved.color(STAGE_SURFACE),
        "top_bar": top_bg,
        "border": border,
    }


def _memory_edit_button_style(resolved: ResolvedTheme) -> str:
    """Outline action button matching Memory Manager row Edit/Flag controls."""
    border = resolved.border_subtle if resolved.is_dark else resolved.border
    hover_bg = with_alpha(resolved.accent, 0.10)
    return f"""
        QPushButton {{
            background: transparent;
            color: {resolved.text_primary};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {hover_bg};
            border: 1px solid {resolved.accent};
            color: {resolved.accent};
        }}
    """


def _settings_nav_row_title_style(resolved: ResolvedTheme, *, selected: bool) -> str:
    color = resolved.list_row_title_selected if selected else resolved.text_primary
    return (
        f"color: {color}; background: transparent;"
        f" border: none; font-size: 9px; font-weight: 500; padding: 0px;"
    )


def _settings_nav_row_frame_style(resolved: ResolvedTheme, *, selected: bool) -> str:
    if not selected:
        return "QFrame { background: transparent; border: none; border-radius: 6px; }"
    bg = (
        with_alpha(resolved.text_primary, 0.05)
        if resolved.is_dark
        else with_alpha(resolved.text_primary, 0.03)
    )
    return (
        f"QFrame {{"
        f" background-color: {bg};"
        f" border: 1px solid {resolved.accent};"
        f" border-radius: 6px;"
        f" }}"
    )


def _settings_page_title_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_primary}; font-weight: 800; font-size: 11px;"
        f" background: transparent; border: none;"
    )


def _settings_section_page_title_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_primary}; font-weight: 800; font-size: 10px;"
        f" background: transparent; border: none;"
    )


def _status_bubble_style(resolved: ResolvedTheme, *, state: str) -> str:
    if state == "speaking":
        fg, bg = resolved.warning, with_alpha(resolved.warning, 0.12)
    elif state == "needs_model":
        fg, bg = resolved.info, with_alpha(resolved.info, 0.12)
    else:
        fg, bg = resolved.success, with_alpha(resolved.success, 0.12)
    return (
        f"color: {fg}; background-color: {bg}; border-radius: 10px;"
        f" font-size: 9px; font-weight: 700; padding: 2px 6px;"
    )


class ThemeConversationsPreviewScene(QFrame):
    """Miniature Conversations shell with the tools pane open."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewConversationsScene")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._shell = QFrame()
        self._shell.setObjectName("ThemePreviewShell")
        shell_layout = QVBoxLayout(self._shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self._top_bar = QFrame()
        self._top_bar.setObjectName("ThemePreviewTopBar")
        self._top_bar.setFixedHeight(30)
        top_layout = QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(8, 0, 8, 0)
        top_layout.setSpacing(6)

        left_container = QWidget()
        left_container.setFixedWidth(40)
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self._logo_dot = QLabel()
        self._logo_dot.setFixedSize(10, 10)
        left_layout.addWidget(self._logo_dot)
        self._mic_icon = QLabel()
        self._mic_icon.setFixedSize(12, 12)
        left_layout.addWidget(self._mic_icon)
        left_layout.addStretch()
        top_layout.addWidget(left_container)

        top_layout.addStretch(1)

        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        self._status_bubble = QLabel(" IDLE")
        self._status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_bubble.setFixedSize(72, 18)
        center_layout.addWidget(self._status_bubble)
        self._rag_dot = QLabel("● RAG")
        self._rag_dot.setFixedWidth(36)
        self._web_dot = QLabel("● WEB")
        self._web_dot.setFixedWidth(36)
        self._hybrid_dot = QLabel("● HYBRID")
        self._hybrid_dot.setFixedWidth(48)
        for dot in (self._rag_dot, self._web_dot, self._hybrid_dot):
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            center_layout.addWidget(dot)
        top_layout.addWidget(center_container)

        top_layout.addStretch(1)

        right_container = QWidget()
        right_container.setFixedWidth(40)
        top_layout.addWidget(right_container)

        shell_layout.addWidget(self._top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._nav = QFrame()
        self._nav.setObjectName("ThemePreviewNavSidebar")
        self._nav.setFixedWidth(_PREVIEW_NAV_WIDTH)
        nav_layout = QVBoxLayout(self._nav)
        nav_layout.setContentsMargins(0, 8, 0, 8)
        nav_layout.setSpacing(6)
        self._nav_buttons: list[QPushButton] = []
        for idx, icon in enumerate(("fa5s.comment-alt", "fa5s.book", "fa5s.memory", "fa5s.cog")):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setProperty("class", "NavButton")
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.setIconSize(QSize(11, 11))
            btn._preview_icon = icon  # type: ignore[attr-defined]
            self._nav_buttons.append(btn)
            nav_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addStretch()
        body.addWidget(self._nav)

        self._history = QFrame()
        self._history.setObjectName("ThemePreviewHistorySidebar")
        self._history.setFixedWidth(_PREVIEW_HISTORY_WIDTH)
        history_layout = QVBoxLayout(self._history)
        history_layout.setContentsMargins(8, 8, 6, 8)
        history_layout.setSpacing(4)
        history_title = QLabel("CHATS")
        self._history_title = history_title
        history_layout.addWidget(history_title)

        self._folder_row = QWidget()
        folder_layout = QHBoxLayout(self._folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(2)
        self._folder_chevron = QPushButton()
        self._folder_chevron.setFixedSize(14, 14)
        self._folder_chevron.setIconSize(QSize(8, 8))
        self._folder_chevron.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._folder_chevron.setStyleSheet(
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        self._folder_title = QLabel("Work chats")
        self._folder_title.setObjectName("HistoryFolderTitle")
        folder_layout.addWidget(self._folder_chevron)
        folder_layout.addWidget(self._folder_title, stretch=1)
        history_layout.addWidget(self._folder_row)

        self._session_row = QFrame()
        self._session_row.setObjectName("ThemePreviewHistorySessionSelected")
        session_layout = QHBoxLayout(self._session_row)
        session_layout.setContentsMargins(14, 4, 6, 4)
        session_layout.setSpacing(0)
        self._session_title = QLabel("Project planning")
        self._session_title.setObjectName("HistoryRowTitle")
        session_layout.addWidget(self._session_title)
        history_layout.addWidget(self._session_row)
        history_layout.addStretch()
        body.addWidget(self._history)

        self._chat_content = QFrame()
        self._chat_content.setObjectName("ThemePreviewChatStage")
        chat_layout = QVBoxLayout(self._chat_content)
        chat_layout.setContentsMargins(10, 8, 10, 8)
        chat_layout.setSpacing(6)

        self._agent_block = QFrame()
        self._agent_block.setObjectName("ThemePreviewAgentBlock")
        agent_layout = QVBoxLayout(self._agent_block)
        agent_layout.setContentsMargins(14, 10, 14, 8)
        agent_layout.setSpacing(2)
        self._agent_header = QLabel("QUBE")
        self._agent_header.setObjectName("ThemePreviewAgentHeader")
        self._agent_message = QLabel(
            "Here's how your theme looks in a typical assistant reply."
        )
        self._agent_message.setObjectName("ThemePreviewAgentMessage")
        self._agent_message.setWordWrap(True)
        agent_layout.addWidget(self._agent_header)
        agent_layout.addWidget(self._agent_message)
        chat_layout.addWidget(self._agent_block)

        user_row = QHBoxLayout()
        user_row.addStretch()
        self._user_bubble_frame = QFrame()
        self._user_bubble_frame.setObjectName("ThemePreviewUserBubble")
        user_inner = QHBoxLayout(self._user_bubble_frame)
        user_inner.setContentsMargins(8, 6, 8, 6)
        self._user_bubble_label = QLabel("Sample user message")
        self._user_bubble_label.setObjectName("ThemePreviewUserBubbleLabel")
        user_inner.addWidget(self._user_bubble_label)
        user_row.addWidget(self._user_bubble_frame)
        chat_layout.addLayout(user_row)
        chat_layout.addStretch()

        toggles = QHBoxLayout()
        toggles.setSpacing(4)
        self._web_toggle = QPushButton("Web")
        self._think_toggle = QPushButton("Think")
        for btn in (self._web_toggle, self._think_toggle):
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            toggles.addWidget(btn)
        toggles.addStretch()
        chat_layout.addLayout(toggles)

        composer_row = QHBoxLayout()
        composer_row.setSpacing(4)
        self._composer_input = QLineEdit("Type a message…")
        self._composer_input.setReadOnly(True)
        self._composer_input.setFixedHeight(24)
        self._send_btn = QPushButton()
        self._send_btn.setFixedSize(22, 22)
        self._send_btn.setIconSize(QSize(10, 10))
        composer_row.addWidget(self._composer_input, stretch=1)
        composer_row.addWidget(self._send_btn)
        chat_layout.addLayout(composer_row)
        self._chat_wallpaper_host = TranscriptWallpaperHost(
            SURFACE_CHAT_TRANSCRIPT,
            self._chat_content,
            parent=self,
        )
        body.addWidget(self._chat_wallpaper_host, stretch=1)

        self._tools = QFrame()
        self._tools.setObjectName("ThemePreviewToolsPane")
        self._tools.setFixedWidth(_PREVIEW_TOOLS_WIDTH)
        tools_layout = QVBoxLayout(self._tools)
        tools_layout.setContentsMargins(8, 8, 8, 8)
        tools_layout.setSpacing(6)

        tools_title = QLabel("LOCAL LLM")
        self._tools_title = tools_title
        tools_layout.addWidget(tools_title)

        self._tools_selector = QPushButton("Model ▾")
        self._tools_selector.setObjectName("ThemePreviewToolsSelector")
        tools_layout.addWidget(self._tools_selector)

        param_row = QHBoxLayout()
        param_lbl = QLabel("Temp")
        param_lbl.setProperty("class", "ToolsPaneControl")
        self._tools_param_label = param_lbl
        self._tools_spin = QDoubleSpinBox()
        self._tools_spin.setRange(0.0, 2.0)
        self._tools_spin.setValue(0.7)
        self._tools_spin.setDecimals(1)
        self._tools_spin.setFixedWidth(52)
        self._tools_spin.setFixedHeight(26)
        param_row.addWidget(param_lbl)
        param_row.addStretch()
        param_row.addWidget(self._tools_spin)
        tools_layout.addLayout(param_row)
        tools_layout.addStretch()
        body.addWidget(self._tools)

        shell_layout.addLayout(body)
        root.addWidget(self._shell)
        self._preview_chat_profile: SurfaceProfile | None = None
        self._preview_resolved_wallpaper = None

    def apply_theme(
        self,
        resolved: ResolvedTheme,
        *,
        chat_profile: SurfaceProfile | None = None,
        chat_resolved_wallpaper=None,
    ) -> None:
        colors = _preview_shell_colors(resolved)
        shell_border = resolved.border
        self.setStyleSheet(
            "QFrame#ThemePreviewConversationsScene { background: transparent; border: none; }"
        )
        self._shell.setStyleSheet(
            f"QFrame#ThemePreviewShell {{"
            f" background-color: {colors['main_container']};"
            f" border: 1px solid {shell_border};"
            f" border-radius: 8px;"
            f" }}"
        )
        self._top_bar.setStyleSheet(
            f"QFrame#ThemePreviewTopBar {{"
            f" background-color: {colors['top_bar']};"
            f" border-bottom: 1px solid {colors['border']};"
            f" border-top-left-radius: 8px;"
            f" border-top-right-radius: 8px;"
            f" }}"
        )
        self._logo_dot.setStyleSheet(
            f"background-color: {BRAND_LOGO_STROKE_HEX}; border-radius: 5px; border: none;"
        )
        self._mic_icon.setPixmap(
            qta.icon("fa5s.microphone", color=muted_icon_color(resolved)).pixmap(QSize(12, 12))
        )
        self._status_bubble.setStyleSheet(_status_bubble_style(resolved, state="idle"))

        indicators = retrieval_indicator_colors(resolved)
        self._rag_dot.setStyleSheet(_indicator_label_style(resolved.color(RAG_INDICATOR_STANDBY)))
        self._web_dot.setStyleSheet(
            _indicator_label_style(resolved.color(WEB_INDICATOR_STANDBY))
        )
        self._hybrid_dot.setStyleSheet(
            _indicator_label_style(indicators["off"])
        )

        nav_active, nav_inactive = nav_icon_colors(resolved)
        nav_bg = colors["nav_sidebar"]
        self._nav.setStyleSheet(
            f"QFrame#ThemePreviewNavSidebar {{ background-color: {nav_bg}; border: none; }}"
        )
        for idx, btn in enumerate(self._nav_buttons):
            icon_name = btn._preview_icon  # type: ignore[attr-defined]
            color = nav_active if btn.isChecked() else nav_inactive
            btn.setIcon(qta.icon(icon_name, color=color))
            checked_bg = with_alpha(resolved.text_primary, 0.1) if resolved.is_dark else resolved.surface_pressed
            btn.setStyleSheet(
                f"QPushButton {{ background: {'transparent' if not btn.isChecked() else checked_bg};"
                f" border: none; border-radius: 6px; }}"
            )
            if idx == 0:
                btn.setChecked(True)

        history_bg = colors["history_sidebar"]
        history_border = colors["border"]
        self._history.setStyleSheet(
            f"QFrame#ThemePreviewHistorySidebar {{"
            f" background-color: {history_bg};"
            f" border-right: 1px solid {history_border};"
            f" border: none;"
            f" border-right: 1px solid {history_border};"
            f" }}"
        )
        self._history_title.setStyleSheet(_sidebar_title_style(resolved))

        chevron_color = sidebar_row_action_icon_color(resolved, highlighted=False)
        self._folder_chevron.setIcon(qta.icon("fa5s.chevron-down", color=chevron_color))
        self._folder_title.setStyleSheet(_history_folder_row_style(resolved))
        self._session_row.setStyleSheet(_history_session_selected_frame_style(resolved))
        self._session_title.setStyleSheet(_history_session_selected_label_style(resolved))

        self._chat_content.setStyleSheet(
            "QFrame#ThemePreviewChatStage { background: transparent; border: none; }"
        )
        self._preview_chat_profile = chat_profile
        self._preview_resolved_wallpaper = chat_resolved_wallpaper
        self._chat_wallpaper_host.set_preview_profile(
            chat_profile,
            resolved_wallpaper=chat_resolved_wallpaper,
            theme=resolved,
        )
        assistant_bg = get_ui_assistant_message_background()
        self._agent_block.setStyleSheet(
            resolved.style(
                AGENT_MESSAGE_FRAME,
                enabled=assistant_bg,
                high_contrast=False,
                object_name="ThemePreviewAgentBlock",
            )
        )
        self._agent_block.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            assistant_bg,
        )
        self._agent_block.layout().setContentsMargins(
            *( (14, 10, 14, 8) if assistant_bg else (0, 0, 0, 0) )
        )
        self._agent_header.setStyleSheet(
            f"color: {resolved.chat_header}; font-weight: bold; font-size: 8px;"
            f" letter-spacing: 1px; background: transparent; border: none; margin: 0px;"
        )
        self._agent_message.setStyleSheet(
            f"{resolved.style(AGENT_MESSAGE_SHELL, font_pt=10.0)}"
            f" color: {resolved.chat_agent_text};"
        )
        self._user_bubble_frame.setStyleSheet(
            resolved.style(USER_BUBBLE_FRAME, high_contrast=False)
            + " border-radius: 12px;"
        )
        self._user_bubble_label.setStyleSheet(
            resolved.style(USER_BUBBLE_LABEL, high_contrast=False, font_pt=10.0)
        )

        self._web_toggle.setChecked(True)
        self._think_toggle.setChecked(False)
        self._web_toggle.setStyleSheet(
            resolved.style(TOGGLE_BUTTON, checked=True, active_bg=resolved.link)
        )
        self._think_toggle.setStyleSheet(
            resolved.style(TOGGLE_BUTTON, checked=False, active_bg=resolved.success)
        )

        input_border = resolved.border_subtle if resolved.is_dark else resolved.border
        self._composer_input.setStyleSheet(
            f"QLineEdit {{"
            f" background-color: {resolved.surface_elevated};"
            f" color: {resolved.text_primary};"
            f" border: 1px solid {input_border};"
            f" border-radius: 8px;"
            f" padding: 2px 8px;"
            f" font-size: 10px;"
            f" }}"
        )
        self._send_btn.setIcon(qta.icon("fa5s.paper-plane", color=resolved.link))
        self._send_btn.setStyleSheet("background: transparent; border: none;")

        tools_bg = colors["tools_pane"]
        tools_border = colors["border"]
        self._tools.setStyleSheet(
            f"QFrame#ThemePreviewToolsPane {{"
            f" background-color: {tools_bg};"
            f" border-left: 1px solid {tools_border};"
            f" }}"
        )
        self._tools_title.setStyleSheet(_section_header_style(resolved))
        self._tools_param_label.setStyleSheet(
            f"color: {resolved.text_primary}; font-size: 9px; background: transparent; border: none;"
        )
        self._tools_selector.setStyleSheet(_settings_menu_button_style(resolved))
        self._tools_spin.setStyleSheet(_tools_spin_style(resolved))


class ThemeComponentsPreviewScene(QFrame):
    """Miniature Settings page shell with theme-color samples on the mainstage."""

    _SETTINGS_NAV_SECTIONS: tuple[tuple[str, str, bool], ...] = (
        ("fa5s.globe", "General", False),
        ("fa5s.microphone", "Voice & Audio", False),
        ("fa5s.palette", "Themes", True),
        ("fa5s.memory", "Memory", False),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewComponentsScene")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._shell = QFrame()
        self._shell.setObjectName("ThemePreviewSettingsShell")
        shell_layout = QVBoxLayout(self._shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self._top_bar = QFrame()
        self._top_bar.setObjectName("ThemePreviewSettingsTopBar")
        self._top_bar.setFixedHeight(26)
        top_layout = QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(8, 0, 8, 0)
        top_layout.setSpacing(6)
        left_container = QWidget()
        left_container.setFixedWidth(36)
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self._logo_dot = QLabel()
        self._logo_dot.setFixedSize(10, 10)
        left_layout.addWidget(self._logo_dot)
        self._mic_icon = QLabel()
        self._mic_icon.setFixedSize(12, 12)
        left_layout.addWidget(self._mic_icon)
        left_layout.addStretch()
        top_layout.addWidget(left_container)
        top_layout.addStretch(1)
        self._status_bubble = QLabel(" IDLE")
        self._status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_bubble.setFixedSize(68, 16)
        top_layout.addWidget(self._status_bubble)
        top_layout.addStretch(1)
        top_layout.addWidget(QWidget())
        shell_layout.addWidget(self._top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._nav = QFrame()
        self._nav.setObjectName("ThemePreviewSettingsNavSidebar")
        self._nav.setFixedWidth(_PREVIEW_NAV_WIDTH)
        nav_layout = QVBoxLayout(self._nav)
        nav_layout.setContentsMargins(0, 8, 0, 8)
        nav_layout.setSpacing(6)
        self._nav_buttons: list[QPushButton] = []
        for idx, icon in enumerate(("fa5s.comment-alt", "fa5s.book", "fa5s.memory", "fa5s.cog")):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setProperty("class", "NavButton")
            btn.setCheckable(True)
            btn.setChecked(idx == 3)
            btn.setIconSize(QSize(11, 11))
            btn._preview_icon = icon  # type: ignore[attr-defined]
            self._nav_buttons.append(btn)
            nav_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addStretch()
        body.addWidget(self._nav)

        self._settings_hub = QWidget()
        self._settings_hub.setObjectName("ThemePreviewSettingsHub")
        hub_layout = QHBoxLayout(self._settings_hub)
        hub_layout.setContentsMargins(0, 0, 0, 0)
        hub_layout.setSpacing(0)

        self._settings_sidebar = QFrame()
        self._settings_sidebar.setObjectName("ThemePreviewSettingsSidebar")
        self._settings_sidebar.setFixedWidth(_PREVIEW_SETTINGS_SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(self._settings_sidebar)
        sidebar_layout.setContentsMargins(8, 8, 6, 8)
        sidebar_layout.setSpacing(6)
        self._sidebar_title = QLabel("System Settings")
        self._sidebar_title.setObjectName("ViewTitle")
        sidebar_layout.addWidget(self._sidebar_title)
        self._settings_search = QLineEdit("Search settings…")
        self._settings_search.setReadOnly(True)
        self._settings_search.setFixedHeight(22)
        sidebar_layout.addWidget(self._settings_search)

        self._settings_nav_rows: list[tuple[QFrame, QLabel, QLabel, str, bool]] = []
        for icon_name, title, selected in self._SETTINGS_NAV_SECTIONS:
            row_frame = QFrame()
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(6, 4, 6, 4)
            row_layout.setSpacing(6)
            icon_label = QLabel()
            icon_label.setFixedSize(12, 12)
            title_label = QLabel(title)
            title_label.setObjectName("HistoryRowTitle")
            row_layout.addWidget(icon_label)
            row_layout.addWidget(title_label, stretch=1)
            sidebar_layout.addWidget(row_frame)
            self._settings_nav_rows.append(
                (row_frame, icon_label, title_label, icon_name, selected)
            )
        sidebar_layout.addStretch()
        hub_layout.addWidget(self._settings_sidebar)

        self._settings_content = QWidget()
        self._settings_content.setObjectName("ThemePreviewSettingsContent")
        content_layout = QVBoxLayout(self._settings_content)
        content_layout.setContentsMargins(8, 10, 8, 8)
        content_layout.setSpacing(6)

        section_header = QHBoxLayout()
        section_header.setContentsMargins(0, 0, 0, 0)
        section_header.setSpacing(6)
        self._section_icon = QLabel()
        self._section_icon.setFixedSize(14, 14)
        self._section_title = QLabel("Themes")
        section_header.addWidget(self._section_icon)
        section_header.addWidget(self._section_title)
        section_header.addStretch()
        content_layout.addLayout(section_header)

        card = QFrame()
        self._components_card = card
        card.setObjectName("ThemePreviewComponentsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        card_title = QLabel("Theme colors")
        self._card_title = card_title
        card_layout.addWidget(card_title)

        self._checkbox = QCheckBox("Auto-adjust text for readable contrast")
        self._checkbox.setChecked(True)
        card_layout.addWidget(self._checkbox)

        form_row = QHBoxLayout()
        self._line_edit = QLineEdit("Accent color")
        self._line_edit.setReadOnly(True)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0, 100)
        self._spin.setValue(42)
        self._spin.setFixedWidth(64)
        form_row.addWidget(self._line_edit, stretch=1)
        form_row.addWidget(self._spin)
        card_layout.addLayout(form_row)

        self._selector = SelectorButton("Variant ▾", parent=self)
        card_layout.addWidget(self._selector)

        btn_row = QHBoxLayout()
        self._primary_btn = QPushButton("Apply")
        self._danger_btn = QPushButton("Revert")
        btn_row.addWidget(self._primary_btn)
        btn_row.addWidget(self._danger_btn)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)
        content_layout.addWidget(card)

        memory_card = QFrame()
        self._memory_card = memory_card
        memory_card.setObjectName("ThemePreviewMemoryCard")
        memory_layout = QHBoxLayout(memory_card)
        memory_layout.setContentsMargins(8, 6, 8, 6)
        memory_layout.setSpacing(6)
        memory_text = QVBoxLayout()
        memory_text.setSpacing(2)
        self._memory_title = QLabel("User prefers concise answers")
        self._memory_meta = QLabel("preference · enriched yesterday")
        memory_text.addWidget(self._memory_title)
        memory_text.addWidget(self._memory_meta)
        memory_layout.addLayout(memory_text, stretch=1)
        self._memory_edit_btn = QPushButton("Edit")
        memory_layout.addWidget(self._memory_edit_btn)
        content_layout.addWidget(memory_card)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self._status_idle = QLabel(" IDLE")
        self._status_speaking = QLabel(" SPEAKING")
        self._status_needs = QLabel(" NEEDS MODEL")
        for lbl in (self._status_idle, self._status_speaking, self._status_needs):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedHeight(18)
            status_row.addWidget(lbl)
        status_row.addStretch()
        content_layout.addLayout(status_row)

        self._tooltip_sample = QLabel("Tooltip sample")
        content_layout.addWidget(self._tooltip_sample)
        content_layout.addStretch()
        hub_layout.addWidget(self._settings_content, stretch=1)

        body.addWidget(self._settings_hub, stretch=1)
        shell_layout.addLayout(body)
        root.addWidget(self._shell)

    def apply_theme(self, resolved: ResolvedTheme) -> None:
        colors = _preview_shell_colors(resolved)
        shell_border = resolved.border
        self.setStyleSheet(
            "QFrame#ThemePreviewComponentsScene { background: transparent; border: none; }"
        )
        self._shell.setStyleSheet(
            f"QFrame#ThemePreviewSettingsShell {{"
            f" background-color: {colors['main_container']};"
            f" border: 1px solid {shell_border};"
            f" border-radius: 8px;"
            f" }}"
        )
        self._top_bar.setStyleSheet(
            f"QFrame#ThemePreviewSettingsTopBar {{"
            f" background-color: {colors['top_bar']};"
            f" border-bottom: 1px solid {colors['border']};"
            f" border-top-left-radius: 8px;"
            f" border-top-right-radius: 8px;"
            f" }}"
        )
        self._logo_dot.setStyleSheet(
            f"background-color: {BRAND_LOGO_STROKE_HEX}; border-radius: 5px; border: none;"
        )
        self._mic_icon.setPixmap(
            qta.icon("fa5s.microphone", color=muted_icon_color(resolved)).pixmap(QSize(12, 12))
        )
        self._status_bubble.setStyleSheet(_status_bubble_style(resolved, state="idle"))

        nav_active, nav_inactive = nav_icon_colors(resolved)
        self._nav.setStyleSheet(
            f"QFrame#ThemePreviewSettingsNavSidebar {{"
            f" background-color: {colors['nav_sidebar']}; border: none;"
            f" }}"
        )
        for idx, btn in enumerate(self._nav_buttons):
            icon_name = btn._preview_icon  # type: ignore[attr-defined]
            btn.setChecked(idx == 3)
            color = nav_active if btn.isChecked() else nav_inactive
            btn.setIcon(qta.icon(icon_name, color=color))
            checked_bg = (
                with_alpha(resolved.text_primary, 0.1)
                if resolved.is_dark
                else resolved.surface_pressed
            )
            btn.setStyleSheet(
                f"QPushButton {{ background: {'transparent' if not btn.isChecked() else checked_bg};"
                f" border: none; border-radius: 6px; }}"
            )

        sidebar_border = colors["border"]
        self._settings_sidebar.setStyleSheet(
            f"QFrame#ThemePreviewSettingsSidebar {{"
            f" background-color: {resolved.sidebar_surface};"
            f" border-right: 1px solid {sidebar_border};"
            f" }}"
        )
        self._sidebar_title.setStyleSheet(_settings_page_title_style(resolved))
        input_border = resolved.border_subtle if resolved.is_dark else resolved.border
        self._settings_search.setStyleSheet(
            f"QLineEdit {{"
            f" background-color: {resolved.surface_elevated};"
            f" color: {resolved.text_secondary};"
            f" border: 1px solid {input_border};"
            f" border-radius: 6px;"
            f" padding: 2px 6px;"
            f" font-size: 9px;"
            f" }}"
        )
        nav_icon_color = resolved.color(SETTINGS_NAV_ICON)
        for row_frame, icon_label, title_label, icon_name, selected in self._settings_nav_rows:
            row_frame.setStyleSheet(_settings_nav_row_frame_style(resolved, selected=selected))
            icon_label.setPixmap(
                qta.icon(icon_name, color=nav_icon_color).pixmap(QSize(12, 12))
            )
            title_label.setStyleSheet(
                _settings_nav_row_title_style(resolved, selected=selected)
            )

        self._settings_content.setStyleSheet(
            f"QWidget#ThemePreviewSettingsContent {{"
            f" background-color: {resolved.background};"
            f" border: none;"
            f" }}"
        )
        self._section_icon.setPixmap(
            qta.icon("fa5s.palette", color=nav_icon_color).pixmap(QSize(14, 14))
        )
        self._section_title.setStyleSheet(_settings_section_page_title_style(resolved))

        card_style = resolved.style(SETTINGS_SECTION_CARD, object_name="ThemePreviewComponentsCard")
        self._components_card.setStyleSheet(card_style)
        self._card_title.setStyleSheet(
            f"color: {resolved.text_primary}; font-weight: 700; font-size: 10px;"
            f" background: transparent; border: none;"
        )
        self._checkbox.setStyleSheet(resolved.style(SETTINGS_CHECKBOX))
        self._line_edit.setStyleSheet(resolved.style(SETTINGS_LINE_EDIT))
        self._spin.setStyleSheet(resolved.style(SETTINGS_FORM_CONTROLS))
        self._selector.setText("Variant ▾")
        self._selector.apply_theme(is_dark=resolved.is_dark, theme=resolved)
        apply_brand_primary(self._primary_btn, theme=resolved)
        apply_brand_danger(self._danger_btn, theme=resolved)

        memory_bg = resolved.surface_elevated if resolved.is_dark else resolved.surface
        memory_border = resolved.border_subtle if resolved.is_dark else resolved.border
        self._memory_card.setStyleSheet(
            f"QFrame#ThemePreviewMemoryCard {{"
            f" background-color: {memory_bg};"
            f" border: 1px solid {memory_border};"
            f" border-radius: 8px;"
            f" }}"
        )
        self._memory_title.setStyleSheet(
            f"color: {resolved.text_primary}; font-size: 10px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        self._memory_meta.setStyleSheet(
            f"color: {resolved.text_muted}; font-size: 9px;"
            f" background: transparent; border: none;"
        )
        self._memory_edit_btn.setStyleSheet(_memory_edit_button_style(resolved))

        self._status_idle.setStyleSheet(_status_bubble_style(resolved, state="idle"))
        self._status_speaking.setStyleSheet(_status_bubble_style(resolved, state="speaking"))
        self._status_needs.setStyleSheet(_status_bubble_style(resolved, state="needs_model"))
        self._tooltip_sample.setStyleSheet(
            f"color: {resolved.text_primary};"
            f" background-color: {resolved.tooltip_bg};"
            f" border: 1px solid {resolved.tooltip_border};"
            f" border-radius: 6px; padding: 4px 8px; font-size: 9px;"
        )


_LIBRARY_PREVIEW_COLUMN_MAX = 280
_LIBRARY_TRANSCRIPT_CARD_MARGINS = (14, 10, 14, 8)
_LIBRARY_PREVIEW_UTILITY_BTN = 22
_LIBRARY_PREVIEW_UTILITY_ICON_PX = 12


class _PreviewColumnWidthHost(QWidget):
    """Centers a column; inner width = min(available, cap)."""

    def __init__(self, inner: QWidget, max_w: int, parent=None) -> None:
        super().__init__(parent)
        self._inner = inner
        self._max_w = max(1, int(max_w))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addStretch(1)
        lay.addWidget(inner, 0)
        lay.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _sync_inner_width(self) -> None:
        w = min(self._max_w, max(1, self.width()))
        if self._inner.width() != w:
            self._inner.setFixedWidth(w)
        self._inner.updateGeometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_inner_width()


_PREVIEW_LIBRARY_SIDEBAR_WIDTH = 110


def _library_doc_row_style(resolved: ResolvedTheme) -> str:
    return (
        f"color: {resolved.text_primary}; background: transparent;"
        f" border: none; font-size: 9px; font-weight: 500; padding: 0px;"
    )


def _library_doc_selected_frame_style(resolved: ResolvedTheme) -> str:
    bg = (
        with_alpha(resolved.text_primary, 0.03)
        if not resolved.is_dark
        else with_alpha(resolved.text_primary, 0.05)
    )
    return (
        f"QFrame#ThemePreviewLibraryDocSelected {{"
        f" background-color: {bg};"
        f" border: 1px solid {resolved.accent};"
        f" border-radius: 6px;"
        f" }}"
    )


class ThemeLibraryPreviewScene(QFrame):
    """Miniature Library page shell with list sidebar, preview mainstage, and tools pane."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewLibraryScene")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._shell = QFrame()
        self._shell.setObjectName("ThemePreviewLibraryShell")
        shell_layout = QVBoxLayout(self._shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self._top_bar = QFrame()
        self._top_bar.setObjectName("ThemePreviewLibraryTopBar")
        self._top_bar.setFixedHeight(30)
        top_layout = QHBoxLayout(self._top_bar)
        top_layout.setContentsMargins(8, 0, 8, 0)
        top_layout.setSpacing(6)

        left_container = QWidget()
        left_container.setFixedWidth(40)
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self._logo_dot = QLabel()
        self._logo_dot.setFixedSize(10, 10)
        left_layout.addWidget(self._logo_dot)
        self._mic_icon = QLabel()
        self._mic_icon.setFixedSize(12, 12)
        left_layout.addWidget(self._mic_icon)
        left_layout.addStretch()
        top_layout.addWidget(left_container)
        top_layout.addStretch(1)

        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)
        self._status_bubble = QLabel(" IDLE")
        self._status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_bubble.setFixedSize(72, 18)
        center_layout.addWidget(self._status_bubble)
        self._rag_dot = QLabel("● RAG")
        self._rag_dot.setFixedWidth(36)
        self._web_dot = QLabel("● WEB")
        self._web_dot.setFixedWidth(36)
        self._hybrid_dot = QLabel("● HYBRID")
        self._hybrid_dot.setFixedWidth(48)
        for dot in (self._rag_dot, self._web_dot, self._hybrid_dot):
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            center_layout.addWidget(dot)
        top_layout.addWidget(center_container)
        top_layout.addStretch(1)
        top_layout.addWidget(QWidget())
        shell_layout.addWidget(self._top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._nav = QFrame()
        self._nav.setObjectName("ThemePreviewLibraryNavSidebar")
        self._nav.setFixedWidth(_PREVIEW_NAV_WIDTH)
        nav_layout = QVBoxLayout(self._nav)
        nav_layout.setContentsMargins(0, 8, 0, 8)
        nav_layout.setSpacing(6)
        self._nav_buttons: list[QPushButton] = []
        for idx, icon in enumerate(("fa5s.comment-alt", "fa5s.book", "fa5s.memory", "fa5s.cog")):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setProperty("class", "NavButton")
            btn.setCheckable(True)
            btn.setChecked(idx == 1)
            btn.setIconSize(QSize(11, 11))
            btn._preview_icon = icon  # type: ignore[attr-defined]
            self._nav_buttons.append(btn)
            nav_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        nav_layout.addStretch()
        body.addWidget(self._nav)

        self._library_sidebar = QFrame()
        self._library_sidebar.setObjectName("ThemePreviewLibrarySidebar")
        self._library_sidebar.setFixedWidth(_PREVIEW_LIBRARY_SIDEBAR_WIDTH)
        library_layout = QVBoxLayout(self._library_sidebar)
        library_layout.setContentsMargins(8, 8, 6, 8)
        library_layout.setSpacing(4)
        self._library_title = QLabel("LIBRARY")
        library_layout.addWidget(self._library_title)

        self._library_search = QLineEdit("Search titles…")
        self._library_search.setReadOnly(True)
        self._library_search.setFixedHeight(22)
        library_layout.addWidget(self._library_search)

        self._library_folder_row = QWidget()
        folder_layout = QHBoxLayout(self._library_folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(2)
        self._library_folder_chevron = QPushButton()
        self._library_folder_chevron.setFixedSize(14, 14)
        self._library_folder_chevron.setIconSize(QSize(8, 8))
        self._library_folder_chevron.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._library_folder_chevron.setStyleSheet(
            "QPushButton { border: none; background: transparent; padding: 0px; }"
        )
        self._library_folder_title = QLabel("Main")
        folder_layout.addWidget(self._library_folder_chevron)
        folder_layout.addWidget(self._library_folder_title, stretch=1)
        library_layout.addWidget(self._library_folder_row)

        self._library_doc_selected = QFrame()
        self._library_doc_selected.setObjectName("ThemePreviewLibraryDocSelected")
        selected_layout = QHBoxLayout(self._library_doc_selected)
        selected_layout.setContentsMargins(14, 4, 6, 4)
        selected_layout.setSpacing(0)
        self._library_doc_selected_label = QLabel("Quarterly Report.pdf")
        selected_layout.addWidget(self._library_doc_selected_label)
        library_layout.addWidget(self._library_doc_selected)

        self._library_doc_other = QLabel("Design Notes.md")
        library_layout.addWidget(self._library_doc_other)
        library_layout.addStretch()
        body.addWidget(self._library_sidebar)

        self._mainstage = QWidget()
        self._mainstage.setObjectName("ThemePreviewLibraryMainstage")
        layout = QVBoxLayout(self._mainstage)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        utility_toolbar = QFrame()
        utility_toolbar.setObjectName("ThemePreviewLibraryUtilityToolbar")
        utility_toolbar.setFixedHeight(28)
        utility_layout = QHBoxLayout(utility_toolbar)
        utility_layout.setContentsMargins(0, 0, 0, 0)
        utility_layout.setSpacing(4)
        self._font_minus_btn = QPushButton("A−")
        self._font_plus_btn = QPushButton("A+")
        for btn in (self._font_minus_btn, self._font_plus_btn):
            btn.setFixedSize(_LIBRARY_PREVIEW_UTILITY_BTN, _LIBRARY_PREVIEW_UTILITY_BTN)
            btn.setEnabled(False)
            utility_layout.addWidget(btn)
        self._line_height_btn = QPushButton()
        self._text_align_btn = QPushButton()
        self._reader_focus_btn = QPushButton()
        self._high_contrast_btn = QPushButton()
        self._layout_mode_btn = QPushButton()
        for btn in (
            self._line_height_btn,
            self._text_align_btn,
            self._reader_focus_btn,
            self._high_contrast_btn,
            self._layout_mode_btn,
        ):
            btn.setFixedSize(_LIBRARY_PREVIEW_UTILITY_BTN, _LIBRARY_PREVIEW_UTILITY_BTN)
            btn.setEnabled(False)
            utility_layout.addWidget(btn)
        utility_layout.addStretch()
        layout.addWidget(utility_toolbar)

        header_host = QWidget()
        header_layout = QVBoxLayout(header_host)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)
        self._doc_title = QLabel("Quarterly Report.pdf")
        self._doc_title.setObjectName("ThemePreviewLibraryDocTitle")
        self._doc_title.setWordWrap(True)
        self._doc_title.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self._doc_stats = QLabel("12 pages · 4,200 words")
        self._doc_stats.setObjectName("ThemePreviewLibraryDocStats")
        self._doc_stats.setWordWrap(True)
        self._doc_stats.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        header_layout.addWidget(self._doc_title)
        header_layout.addWidget(self._doc_stats)
        self._header_width_host = _PreviewColumnWidthHost(
            header_host, _LIBRARY_PREVIEW_COLUMN_MAX
        )
        layout.addWidget(self._header_width_host)

        self._transcript_card = QFrame()
        self._transcript_card.setObjectName("ThemePreviewLibraryTranscriptCard")
        card_layout = QVBoxLayout(self._transcript_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        self._body_text = QLabel(
            "The quarterly report summarizes revenue growth across all regions. "
            "Key findings highlight improved margins in the enterprise segment.\n\n"
            "Subscription services continued to expand, with retention rates "
            "exceeding targets in North America and EMEA."
        )
        self._body_text.setObjectName("ThemePreviewLibraryBody")
        self._body_text.setWordWrap(True)
        self._body_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        card_layout.addWidget(self._body_text)
        self._transcript_width_host = _PreviewColumnWidthHost(
            self._transcript_card, _LIBRARY_PREVIEW_COLUMN_MAX
        )
        layout.addWidget(self._transcript_width_host, stretch=1)

        self._wallpaper_host = TranscriptWallpaperHost(
            SURFACE_LIBRARY_PREVIEW,
            self._mainstage,
            parent=self,
        )
        body.addWidget(self._wallpaper_host, stretch=1)

        self._tools = QFrame()
        self._tools.setObjectName("ThemePreviewLibraryToolsPane")
        self._tools.setFixedWidth(_PREVIEW_TOOLS_WIDTH)
        tools_layout = QVBoxLayout(self._tools)
        tools_layout.setContentsMargins(8, 8, 8, 8)
        tools_layout.setSpacing(6)
        self._tools_title = QLabel("LOCAL LLM")
        tools_layout.addWidget(self._tools_title)
        self._tools_selector = QPushButton("Model ▾")
        self._tools_selector.setObjectName("ThemePreviewLibraryToolsSelector")
        tools_layout.addWidget(self._tools_selector)
        param_row = QHBoxLayout()
        self._tools_param_label = QLabel("Temp")
        self._tools_spin = QDoubleSpinBox()
        self._tools_spin.setRange(0.0, 2.0)
        self._tools_spin.setValue(0.7)
        self._tools_spin.setDecimals(1)
        self._tools_spin.setFixedWidth(52)
        self._tools_spin.setFixedHeight(26)
        param_row.addWidget(self._tools_param_label)
        param_row.addStretch()
        param_row.addWidget(self._tools_spin)
        tools_layout.addLayout(param_row)
        tools_layout.addStretch()
        body.addWidget(self._tools)

        shell_layout.addLayout(body)
        root.addWidget(self._shell)

    def apply_theme(
        self,
        resolved: ResolvedTheme,
        *,
        library_profile: SurfaceProfile | None = None,
        library_resolved_wallpaper=None,
    ) -> None:
        colors = _preview_shell_colors(resolved)
        shell_border = resolved.border
        self.setStyleSheet(
            "QFrame#ThemePreviewLibraryScene { background: transparent; border: none; }"
        )
        self._shell.setStyleSheet(
            f"QFrame#ThemePreviewLibraryShell {{"
            f" background-color: {colors['main_container']};"
            f" border: 1px solid {shell_border};"
            f" border-radius: 8px;"
            f" }}"
        )
        self._top_bar.setStyleSheet(
            f"QFrame#ThemePreviewLibraryTopBar {{"
            f" background-color: {colors['top_bar']};"
            f" border-bottom: 1px solid {colors['border']};"
            f" border-top-left-radius: 8px;"
            f" border-top-right-radius: 8px;"
            f" }}"
        )
        self._logo_dot.setStyleSheet(
            f"background-color: {BRAND_LOGO_STROKE_HEX}; border-radius: 5px; border: none;"
        )
        self._mic_icon.setPixmap(
            qta.icon("fa5s.microphone", color=muted_icon_color(resolved)).pixmap(QSize(12, 12))
        )
        self._status_bubble.setStyleSheet(_status_bubble_style(resolved, state="idle"))
        self._rag_dot.setStyleSheet(_indicator_label_style(resolved.color(RAG_INDICATOR_STANDBY)))
        self._web_dot.setStyleSheet(
            _indicator_label_style(resolved.color(WEB_INDICATOR_STANDBY))
        )
        indicators = retrieval_indicator_colors(resolved)
        self._hybrid_dot.setStyleSheet(
            _indicator_label_style(indicators["off"])
        )

        nav_active, nav_inactive = nav_icon_colors(resolved)
        nav_bg = colors["nav_sidebar"]
        self._nav.setStyleSheet(
            f"QFrame#ThemePreviewLibraryNavSidebar {{ background-color: {nav_bg}; border: none; }}"
        )
        for idx, btn in enumerate(self._nav_buttons):
            icon_name = btn._preview_icon  # type: ignore[attr-defined]
            color = nav_active if btn.isChecked() else nav_inactive
            btn.setIcon(qta.icon(icon_name, color=color))
            checked_bg = (
                with_alpha(resolved.text_primary, 0.1)
                if resolved.is_dark
                else resolved.surface_pressed
            )
            btn.setStyleSheet(
                f"QPushButton {{ background: {'transparent' if not btn.isChecked() else checked_bg};"
                f" border: none; border-radius: 6px; }}"
            )
            if idx == 1:
                btn.setChecked(True)

        library_bg = colors["history_sidebar"]
        library_border = colors["border"]
        self._library_sidebar.setStyleSheet(
            f"QFrame#ThemePreviewLibrarySidebar {{"
            f" background-color: {library_bg};"
            f" border-right: 1px solid {library_border};"
            f" }}"
        )
        self._library_title.setStyleSheet(_sidebar_title_style(resolved))
        input_border = resolved.border_subtle if resolved.is_dark else resolved.border
        self._library_search.setStyleSheet(
            f"QLineEdit {{"
            f" background-color: {resolved.surface_elevated};"
            f" color: {resolved.text_secondary};"
            f" border: 1px solid {input_border};"
            f" border-radius: 6px;"
            f" padding: 2px 6px;"
            f" font-size: 9px;"
            f" }}"
        )
        chevron_color = sidebar_row_action_icon_color(resolved, highlighted=False)
        self._library_folder_chevron.setIcon(
            qta.icon("fa5s.chevron-down", color=chevron_color)
        )
        self._library_folder_title.setStyleSheet(_history_folder_row_style(resolved))
        self._library_doc_selected.setStyleSheet(_library_doc_selected_frame_style(resolved))
        self._library_doc_selected_label.setStyleSheet(
            _history_session_selected_label_style(resolved)
        )
        self._library_doc_other.setStyleSheet(_library_doc_row_style(resolved))

        self._mainstage.setStyleSheet(
            "QWidget#ThemePreviewLibraryMainstage { background: transparent; border: none; }"
        )
        self._wallpaper_host.set_preview_profile(
            library_profile,
            resolved_wallpaper=library_resolved_wallpaper,
            theme=resolved,
        )

        icon_muted = resolved.color(ACCENT_ICON)
        utility_icon_style = resolved.style(UTILITY_ICON_BUTTON)
        font_btn_style = readability_font_pair_stylesheet(
            is_dark=resolved.is_dark,
            theme=resolved,
            button_px=_LIBRARY_PREVIEW_UTILITY_BTN,
        )
        self._font_minus_btn.setStyleSheet(font_btn_style)
        self._font_plus_btn.setStyleSheet(font_btn_style)
        self._line_height_btn.setIcon(
            qta.icon("fa5s.text-height", color=icon_muted)
        )
        self._line_height_btn.setIconSize(
            QSize(_LIBRARY_PREVIEW_UTILITY_ICON_PX, _LIBRARY_PREVIEW_UTILITY_ICON_PX)
        )
        self._text_align_btn.setIcon(
            qta.icon("fa5s.align-left", color=icon_muted)
        )
        self._text_align_btn.setIconSize(
            QSize(_LIBRARY_PREVIEW_UTILITY_ICON_PX, _LIBRARY_PREVIEW_UTILITY_ICON_PX)
        )
        self._reader_focus_btn.setIcon(
            qta.icon("fa5s.crosshairs", color=icon_muted)
        )
        self._reader_focus_btn.setIconSize(
            QSize(_LIBRARY_PREVIEW_UTILITY_ICON_PX, _LIBRARY_PREVIEW_UTILITY_ICON_PX)
        )
        self._high_contrast_btn.setIcon(
            qta.icon("fa5s.adjust", color=icon_muted)
        )
        self._high_contrast_btn.setIconSize(
            QSize(_LIBRARY_PREVIEW_UTILITY_ICON_PX, _LIBRARY_PREVIEW_UTILITY_ICON_PX)
        )
        self._layout_mode_btn.setIcon(
            qta.icon("fa5s.columns", color=icon_muted)
        )
        self._layout_mode_btn.setIconSize(
            QSize(_LIBRARY_PREVIEW_UTILITY_ICON_PX, _LIBRARY_PREVIEW_UTILITY_ICON_PX)
        )
        for btn in (
            self._line_height_btn,
            self._text_align_btn,
            self._reader_focus_btn,
            self._high_contrast_btn,
            self._layout_mode_btn,
        ):
            btn.setStyleSheet(utility_icon_style)

        self._doc_title.setStyleSheet(
            f"color: {resolved.text_primary}; font-weight: 700; font-size: 11px;"
            f" letter-spacing: 0.5px; background: transparent; border: none;"
        )
        self._doc_stats.setStyleSheet(
            f"color: {resolved.text_secondary}; font-size: 9px; font-weight: 500;"
            f" background: transparent; border: none;"
        )

        transcript_bg = get_ui_library_transcript_background()
        self._transcript_card.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            transcript_bg,
        )
        self._transcript_card.setStyleSheet(
            resolved.style(
                AGENT_MESSAGE_FRAME,
                enabled=transcript_bg,
                high_contrast=False,
                object_name="ThemePreviewLibraryTranscriptCard",
            )
        )
        card_layout = self._transcript_card.layout()
        if card_layout is not None:
            margins = _LIBRARY_TRANSCRIPT_CARD_MARGINS if transcript_bg else (0, 0, 0, 0)
            card_layout.setContentsMargins(*margins)

        self._body_text.setStyleSheet(
            resolved.style(TRANSPARENT_TEXT_PREVIEW, color=resolved.text_primary, font_pt=9.0)
        )

        tools_bg = colors["tools_pane"]
        tools_border = colors["border"]
        self._tools.setStyleSheet(
            f"QFrame#ThemePreviewLibraryToolsPane {{"
            f" background-color: {tools_bg};"
            f" border-left: 1px solid {tools_border};"
            f" }}"
        )
        self._tools_title.setStyleSheet(_section_header_style(resolved))
        self._tools_param_label.setStyleSheet(
            f"color: {resolved.text_primary}; font-size: 9px; background: transparent; border: none;"
        )
        self._tools_selector.setStyleSheet(_settings_menu_button_style(resolved))
        self._tools_spin.setStyleSheet(_tools_spin_style(resolved))


_PREVIEW_SNAPSHOT_HEIGHT = 280
# Main-window chrome at the design minimum (1200px wide, tools pane collapsed).
_MAIN_WINDOW_MIN_WIDTH = 1200
_MAIN_NAV_WIDTH = 70
_TOOLS_PANE_COLLAPSED_WIDTH = 40
_SETTINGS_VIEW_RIGHT_MARGIN = 40
_SETTINGS_RIGHT_HOST_LEFT_MARGIN = 10
_SETTINGS_CONTENT_LEFT_MARGIN = 8
_THEMES_PAGE_HORIZONTAL_MARGIN = 30
_SETTINGS_SECTION_CARD = "SettingsSectionCard"


def _preview_card_horizontal_padding() -> int:
    return settings_card_content_horizontal_padding_total()
# Fixed miniature shell proportions (sidebars stay constant; mainstage fills remainder).
_PREVIEW_NAV_WIDTH = 34
_PREVIEW_HISTORY_WIDTH = 110
_PREVIEW_SETTINGS_SIDEBAR_WIDTH = 108
_PREVIEW_TOOLS_WIDTH = 118
_PREVIEW_LAYOUT_MIN_WIDTH = 240


def _design_preview_width_at_min_window() -> int:
    """Width available to the preview card at the app layout minimum."""
    main_stage = (
        _MAIN_WINDOW_MIN_WIDTH
        - _MAIN_NAV_WIDTH
        - _TOOLS_PANE_COLLAPSED_WIDTH
    )
    settings_hub = main_stage - _SETTINGS_VIEW_RIGHT_MARGIN
    return max(
        320,
        settings_hub
        - LEFT_NAV_LIST_SIDEBAR_WIDTH
        - _SETTINGS_RIGHT_HOST_LEFT_MARGIN
        - _SETTINGS_CONTENT_LEFT_MARGIN
        - _THEMES_PAGE_HORIZONTAL_MARGIN
        - _preview_card_horizontal_padding(),
    )


def _find_settings_section_card(panel: QWidget) -> QWidget | None:
    widget = panel.parentWidget()
    while widget is not None:
        if widget.objectName() == _SETTINGS_SECTION_CARD:
            return widget
        widget = widget.parentWidget()
    return None


def _preview_card_inner_width(panel: QWidget) -> int | None:
    card = _find_settings_section_card(panel)
    if card is None:
        return None
    return max(0, card.width() - _preview_card_horizontal_padding())


def _preview_scroll_viewport_width(panel: QWidget) -> int | None:
    widget = panel.parentWidget()
    while widget is not None:
        if isinstance(widget, QScrollArea):
            return widget.viewport().width()
        widget = widget.parentWidget()
    return None


def _available_preview_width(panel: QWidget) -> int:
    """Fit the preview card; never grow with a wide settings column or window."""
    design = _design_preview_width_at_min_window()
    card_inner = _preview_card_inner_width(panel)
    viewport = _preview_scroll_viewport_width(panel)

    card_ready = card_inner is not None and card_inner >= _PREVIEW_LAYOUT_MIN_WIDTH
    viewport_ready = viewport is not None and viewport >= _PREVIEW_LAYOUT_MIN_WIDTH

    if not card_ready and not viewport_ready:
        return design

    candidates = [design]
    if card_ready:
        candidates.append(card_inner)  # type: ignore[arg-type]
    if viewport_ready:
        candidates.append(viewport)  # type: ignore[arg-type]
    return min(candidates)


def _preview_snapshot_width(panel: QWidget) -> int:
    return _available_preview_width(panel)


def _configure_live_preview_scene(widget: QWidget) -> None:
    """Read-only miniature shell shown directly in the settings card."""
    widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)


def _preview_scene_size(widget: QWidget, width: int) -> tuple[int, int]:
    """Return a stable (width, height) for an inline preview scene."""
    widget.setFixedWidth(width)
    widget.adjustSize()
    height = widget.sizeHint().height()
    if widget.hasHeightForWidth():
        height = max(height, widget.heightForWidth(width))
    return width, max(height, _PREVIEW_SNAPSHOT_HEIGHT)


class _ThemePreviewPanelBase(QFrame):
    """Shared width tracking for inline theme preview panels."""

    _last_preview_width: int
    _width_watch_card: QWidget | None

    def _install_preview_card_width_watch(self) -> None:
        card = _find_settings_section_card(self)
        if card is self._width_watch_card:
            return
        if self._width_watch_card is not None:
            self._width_watch_card.removeEventFilter(self)
        self._width_watch_card = card
        if card is not None:
            card.installEventFilter(self)

    def _request_preview_layout_refresh(self) -> None:
        if not self._has_pending_preview():
            width = _available_preview_width(self)
            previous = self._last_preview_width
            if previous and abs(width - previous) < 2:
                return
            self._last_preview_width = width
        QTimer.singleShot(0, self._repaint_preview)

    def _has_pending_preview(self) -> bool:
        raise NotImplementedError

    def _repaint_preview(self) -> None:
        raise NotImplementedError

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self._width_watch_card
            and event.type() == QEvent.Type.Resize
        ):
            self._request_preview_layout_refresh()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._request_preview_layout_refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._install_preview_card_width_watch()
        self._request_preview_layout_refresh()


class ThemePreviewPanel(_ThemePreviewPanelBase):
    """Live draft preview — Conversations shell with chat wallpaper draft."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        design_width = _design_preview_width_at_min_window()
        self.setMaximumWidth(design_width)
        self.setFixedWidth(design_width)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._conversations_live = ThemeConversationsPreviewScene(parent=self)
        _configure_live_preview_scene(self._conversations_live)
        layout.addWidget(self._conversations_live)
        self.setMinimumHeight(_PREVIEW_SNAPSHOT_HEIGHT)

        self._pending_conversations: tuple | None = None
        self._last_preview_width = 0
        self._width_watch_card = None

    @property
    def _conversations_view(self):
        """Backward-compatible alias for tests that referenced the snapshot label."""
        return self._conversations_live

    def _has_pending_preview(self) -> bool:
        return self._pending_conversations is not None

    def _repaint_preview(self) -> None:
        pending = self._pending_conversations
        if pending is None:
            return
        resolved, chat_profile, chat_resolved_wallpaper = pending
        width = _preview_snapshot_width(self)
        self._conversations_live.setFixedSize(width, _PREVIEW_SNAPSHOT_HEIGHT)
        self._conversations_live.apply_theme(
            resolved,
            chat_profile=chat_profile,
            chat_resolved_wallpaper=chat_resolved_wallpaper,
        )
        height = max(_PREVIEW_SNAPSHOT_HEIGHT, self._conversations_live.height())
        self.setFixedSize(width, height)
        self.updateGeometry()
        self._last_preview_width = width

    def apply_theme(
        self,
        resolved: ResolvedTheme,
        *,
        chat_profile: SurfaceProfile | None = None,
        chat_resolved_wallpaper=None,
    ) -> None:
        """Repaint preview chrome from ``resolved`` only — never touches the app."""
        self.setStyleSheet(
            "QFrame#ThemePreviewPanel { background: transparent; border: none; }"
        )
        self._pending_conversations = (
            resolved,
            chat_profile,
            chat_resolved_wallpaper,
        )
        self._last_preview_width = 0
        QTimer.singleShot(0, self._repaint_preview)


class ThemeComponentsPreviewPanel(_ThemePreviewPanelBase):
    """Live draft preview for theme colors on a miniature Settings page shell."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemeComponentsPreviewPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        design_width = _design_preview_width_at_min_window()
        self.setMaximumWidth(design_width)
        self.setFixedWidth(design_width)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._components_live = ThemeComponentsPreviewScene(parent=self)
        _configure_live_preview_scene(self._components_live)
        layout.addWidget(self._components_live)
        self.setMinimumHeight(_PREVIEW_SNAPSHOT_HEIGHT)

        self._pending_components: ResolvedTheme | None = None
        self._last_preview_width = 0
        self._width_watch_card = None

    @property
    def _components_scene(self) -> ThemeComponentsPreviewScene:
        """Backward-compatible alias for tests and callers."""
        return self._components_live

    @property
    def _components_view(self):
        """Backward-compatible alias for tests that referenced the snapshot label."""
        return self._components_live

    def _has_pending_preview(self) -> bool:
        return self._pending_components is not None

    def _repaint_preview(self) -> None:
        resolved = self._pending_components
        if resolved is None:
            return
        width = _preview_snapshot_width(self)
        self._components_live.apply_theme(resolved)
        snap_width, snap_height = _preview_scene_size(self._components_live, width)
        self._components_live.setFixedSize(snap_width, snap_height)
        self.setFixedSize(snap_width, snap_height)
        self.updateGeometry()
        self._last_preview_width = snap_width

    def apply_theme(self, resolved: ResolvedTheme) -> None:
        """Repaint component mock from ``resolved`` only — never touches the app."""
        self.setStyleSheet(
            "QFrame#ThemeComponentsPreviewPanel { background: transparent; border: none; }"
        )
        self._pending_components = resolved
        self._last_preview_width = 0
        QTimer.singleShot(0, self._repaint_preview)


class ThemeLibraryPreviewPanel(_ThemePreviewPanelBase):
    """Live draft preview for Library document wallpaper settings."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemeLibraryPreviewPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        design_width = _design_preview_width_at_min_window()
        self.setMaximumWidth(design_width)
        self.setFixedWidth(design_width)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._live = ThemeLibraryPreviewScene(parent=self)
        _configure_live_preview_scene(self._live)
        layout.addWidget(self._live)
        self.setMinimumHeight(_PREVIEW_SNAPSHOT_HEIGHT)

        self._pending_library: tuple | None = None
        self._last_preview_width = 0
        self._width_watch_card = None

    @property
    def _view(self):
        """Backward-compatible alias for tests that referenced the snapshot label."""
        return self._live

    def _has_pending_preview(self) -> bool:
        return self._pending_library is not None

    def _repaint_preview(self) -> None:
        pending = self._pending_library
        if pending is None:
            return
        resolved, library_profile, library_resolved_wallpaper = pending
        width = _preview_snapshot_width(self)
        self._live.setFixedSize(width, _PREVIEW_SNAPSHOT_HEIGHT)
        self._live.apply_theme(
            resolved,
            library_profile=library_profile,
            library_resolved_wallpaper=library_resolved_wallpaper,
        )
        height = max(_PREVIEW_SNAPSHOT_HEIGHT, self._live.height())
        self.setFixedSize(width, height)
        self.updateGeometry()
        self._last_preview_width = width

    def apply_theme(
        self,
        resolved: ResolvedTheme,
        *,
        library_profile: SurfaceProfile | None = None,
        library_resolved_wallpaper=None,
    ) -> None:
        """Repaint library preview from draft theme and wallpaper only."""
        self.setStyleSheet(
            "QFrame#ThemeLibraryPreviewPanel { background: transparent; border: none; }"
        )
        self._pending_library = (
            resolved,
            library_profile,
            library_resolved_wallpaper,
        )
        self._last_preview_width = 0
        QTimer.singleShot(0, self._repaint_preview)
