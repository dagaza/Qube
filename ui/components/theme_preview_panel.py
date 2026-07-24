"""Isolated theme preview widgets for Settings → Themes (no global apply)."""

from __future__ import annotations

import qtawesome as qta
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.brand_identity import BRAND_LOGO_STROKE_HEX
from core.theme.color_utils import with_alpha
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import (
    AGENT_MESSAGE_SHELL,
    LIST_SURFACE,
    RAG_INDICATOR_STANDBY,
    SETTINGS_CHECKBOX,
    SETTINGS_FORM_CONTROLS,
    SETTINGS_LINE_EDIT,
    SETTINGS_SECTION_CARD,
    STAGE_SURFACE,
    TOGGLE_BUTTON,
    USER_BUBBLE_FRAME,
    USER_BUBBLE_LABEL,
    WEB_INDICATOR_STANDBY,
)
from ui.components.brand_buttons import apply_brand_danger, apply_brand_primary
from ui.components.selector_button import SelectorButton
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
        self._nav.setFixedWidth(34)
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
        self._history.setFixedWidth(96)
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

        self._chat = QFrame()
        self._chat.setObjectName("ThemePreviewChatStage")
        chat_layout = QVBoxLayout(self._chat)
        chat_layout.setContentsMargins(10, 8, 10, 8)
        chat_layout.setSpacing(6)

        self._agent_block = QFrame()
        self._agent_block.setObjectName("ThemePreviewAgentBlock")
        agent_layout = QVBoxLayout(self._agent_block)
        agent_layout.setContentsMargins(0, 0, 0, 0)
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
        body.addWidget(self._chat, stretch=1)

        self._tools = QFrame()
        self._tools.setObjectName("ThemePreviewToolsPane")
        self._tools.setFixedWidth(108)
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

    def apply_theme(self, resolved: ResolvedTheme) -> None:
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

        chat_bg = colors["chat_stage"]
        self._chat.setStyleSheet(
            f"QFrame#ThemePreviewChatStage {{ background-color: {chat_bg}; border: none; }}"
        )
        self._agent_block.setStyleSheet("background: transparent; border: none;")
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
    """Settings / Memory-style controls not shown on the Conversations mockup."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewComponentsScene")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        hint = QLabel(
            "Form controls, dialogs, and status chips from Settings, Memory, and other views."
        )
        hint.setWordWrap(True)
        self._hint = hint
        layout.addWidget(hint)

        card = QFrame()
        self._components_card = card
        card.setObjectName("ThemePreviewComponentsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(8)

        card_title = QLabel("Sample settings block")
        self._card_title = card_title
        card_layout.addWidget(card_title)

        self._checkbox = QCheckBox("Enable sample feature")
        self._checkbox.setChecked(True)
        card_layout.addWidget(self._checkbox)

        form_row = QHBoxLayout()
        self._line_edit = QLineEdit("Editable text field")
        self._line_edit.setReadOnly(True)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(0, 100)
        self._spin.setValue(42)
        self._spin.setFixedWidth(72)
        form_row.addWidget(self._line_edit, stretch=1)
        form_row.addWidget(self._spin)
        card_layout.addLayout(form_row)

        self._selector = SelectorButton("Category ▾", parent=self)
        card_layout.addWidget(self._selector)

        btn_row = QHBoxLayout()
        self._primary_btn = QPushButton("Primary action")
        self._danger_btn = QPushButton("Delete")
        btn_row.addWidget(self._primary_btn)
        btn_row.addWidget(self._danger_btn)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)
        layout.addWidget(card)

        memory_card = QFrame()
        self._memory_card = memory_card
        memory_card.setObjectName("ThemePreviewMemoryCard")
        memory_layout = QHBoxLayout(memory_card)
        memory_layout.setContentsMargins(10, 8, 10, 8)
        memory_layout.setSpacing(8)
        memory_text = QVBoxLayout()
        memory_text.setSpacing(2)
        self._memory_title = QLabel("User prefers concise answers")
        self._memory_meta = QLabel("preference · enriched yesterday")
        memory_text.addWidget(self._memory_title)
        memory_text.addWidget(self._memory_meta)
        memory_layout.addLayout(memory_text, stretch=1)
        self._memory_edit_btn = QPushButton("Edit")
        memory_layout.addWidget(self._memory_edit_btn)
        layout.addWidget(memory_card)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self._status_idle = QLabel(" IDLE")
        self._status_speaking = QLabel(" SPEAKING")
        self._status_needs = QLabel(" NEEDS MODEL")
        for lbl in (self._status_idle, self._status_speaking, self._status_needs):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedHeight(20)
            status_row.addWidget(lbl)
        status_row.addStretch()
        layout.addLayout(status_row)

        self._tooltip_sample = QLabel("Tooltip sample")
        layout.addWidget(self._tooltip_sample)
        layout.addStretch()

    def apply_theme(self, resolved: ResolvedTheme) -> None:
        self._hint.setStyleSheet(
            f"color: {resolved.text_secondary}; font-size: 11px;"
            f" background: transparent; border: none;"
        )
        card_style = resolved.style(SETTINGS_SECTION_CARD, object_name="ThemePreviewComponentsCard")
        self.setStyleSheet(
            f"QFrame#ThemePreviewComponentsScene {{ background: transparent; border: none; }}"
        )
        self._components_card.setStyleSheet(card_style)
        self._card_title.setStyleSheet(
            f"color: {resolved.text_primary}; font-weight: 700; font-size: 11px;"
            f" background: transparent; border: none;"
        )
        self._checkbox.setStyleSheet(resolved.style(SETTINGS_CHECKBOX))
        self._line_edit.setStyleSheet(resolved.style(SETTINGS_LINE_EDIT))
        self._spin.setStyleSheet(resolved.style(SETTINGS_FORM_CONTROLS))
        self._selector.setText("Category ▾")
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
            f"color: {resolved.text_primary}; font-size: 11px; font-weight: 600;"
            f" background: transparent; border: none;"
        )
        self._memory_meta.setStyleSheet(
            f"color: {resolved.text_muted}; font-size: 10px;"
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
            f" border-radius: 6px; padding: 4px 8px; font-size: 10px;"
        )


class ThemePreviewPanel(QFrame):
    """Live draft preview — Conversations shell by default, optional components view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ThemePreviewPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(12)
        self._scene_group = QButtonGroup(self)
        self._scene_group.setExclusive(True)
        self._conversations_cb = QCheckBox("Conversations")
        self._conversations_cb.setChecked(True)
        self._components_cb = QCheckBox("More components")
        for cb in (self._conversations_cb, self._components_cb):
            self._scene_group.addButton(cb)
            toggle_row.addWidget(cb)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        self._stack = QStackedWidget()
        self._conversations_scene = ThemeConversationsPreviewScene()
        self._components_scene = ThemeComponentsPreviewScene()
        self._stack.addWidget(self._conversations_scene)
        self._stack.addWidget(self._components_scene)
        self._stack.setMinimumHeight(280)
        layout.addWidget(self._stack)

        self._conversations_cb.toggled.connect(self._on_scene_toggled)
        self._components_cb.toggled.connect(self._on_scene_toggled)

    def _on_scene_toggled(self, _checked: bool) -> None:
        if self._components_cb.isChecked():
            self._stack.setCurrentWidget(self._components_scene)
        else:
            self._stack.setCurrentWidget(self._conversations_scene)

    def apply_theme(self, resolved: ResolvedTheme) -> None:
        """Repaint preview chrome from ``resolved`` only — never touches the app."""
        self.setStyleSheet(
            "QFrame#ThemePreviewPanel { background: transparent; border: none; }"
        )
        self._stack.setStyleSheet("background: transparent; border: none;")
        self._conversations_scene.apply_theme(resolved)
        self._components_scene.apply_theme(resolved)
        toggle_style = resolved.style(SETTINGS_CHECKBOX)
        self._conversations_cb.setStyleSheet(toggle_style)
        self._components_cb.setStyleSheet(toggle_style)
