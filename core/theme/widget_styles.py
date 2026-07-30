"""Reusable QSS fragments and style roles for widget-level theming."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.theme.color_utils import adjust_lightness, with_alpha
from core.theme.link_styles import link_anchor_css
from core.brand_identity import (
    BRAND_HUB_OFFICIAL_BADGE_FG_DARK,
    BRAND_HUB_OFFICIAL_BADGE_FG_LIGHT,
    BRAND_TELEMETRY_CPU_HEX,
    BRAND_TELEMETRY_GPU_HEX,
    BRAND_TELEMETRY_RAM_HEX,
    BRAND_WEB_INDICATOR_STANDBY_HEX,
)

if TYPE_CHECKING:
    from core.theme.tokens import ResolvedTheme

# Style role names (use with ``ResolvedTheme.style(role)``).
GHOST_ICON_BUTTON = "ghost_icon_button"
UTILITY_ICON_BUTTON = "utility_icon_button"
COMPOSER_SIDE_BUTTON = "composer_side_button"
COMPOSER_SIDE_DIVIDER = "composer_side_divider"
MUTED_LABEL = "muted_label"
TELEMETRY_LABEL = "telemetry_label"
HELP_ACTION_CHIP = "help_action_chip"
USER_BUBBLE_FRAME = "user_bubble_frame"
USER_BUBBLE_LABEL = "user_bubble_label"
AGENT_MESSAGE_SHELL = "agent_message_shell"
AGENT_MESSAGE_FRAME = "agent_message_frame"
AGENT_COPY_BUTTON = "agent_copy_button"
QUBE_RESPONSE_HEADER = "qube_response_header"
PLACEHOLDER_MUTED = "placeholder_muted"
LIST_SURFACE = "list_surface"
STAGE_SURFACE = "stage_surface"
COMBO_POPUP_LIST = "combo_popup_list"
COMBO_POPUP_VIEWPORT = "combo_popup_viewport"
COMBO_POPUP_SHELL = "combo_popup_shell"
META_LABEL = "meta_label"
META_HINT = "meta_hint"
HUB_MUTED_ROW = "hub_muted_row"
HUB_MUTED_HINT = "hub_muted_hint"
CAPABILITY_CHIP = "capability_chip"
ACCENT_CHIP = "accent_chip"
QUANT_BADGE_PRIMARY = "quant_badge_primary"
QUANT_BADGE_SECONDARY = "quant_badge_secondary"
DIVIDER_ACCENT = "divider_accent"
READABILITY_FONT_PAIR = "readability_font_pair"
TRANSPARENT_FRAME = "transparent_frame"
TRANSPARENT_TEXT_PREVIEW = "transparent_text_preview"
CONNECTIVITY_ERROR_BANNER = "connectivity_error_banner"
TOGGLE_BUTTON = "toggle_button"
CHAT_WITH_DOC_FAB = "chat_with_doc_fab"
HIGH_CONTRAST_MARKDOWN = "high_contrast_markdown"
PRESTIGE_DIALOG_SHELL = "prestige_dialog_shell"
PRESTIGE_DIALOG_CONTAINER = "prestige_dialog_container"
PRESTIGE_DIALOG_TITLE = "prestige_dialog_title"
PRESTIGE_DIALOG_MESSAGE = "prestige_dialog_message"
PRESTIGE_DIALOG_INPUT = "prestige_dialog_input"
PRESTIGE_DIALOG_CANCEL = "prestige_dialog_cancel"
PRESTIGE_DIALOG_CONFIRM = "prestige_dialog_confirm"
PRESTIGE_DIALOG_MODE_OPTION = "prestige_dialog_mode_option"
PRESTIGE_GHOST_BUTTON = "prestige_ghost_button"
PRESTIGE_SOURCE_CONTAINER = "prestige_source_container"
PRESTIGE_TEXT_VIEW = "prestige_text_view"
PRESTIGE_CITATIONS_CONTAINER = "prestige_citations_container"
PRESTIGE_CITATION_ROW = "prestige_citation_row"
PRESTIGE_MUTED_LABEL = "prestige_muted_label"
PRESTIGE_ACCENT_LABEL = "prestige_accent_label"
PRESTIGE_BODY_LABEL = "prestige_body_label"
PRESTIGE_LINK_LABEL = "prestige_link_label"
PRESTIGE_DIALOG_LIST = "prestige_dialog_list"
SETTINGS_SECTION_CARD = "settings_section_card"
SETTINGS_FORM_CONTROLS = "settings_form_controls"
SETTINGS_CHECKBOX = "settings_checkbox"
SETTINGS_LINE_EDIT = "settings_line_edit"
SETTINGS_SLIDER = "settings_slider"
SETTINGS_BORDERED_LIST = "settings_bordered_list"
SETTINGS_LABEL = "settings_label"
SETTINGS_ICON_BUTTON = "settings_icon_button"
SETTINGS_DANGER_GHOST_BUTTON = "settings_danger_ghost_button"
SETTINGS_PRESTIGE_MENU = "settings_prestige_menu"
SETTINGS_WARNING_LABEL = "settings_warning_label"
SETTINGS_HINT = "settings_hint"
SETTINGS_GHOST_TOOL_BUTTON = "settings_ghost_tool_button"
SETTINGS_BORDERLESS_TABLE = "settings_borderless_table"
KNOWLEDGE_ACCESS_BADGE = "knowledge_access_badge"
KNOWLEDGE_ACCESS_HINT = "knowledge_access_hint"
KNOWLEDGE_ACTION_BUTTON = "knowledge_action_button"
KNOWLEDGE_SETUP_CALLOUT = "knowledge_setup_callout"
KNOWLEDGE_SETUP_CALLOUT_TITLE = "knowledge_setup_callout_title"
KNOWLEDGE_SETUP_CALLOUT_BODY = "knowledge_setup_callout_body"
KNOWLEDGE_SETUP_CALLOUT_DISMISS = "knowledge_setup_callout_dismiss"
DISCOVERY_PROVIDER_CARD = "discovery_provider_card"
DISCOVERY_ROLE_CHIP = "discovery_role_chip"
DISCOVERY_PROVIDER_NAME = "discovery_provider_name"
DISCOVERY_PRIVACY_CHIP = "discovery_privacy_chip"
DISCOVERY_BODY_TEXT = "discovery_body_text"
DISCOVERY_INFO_CARD = "discovery_info_card"
DISCOVERY_INFO_TITLE = "discovery_info_title"
DISCOVERY_INFO_HIGHLIGHT = "discovery_info_highlight"
DISCOVERY_INFO_KV_KEY = "discovery_info_kv_key"
DISCOVERY_INFO_KV_VALUE = "discovery_info_kv_value"
DISCOVERY_INFO_BULLET = "discovery_info_bullet"
DISCOVERY_INFO_STATUS = "discovery_info_status"
DISCOVERY_DIVIDER = "discovery_divider"
PROVIDER_STATUS_TABLE = "provider_status_table"
ONBOARDING_COACH_PANEL = "onboarding_coach_panel"

# Color-only roles (use with ``ResolvedTheme.color(role)``).
ACCENT_ICON = "accent_icon"
ACCENT_ICON_ACTIVE = "accent_icon_active"
MUTED_ICON = "muted_icon"
LINK_ICON = "link_icon"
DANGER_ICON = "danger_icon"
SUCCESS_STATUS = "success_status"
WARNING_STATUS = "warning_status"
MUTED_STATUS = "muted_status"
SETTINGS_NAV_ICON = "settings_nav_icon"
SETTINGS_CHEVRON_ENABLED = "settings_chevron_enabled"
SETTINGS_CHEVRON_DISABLED = "settings_chevron_disabled"
SETTINGS_DIVIDER = "settings_divider"
MODEL_HUB_OFFICIAL_BADGE = "model_hub_official_badge"
ONBOARDING_SPOTLIGHT_RING = "onboarding_spotlight_ring"
SIDEBAR_ACTION_ICON = "sidebar_action_icon"
RETRIEVAL_INDICATOR_OFF = "retrieval_indicator_off"
RETRIEVAL_INDICATOR_ACTIVE = "retrieval_indicator_active"
RAG_INDICATOR_STANDBY = "rag_indicator_standby"
WEB_INDICATOR_STANDBY = "web_indicator_standby"
NAV_ICON_ACTIVE = "nav_icon_active"
NAV_ICON_INACTIVE = "nav_icon_inactive"
TELEMETRY_CPU = "telemetry_cpu"
TELEMETRY_RAM = "telemetry_ram"
TELEMETRY_GPU = "telemetry_gpu"
THEME_TOGGLE_MOON = "theme_toggle_moon"
THEME_TOGGLE_SUN = "theme_toggle_sun"


def settings_prestige_menu_palette(resolved: "ResolvedTheme") -> dict[str, str]:
    """Hex colors for Settings prestige dropdown menus (QPalette + QSS)."""
    sel_bg = resolved.surface_elevated if resolved.is_dark else resolved.surface
    sel_fg = resolved.text_primary if resolved.is_dark else resolved.text_primary
    return {
        "bg": resolved.background,
        "fg": resolved.text_primary,
        "sel_bg": sel_bg,
        "sel_fg": sel_fg,
        "border": resolved.border_subtle if resolved.is_dark else resolved.border,
        "hover": resolved.surface_hover if resolved.is_dark else resolved.surface,
    }


def prestige_accent_colors(
    resolved: ResolvedTheme,
    *,
    tone: str = "default",
    title: str = "",
) -> tuple[str, str]:
    """Return ``(accent, confirm_foreground)`` for Prestige dialog chrome."""
    tone_key = str(tone or "default").lower().strip()
    if tone_key == "danger":
        return resolved.error, resolved.brand_fg
    if "Delete" in str(title):
        return resolved.error, resolved.text_on_accent
    return resolved.link, resolved.text_on_accent


def _discovery_role_accent(resolved: ResolvedTheme, role: str) -> tuple[str, str, str]:
    """Return ``(accent, bg_tint, border)`` for discovery card roles."""
    role_key = role if role in ("primary", "fallback", "optional") else "fallback"
    if role_key == "primary":
        accent = resolved.link
    elif role_key == "optional":
        accent = resolved.accent_secondary if hasattr(resolved, "accent_secondary") else resolved.accent
    else:
        accent = resolved.warning
    bg_tint = with_alpha(accent, 0.12 if resolved.is_dark else 0.08)
    border = with_alpha(accent, 0.35 if resolved.is_dark else 0.28)
    return accent, bg_tint, border


def _knowledge_access_colors(resolved: ResolvedTheme, access: str) -> tuple[str, str, str]:
    """Return ``(foreground, background, border)`` for access badge pills."""
    surface = resolved.surface_elevated if resolved.is_dark else resolved.background
    mapping = {
        "free": (resolved.text_secondary, surface, resolved.border),
        "optional_key": (resolved.warning, surface, adjust_lightness(resolved.warning, -0.12)),
        "key_required": (resolved.error, surface, adjust_lightness(resolved.error, -0.12)),
        "connected": (resolved.success, surface, adjust_lightness(resolved.success, -0.12)),
        "env_override": (resolved.link, surface, adjust_lightness(resolved.link, -0.12)),
        "coming_soon": (resolved.text_muted, surface, resolved.border),
    }
    return mapping.get(access, mapping["coming_soon"])


def theme_color(resolved: ResolvedTheme, role: str) -> str:
    if role == ACCENT_ICON:
        return resolved.accent
    if role == ACCENT_ICON_ACTIVE:
        return resolved.accent_hover
    if role == MUTED_ICON:
        return resolved.text_secondary
    if role == LINK_ICON:
        return resolved.link
    if role == DANGER_ICON:
        return resolved.error
    if role == SUCCESS_STATUS:
        return resolved.success
    if role == WARNING_STATUS:
        return resolved.warning
    if role == MUTED_STATUS:
        return resolved.text_muted
    if role == LIST_SURFACE:
        return resolved.sidebar_surface
    if role == STAGE_SURFACE:
        return resolved.background
    if role == QUBE_RESPONSE_HEADER:
        return resolved.chat_header
    if role == PLACEHOLDER_MUTED:
        return resolved.text_muted
    if role == SETTINGS_NAV_ICON:
        return resolved.accent if resolved.is_dark else resolved.text_secondary
    if role == SETTINGS_CHEVRON_ENABLED:
        return resolved.text_secondary
    if role == SETTINGS_CHEVRON_DISABLED:
        return resolved.text_muted
    if role == SETTINGS_DIVIDER:
        return resolved.border_subtle if resolved.is_dark else resolved.border
    if role == MODEL_HUB_OFFICIAL_BADGE:
        return (
            BRAND_HUB_OFFICIAL_BADGE_FG_DARK
            if resolved.is_dark
            else BRAND_HUB_OFFICIAL_BADGE_FG_LIGHT
        )
    if role == ONBOARDING_SPOTLIGHT_RING:
        return resolved.link
    if role == SIDEBAR_ACTION_ICON:
        return resolved.link
    if role == RETRIEVAL_INDICATOR_OFF:
        return resolved.text_muted
    if role == RETRIEVAL_INDICATOR_ACTIVE:
        return resolved.success
    if role == RAG_INDICATOR_STANDBY:
        return resolved.link
    if role == WEB_INDICATOR_STANDBY:
        return BRAND_WEB_INDICATOR_STANDBY_HEX
    if role == NAV_ICON_ACTIVE:
        return resolved.link
    if role == NAV_ICON_INACTIVE:
        return resolved.text_primary if resolved.is_dark else resolved.text_secondary
    if role == TELEMETRY_CPU:
        return BRAND_TELEMETRY_CPU_HEX
    if role == TELEMETRY_RAM:
        return BRAND_TELEMETRY_RAM_HEX
    if role == TELEMETRY_GPU:
        return BRAND_TELEMETRY_GPU_HEX
    if role == THEME_TOGGLE_MOON:
        return resolved.warning
    if role == THEME_TOGGLE_SUN:
        return adjust_lightness(resolved.error, 0.12)
    raise ValueError(f"Unknown theme color role: {role!r}")


def _user_bubble_frame(resolved: ResolvedTheme, *, high_contrast: bool) -> str:
    if high_contrast:
        return adjust_lightness(resolved.chat_user_bubble, 0.08)
    return resolved.chat_user_bubble


def _user_bubble_text(resolved: ResolvedTheme, *, high_contrast: bool) -> str:
    if high_contrast:
        return resolved.background if resolved.is_dark else "#000000"
    return resolved.chat_user_text


def _agent_message_frame_bg(resolved: ResolvedTheme, *, high_contrast: bool) -> str:
    if high_contrast:
        return _user_bubble_frame(resolved, high_contrast=True)
    return resolved.surface_elevated


def theme_style(resolved: ResolvedTheme, role: str, **kwargs) -> str:
    if role == GHOST_ICON_BUTTON:
        padding = kwargs.get("padding", "6px")
        return f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                padding: {padding};
            }}
            QPushButton:hover {{
                background-color: {resolved.surface_hover};
            }}
            QPushButton:pressed {{
                background-color: {resolved.surface_pressed};
            }}
            QPushButton:disabled {{
                opacity: 0.45;
            }}
        """
    if role == UTILITY_ICON_BUTTON:
        return f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {resolved.surface_hover};
            }}
            QPushButton:disabled {{
                opacity: 0.45;
            }}
        """
    if role == COMPOSER_SIDE_BUTTON:
        return f"""
            QPushButton#ComposerAttachButton,
            QPushButton#ComposerVoiceButton {{
                background: transparent;
                border: none;
                border-radius: 8px;
                padding: 4px;
            }}
            QPushButton#ComposerAttachButton:hover,
            QPushButton#ComposerVoiceButton:hover {{
                background-color: {resolved.surface_hover};
            }}
            QPushButton#ComposerAttachButton:disabled,
            QPushButton#ComposerVoiceButton:disabled {{
                opacity: 0.45;
            }}
        """
    if role == COMPOSER_SIDE_DIVIDER:
        line = resolved.border_subtle if resolved.is_dark else resolved.border
        return f"QFrame#ComposerSideDivider {{ background-color: {line}; border: none; }}"
    if role == MUTED_LABEL:
        return f"color: {resolved.text_muted}; background: transparent; border: none;"
    if role == TELEMETRY_LABEL:
        return (
            f"color: {resolved.text_secondary}; font-size: 9px; "
            f"background: transparent; border: none; padding: 0px 4px;"
        )
    if role == HELP_ACTION_CHIP:
        fg = resolved.link
        return f"""
            QPushButton {{
                color: {fg};
                background: transparent;
                border: 1px solid {fg};
                border-radius: 10px;
                padding: 2px 8px;
                font-size: 9pt;
            }}
            QPushButton:hover {{
                background-color: {resolved.accent_muted_bg};
            }}
        """
    if role == USER_BUBBLE_FRAME:
        high_contrast = bool(kwargs.get("high_contrast", False))
        bg = _user_bubble_frame(resolved, high_contrast=high_contrast)
        return f"background-color: {bg}; border-radius: 18px;"
    if role == USER_BUBBLE_LABEL:
        high_contrast = bool(kwargs.get("high_contrast", False))
        font_pt = kwargs.get("font_pt", 12.0)
        fg = _user_bubble_text(resolved, high_contrast=high_contrast)
        return (
            f"background: transparent; border: none; padding: 0px; "
            f"font-size: {float(font_pt):.1f}pt; color: {fg};"
        )
    if role == AGENT_MESSAGE_SHELL:
        font_pt = kwargs.get("font_pt", 12.0)
        return (
            f"font-size: {float(font_pt):.1f}pt; background: transparent; "
            f"border: none; padding: 0px;"
        )
    if role == AGENT_MESSAGE_FRAME:
        if not kwargs.get("enabled", False):
            object_name = kwargs.get("object_name", "AgentMessageContainer")
            return f"QFrame#{object_name} {{ background: transparent; border: none; }}"
        high_contrast = bool(kwargs.get("high_contrast", False))
        bg = _agent_message_frame_bg(resolved, high_contrast=high_contrast)
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        object_name = kwargs.get("object_name", "AgentMessageContainer")
        return (
            f"QFrame#{object_name} {{"
            f" background-color: {bg};"
            f" border: 1px solid {border};"
            f" border-radius: 12px;"
            f" }}"
        )
    if role == AGENT_COPY_BUTTON:
        return f"""
            QPushButton::menu-indicator {{ image: none; width: 0px; }}
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {resolved.surface_hover};
            }}
        """
    if role == META_LABEL:
        strong = bool(kwargs.get("strong", False))
        fg = resolved.text_primary if strong else resolved.text_muted
        weight = "700" if strong else "400"
        return f"color: {fg}; font-weight: {weight}; background: transparent; border: none;"
    if role == META_HINT:
        return f"color: {resolved.text_secondary}; background: transparent; border: none;"
    if role == HUB_MUTED_ROW:
        return (
            f"color: {resolved.text_muted}; background: transparent; border: none; "
            f"font-size: 11px; font-weight: 500;"
        )
    if role == HUB_MUTED_HINT:
        return f"color: {resolved.text_muted}; font-size: 11px;"
    if role == CAPABILITY_CHIP:
        return (
            f"QFrame {{ border: 1px solid {resolved.accent}; border-radius: 10px; background: transparent; }}"
            f"QLabel[class='ChipLabel'], QLabel[class='ChipIcon'] {{"
            f" color: {resolved.text_primary}; background: transparent; border: none; }}"
        )
    if role == ACCENT_CHIP:
        return f"""
            QLabel {{
                color: {resolved.text_primary};
                background: {resolved.accent_muted_bg};
                border: 1px solid {resolved.accent};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 600;
            }}
        """
    if role == QUANT_BADGE_PRIMARY:
        bg = with_alpha(resolved.accent, 0.35 if resolved.is_dark else 0.22)
        fg = adjust_lightness(resolved.accent, 0.25 if resolved.is_dark else -0.35)
        border = with_alpha(resolved.accent, 0.55)
        return (
            f"QLabel {{ background-color: {bg}; color: {fg}; border: 1px solid {border};"
            f" border-radius: 9px; padding: 3px 10px; font-size: 11px; font-weight: 600; }}"
        )
    if role == QUANT_BADGE_SECONDARY:
        bg = with_alpha(resolved.text_muted, 0.22 if resolved.is_dark else 0.14)
        fg = resolved.text_secondary
        border = with_alpha(resolved.text_muted, 0.35)
        return (
            f"QLabel {{ background-color: {bg}; color: {fg}; border: 1px solid {border};"
            f" border-radius: 9px; padding: 3px 10px; font-size: 11px; font-weight: 600; }}"
        )
    if role == DIVIDER_ACCENT:
        line = with_alpha(resolved.accent, 0.45)
        return (
            f"QFrame {{ border: none; background: {line}; "
            f"min-height: 1px; max-height: 1px; }}"
        )
    if role == READABILITY_FONT_PAIR:
        button_px = int(kwargs.get("button_px", 30))
        disabled = resolved.text_muted
        return f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
                padding: 2px 4px;
                color: {resolved.accent};
                font-weight: 700;
                font-size: 13px;
                min-width: {button_px}px;
                max-width: {button_px}px;
                min-height: {button_px}px;
                max-height: {button_px}px;
            }}
            QPushButton:hover {{ background-color: {resolved.surface_hover}; }}
            QPushButton:disabled {{ color: {disabled}; }}
        """
    if role == TRANSPARENT_FRAME:
        return "background: transparent; border: none;"
    if role == TRANSPARENT_TEXT_PREVIEW:
        fg = kwargs.get("color", resolved.text_primary)
        font_pt = float(kwargs.get("font_pt", 12.0))
        return (
            f"background: transparent; border: none; color: {fg}; "
            f"font-size: {font_pt:.1f}pt;"
        )
    if role == CONNECTIVITY_ERROR_BANNER:
        bg = adjust_lightness(resolved.error, 0.15 if resolved.is_dark else 0.35)
        border = adjust_lightness(resolved.error, 0.05 if resolved.is_dark else 0.25)
        fg = adjust_lightness(resolved.error, 0.35 if resolved.is_dark else -0.25)
        object_name = kwargs.get("object_name", "")
        selector = f"QFrame#{object_name}" if object_name else "QFrame"
        return f"""
            {selector} {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QLabel {{
                color: {fg};
                background: transparent;
                border: none;
                font-size: 12px;
            }}
        """
    if role == TOGGLE_BUTTON:
        checked = bool(kwargs.get("checked", False))
        active_bg = kwargs.get("active_bg", resolved.link)
        if checked:
            fg = resolved.text_on_accent
            bg = active_bg
            border = active_bg
        else:
            fg = resolved.text_primary
            bg = resolved.surface_pressed
            border = resolved.text_secondary if resolved.is_dark else resolved.border
        hover = resolved.surface_hover
        return f"""
            QPushButton {{
                color: {fg};
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """
    if role == CHAT_WITH_DOC_FAB:
        radius = int(kwargs.get("radius", 26))
        object_name = kwargs.get("object_name", "LibraryChatWithDocFab")
        hover = adjust_lightness(resolved.accent, -0.08)
        pressed = adjust_lightness(resolved.accent, -0.16)
        return f"""
            QPushButton#{object_name} {{
                background-color: {resolved.accent};
                border: none;
                border-radius: {radius}px;
            }}
            QPushButton#{object_name}:hover {{
                background-color: {hover};
            }}
            QPushButton#{object_name}:pressed {{
                background-color: {pressed};
            }}
        """
    if role == COMBO_POPUP_LIST:
        selected_bg = resolved.surface_hover
        hover_bg = resolved.surface_elevated
        return f"""
            QAbstractItemView {{
                background-color: {resolved.background};
                color: {resolved.text_primary};
                border: none;
                outline: none;
            }}
            QAbstractItemView::item {{
                min-height: 32px;
                padding: 8px 12px;
                color: {resolved.text_primary};
            }}
            QAbstractItemView::item:selected {{
                background-color: {selected_bg};
                color: {resolved.text_on_accent if resolved.is_dark else resolved.text_primary};
            }}
            QAbstractItemView::item:hover {{
                background-color: {hover_bg};
                color: {resolved.text_primary};
            }}
        """
    if role == COMBO_POPUP_VIEWPORT:
        return f"background-color: {resolved.background};"
    if role == COMBO_POPUP_SHELL:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        return f"""
            QWidget {{
                background: {resolved.background};
                border: 1px solid {border};
                border-radius: 8px;
            }}
        """
    if role == HIGH_CONTRAST_MARKDOWN:
        if not kwargs.get("enabled", True):
            return ""
        high_contrast = bool(kwargs.get("high_contrast", False))
        fg = _user_bubble_text(resolved, high_contrast=high_contrast)
        code_bg = _user_bubble_frame(resolved, high_contrast=high_contrast)
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        hdr = (
            "h1 { font-size: 1.35em; font-weight: 700; margin-top: 0.45em; margin-bottom: 0.2em; }"
            "h2 { font-size: 1.2em; font-weight: 600; margin-top: 0.4em; margin-bottom: 0.18em; }"
            "h3 { font-size: 1.1em; font-weight: 600; margin-top: 0.35em; margin-bottom: 0.15em; }"
            "h4, h5, h6 { font-size: 1.05em; font-weight: 600; margin-top: 0.3em; margin-bottom: 0.12em; }"
        )
        return (
            f"body, p, span, div, li, ul, ol, dd, dt, "
            f"table, thead, tbody, tr, th, td, "
            f"blockquote, "
            f"h1, h2, h3, h4, h5, h6, strong, em {{ color: {fg}; }}"
            + link_anchor_css(resolved)
            + f"code, pre {{ background-color: {code_bg}; color: {fg}; }}"
            + f"table {{ border-color: {border}; }}"
            + f"th, td {{ border-color: {border}; border-width: 1px; border-style: solid; }}"
            + f"hr {{ border-color: {border}; color: {border}; }}"
            + hdr
        )
    if role == PRESTIGE_DIALOG_CONTAINER:
        accent = kwargs.get("accent", resolved.link)
        object_name = kwargs.get("object_name", "DialogContainer")
        return f"""
            QFrame#{object_name} {{
                background: {resolved.background};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel {{ color: {resolved.text_primary}; border: none; background: transparent; }}
        """
    if role == PRESTIGE_DIALOG_TITLE:
        accent = kwargs.get("accent", resolved.link)
        return (
            f"color: {accent}; font-weight: bold; font-size: 12px; "
            f"letter-spacing: 2px; background: transparent; border: none;"
        )
    if role == PRESTIGE_DIALOG_MESSAGE:
        size = kwargs.get("font_size", "15px")
        return (
            f"color: {resolved.text_primary}; font-size: {size}; line-height: 1.4; "
            f"background: transparent; border: none;"
        )
    if role == PRESTIGE_DIALOG_INPUT:
        accent = kwargs.get("accent", resolved.link)
        input_bg = resolved.surface_elevated
        return f"""
            QLineEdit {{
                background: {input_bg};
                color: {resolved.text_primary};
                border-radius: 10px;
                padding: 10px 15px;
                border: 1px solid {accent};
                font-size: 14px;
            }}
        """
    if role == PRESTIGE_DIALOG_CANCEL:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        btn_base = kwargs.get("btn_base", "")
        return (
            btn_base
            + f"""
            QPushButton {{
                color: {resolved.text_primary};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: {resolved.surface_pressed};
            }}
            QPushButton:disabled {{
                color: {resolved.text_muted};
                border: 1px solid {with_alpha(border, 0.45)};
                background: {with_alpha(resolved.text_muted, 0.10 if resolved.is_dark else 0.06)};
            }}
        """
        )
    if role == PRESTIGE_DIALOG_MODE_OPTION:
        btn_base = kwargs.get("btn_base", "")
        inactive = bool(kwargs.get("inactive", False))
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        if inactive:
            bg = with_alpha(resolved.text_muted, 0.20 if resolved.is_dark else 0.12)
            fg = resolved.text_muted
            border_color = with_alpha(resolved.text_muted, 0.32 if resolved.is_dark else 0.24)
            return (
                btn_base
                + f"""
            QPushButton {{
                color: {fg};
                border: 1px solid {border_color};
                background: {bg};
            }}
            QPushButton:disabled {{
                color: {fg};
                border: 1px solid {border_color};
                background: {bg};
            }}
        """
            )
        return (
            btn_base
            + f"""
            QPushButton {{
                color: {resolved.text_primary};
                border: 1px solid {border};
                background: {resolved.surface_elevated};
            }}
            QPushButton:hover {{
                background: {resolved.surface_pressed};
            }}
            QPushButton:pressed {{
                background: {resolved.surface_pressed};
            }}
        """
        )
    if role == PRESTIGE_DIALOG_CONFIRM:
        accent = kwargs.get("accent", resolved.link)
        confirm_fg = kwargs.get("confirm_fg", resolved.text_on_accent)
        btn_base = kwargs.get("btn_base", "")
        return (
            btn_base
            + f"""
            QPushButton {{
                background: {accent};
                color: {confirm_fg};
                border: none;
            }}
            QPushButton:hover {{
                background: {accent};
                opacity: 0.9;
            }}
        """
        )
    if role == PRESTIGE_GHOST_BUTTON:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        compact = bool(kwargs.get("compact", False))
        padding = "10px 14px" if compact else "12px 22px"
        min_h = "28px" if compact else "32px"
        radius = "10px" if compact else "12px"
        return f"""
            QPushButton {{
                padding: {padding};
                min-height: {min_h};
                border-radius: {radius};
                font-weight: bold;
                font-size: {"11px" if compact else "12px"};
                letter-spacing: {"0.5px" if compact else "1px"};
                color: {resolved.text_primary};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: {resolved.surface_pressed};
            }}
        """
    if role == PRESTIGE_SOURCE_CONTAINER:
        accent = kwargs.get("accent", resolved.link)
        object_name = kwargs.get("object_name", "SourcePreviewContainer")
        return f"""
            QFrame#{object_name} {{
                background: {resolved.background};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
    if role == PRESTIGE_CITATIONS_CONTAINER:
        accent = kwargs.get("accent", resolved.link)
        object_name = kwargs.get("object_name", "CitationSourcesContainer")
        return f"""
            QFrame#{object_name} {{
                background: {resolved.background};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
    if role == PRESTIGE_TEXT_VIEW:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        surface = resolved.surface_elevated if resolved.is_dark else resolved.surface
        return f"""
            QTextEdit {{
                background: {surface};
                color: {resolved.text_primary};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 14px 16px;
                font-size: 14px;
                line-height: 1.55;
            }}
        """
    if role == PRESTIGE_CITATION_ROW:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        surface = resolved.surface_elevated if resolved.is_dark else resolved.surface
        return f"""
            QFrame#CitationSourceRow {{
                background: {surface};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QFrame#CitationSourceRow:hover {{
                background: {resolved.surface_hover};
            }}
            QLabel {{ background: transparent; border: none; }}
        """
    if role == PRESTIGE_ACCENT_LABEL:
        accent = kwargs.get("accent", resolved.link)
        size = kwargs.get("font_size", "11px")
        spacing = kwargs.get("letter_spacing", "2px")
        weight = kwargs.get("font_weight", "bold")
        return (
            f"color: {accent}; font-weight: {weight}; font-size: {size}; "
            f"letter-spacing: {spacing}; background: transparent; border: none;"
        )
    if role == PRESTIGE_MUTED_LABEL:
        size = kwargs.get("font_size", "13px")
        weight = kwargs.get("font_weight", "400")
        italic = "font-style: italic;" if kwargs.get("italic") else ""
        return (
            f"color: {resolved.text_secondary}; font-size: {size}; font-weight: {weight}; "
            f"background: transparent; border: none; {italic}"
        )
    if role == PRESTIGE_BODY_LABEL:
        size = kwargs.get("font_size", "14px")
        weight = kwargs.get("font_weight", "600")
        return (
            f"color: {resolved.text_primary}; font-size: {size}; font-weight: {weight}; "
            f"background: transparent; border: none;"
        )
    if role == PRESTIGE_LINK_LABEL:
        size = kwargs.get("font_size", "12px")
        return (
            f"color: {resolved.link}; font-size: {size}; background: transparent; border: none;"
        )
    if role == SETTINGS_SECTION_CARD:
        object_name = kwargs.get("object_name", "SettingsSectionCard")
        bg = resolved.sidebar_surface
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        return (
            f"#{object_name} {{ background-color: {bg}; border: 1px solid {border}; "
            f"border-radius: 10px; }}"
        )
    if role == SETTINGS_FORM_CONTROLS:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        bg = resolved.surface_elevated
        text = resolved.text_primary
        disabled_bg = resolved.surface_pressed
        disabled_text = resolved.text_muted
        disabled_border = with_alpha(resolved.border, 0.5)
        return f"""
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 5px 10px;
            }}
            QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
                background-color: {disabled_bg};
                color: {disabled_text};
                border: 1px solid {disabled_border};
            }}
        """
    if role == SETTINGS_CHECKBOX:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        disabled_text = resolved.text_muted
        disabled_border = with_alpha(resolved.border, 0.5)
        disabled_indicator_bg = resolved.surface_pressed
        focus_border = resolved.accent if resolved.is_dark else adjust_lightness(resolved.border, -0.15)
        return f"""
            QCheckBox {{ color: {resolved.text_primary}; font-size: 13px; spacing: 8px; }}
            QCheckBox:disabled {{ color: {disabled_text}; }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {border};
                border-radius: 4px;
                background-color: transparent;
                image: none;
            }}
            QCheckBox::indicator:unchecked:disabled {{
                background-color: {disabled_indicator_bg};
                border: 1px solid {disabled_border};
                image: none;
            }}
            QCheckBox::indicator:checked {{
                background-color: {resolved.accent};
                border: 1px solid {resolved.accent_pressed};
                image: url(assets/icons/check_mark.png);
            }}
            QCheckBox::indicator:checked:disabled {{
                background-color: {resolved.accent_pressed};
                border: 1px solid {disabled_border};
                image: url(assets/icons/check_mark.png);
            }}
            QCheckBox::indicator:focus {{
                border: 1px solid {focus_border};
            }}
            QCheckBox::indicator:checked:focus {{
                border: 1px solid {resolved.accent_pressed};
            }}
        """
    if role == SETTINGS_LINE_EDIT:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        bg = resolved.surface_elevated
        disabled_bg = resolved.surface_pressed
        disabled_text = resolved.text_muted
        disabled_border = with_alpha(resolved.border, 0.5)
        focus_border = resolved.accent if resolved.is_dark else adjust_lightness(resolved.border, -0.15)
        return f"""
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {bg};
                color: {resolved.text_primary};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QPlainTextEdit {{
                background-color: {bg};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border: 1px solid {focus_border};
                background-color: {bg};
            }}
            QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
                background-color: {disabled_bg};
                color: {disabled_text};
                border: 1px solid {disabled_border};
            }}
        """
    if role == SETTINGS_SLIDER:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        bg = resolved.surface_elevated if resolved.is_dark else resolved.background
        handle = resolved.accent
        return f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {bg};
                border: 1px solid {border};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {handle};
                border: 1px solid {border};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {handle};
                border-radius: 3px;
            }}
            QSlider:disabled {{ opacity: 0.5; }}
        """
    if role == SETTINGS_LABEL:
        size = kwargs.get("font_size", "13px")
        weight = kwargs.get("font_weight", "normal")
        min_width = kwargs.get("min_width")
        width_rule = f" min-width: {min_width};" if min_width else ""
        return (
            f"color: {resolved.text_primary}; font-size: {size}; font-weight: {weight}; "
            f"background: transparent; border: none;{width_rule}"
        )
    if role == SETTINGS_ICON_BUTTON:
        bg = resolved.surface_elevated if resolved.is_dark else resolved.surface
        hover = resolved.surface_hover if resolved.is_dark else resolved.border
        return f"""
            QPushButton {{
                background: {bg};
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background: {hover};
            }}
        """
    if role == SETTINGS_DANGER_GHOST_BUTTON:
        hover = with_alpha(resolved.error, 0.1)
        return f"""
            QPushButton {{
                background: transparent;
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
        """
    if role == SETTINGS_PRESTIGE_MENU:
        colors = settings_prestige_menu_palette(resolved)
        bg = colors["bg"]
        fg = colors["fg"]
        hover = colors["hover"]
        sel_fg = colors["sel_fg"]
        border = colors["border"]
        return f"""
            QMenu {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px;
            }}
            QListWidget#PrestigeMenuList {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget#PrestigeMenuList::item {{
                background-color: transparent;
                color: {fg};
                padding: 8px 25px;
                border-radius: 4px;
                min-height: 24px;
            }}
            QListWidget#PrestigeMenuList::item:selected,
            QListWidget#PrestigeMenuList::item:hover {{
                background-color: {hover};
                color: {sel_fg};
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
    if role == SETTINGS_BORDERED_LIST:
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        bg = resolved.background
        object_name = kwargs.get("object_name", "SettingsTriggerList")
        widget_type = kwargs.get("widget_type", "QListWidget")
        item_padding = kwargs.get("item_padding", "2px 12px")
        return f"""
            {widget_type}#{object_name}, {widget_type} {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            {widget_type}::item {{
                padding: {item_padding};
                margin-bottom: 0px;
                border-bottom: 1px solid {border};
            }}
        """
    if role == SETTINGS_WARNING_LABEL:
        return (
            f"color: {resolved.warning}; font-size: 12px; "
            f"background: transparent; border: none;"
        )
    if role == SETTINGS_HINT:
        return (
            f"color: {resolved.text_muted}; font-size: 12px; "
            f"font-weight: normal; background: transparent; border: none;"
        )
    if role == SETTINGS_GHOST_TOOL_BUTTON:
        return "QToolButton { border: none; padding: 0px; background: transparent; }"
    if role == SETTINGS_BORDERLESS_TABLE:
        object_name = kwargs.get("object_name", "SettingsBorderlessTable")
        return f"""
            QTableWidget#{object_name} {{
                background: transparent;
                border: none;
            }}
            QTableWidget#{object_name}::item {{
                padding: 6px 4px;
                border: none;
            }}
            QTableWidget#{object_name} QHeaderView::section {{
                background: transparent;
                border: none;
                padding: 4px;
                font-weight: 600;
            }}
        """
    if role == KNOWLEDGE_ACCESS_BADGE:
        access = str(kwargs.get("access", "coming_soon"))
        fg, bg, border = _knowledge_access_colors(resolved, access)
        return f"""
            QLabel#KnowledgeAccessBadge {{
                padding: 0 10px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                color: {fg} !important;
                background-color: {bg} !important;
                border: 1px solid {border} !important;
            }}
        """
    if role == KNOWLEDGE_ACCESS_HINT:
        return f"""
            QLabel#KnowledgeAccessHint {{
                color: {resolved.text_secondary};
                font-size: 11px;
                font-weight: normal;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == KNOWLEDGE_ACTION_BUTTON:
        variant = kwargs.get("variant", "configure")
        object_name = kwargs.get("object_name", "KnowledgeConfigureButton")
        surface = resolved.surface_elevated if resolved.is_dark else resolved.background
        if variant == "free":
            fg, border = resolved.success, adjust_lightness(resolved.success, -0.12)
            hover = with_alpha(resolved.success, 0.12)
        else:
            fg, border = resolved.link, adjust_lightness(resolved.link, -0.12)
            hover = with_alpha(resolved.link, 0.12)
        return f"""
            QPushButton#{object_name} {{
                padding: 4px 10px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
                color: {fg} !important;
                border: 1px solid {border} !important;
                background-color: {surface} !important;
            }}
            QPushButton#{object_name}:hover {{
                background-color: {hover} !important;
            }}
            QPushButton#{object_name}:disabled {{
                color: {fg} !important;
                border: 1px solid {border} !important;
                background-color: {surface} !important;
            }}
        """
    if role == KNOWLEDGE_SETUP_CALLOUT:
        fg, _, border = _knowledge_access_colors(resolved, "optional_key")
        surface = resolved.surface_elevated if resolved.is_dark else resolved.background
        return f"""
            QWidget#KnowledgeSetupCallout {{
                background-color: {surface};
                border: 1px solid {border};
                border-radius: 8px;
            }}
        """
    if role == KNOWLEDGE_SETUP_CALLOUT_TITLE:
        fg, _, _ = _knowledge_access_colors(resolved, "optional_key")
        return f"""
            QLabel#KnowledgeSetupCalloutTitle {{
                color: {fg};
                font-size: 11px;
                font-weight: 600;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == KNOWLEDGE_SETUP_CALLOUT_BODY:
        return f"""
            QLabel#KnowledgeSetupCalloutBody {{
                color: {resolved.text_secondary};
                font-size: 12px;
                font-weight: 400;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == KNOWLEDGE_SETUP_CALLOUT_DISMISS:
        surface = resolved.surface_elevated if resolved.is_dark else resolved.background
        return f"""
            QPushButton#KnowledgeSetupCalloutDismiss {{
                padding: 4px 12px;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
                color: {resolved.text_secondary};
                border: 1px solid {resolved.border};
                background-color: {surface};
            }}
            QPushButton#KnowledgeSetupCalloutDismiss:hover {{
                background-color: {resolved.surface_hover};
            }}
        """
    if role == DISCOVERY_PROVIDER_CARD:
        disc_role = str(kwargs.get("discovery_role", "fallback"))
        accent, _, border = _discovery_role_accent(resolved, disc_role)
        shell_bg = resolved.surface_elevated if resolved.is_dark else resolved.background
        return f"""
            QWidget#DiscoveryProviderCard {{
                background-color: {shell_bg};
                border: 1px solid {border};
                border-left: 3px solid {accent};
                border-radius: 10px;
            }}
        """
    if role == DISCOVERY_ROLE_CHIP:
        disc_role = str(kwargs.get("discovery_role", "fallback"))
        accent, bg_tint, border = _discovery_role_accent(resolved, disc_role)
        background = bg_tint if resolved.is_dark else resolved.background
        return f"""
            QLabel#DiscoveryCardRoleChip {{
                color: {accent} !important;
                background-color: {background} !important;
                border: 1px solid {border} !important;
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.08em;
            }}
        """
    if role == DISCOVERY_PROVIDER_NAME:
        return f"""
            QLabel#DiscoveryCardProviderName {{
                color: {resolved.text_primary};
                font-size: 15px;
                font-weight: 600;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == DISCOVERY_PRIVACY_CHIP:
        bg = with_alpha(resolved.text_muted, 0.14 if resolved.is_dark else 0.08)
        border = with_alpha(resolved.text_muted, 0.22 if resolved.is_dark else 0.28)
        return f"""
            QLabel#DiscoveryCardPrivacyChip {{
                color: {resolved.text_secondary} !important;
                background-color: {bg} !important;
                border: 1px solid {border} !important;
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 500;
            }}
        """
    if role == DISCOVERY_BODY_TEXT:
        return f"""
            QLabel#DiscoveryCardBody {{
                color: {resolved.text_secondary};
                font-size: 12px;
                font-weight: 400;
                line-height: 1.45;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == DISCOVERY_INFO_CARD:
        variant = str(kwargs.get("variant", "policy"))
        if variant == "privacy":
            accent = resolved.success
        else:
            accent = resolved.link
        bg_tint = with_alpha(accent, 0.12 if resolved.is_dark else 0.08)
        border = with_alpha(accent, 0.32 if resolved.is_dark else 0.22)
        return f"""
            QWidget#DiscoveryInfoCard {{
                background-color: {bg_tint};
                border: 1px solid {border};
                border-top: 2px solid {accent};
                border-radius: 10px;
            }}
        """
    if role == DISCOVERY_INFO_TITLE:
        variant = str(kwargs.get("variant", "policy"))
        accent = resolved.success if variant == "privacy" else resolved.link
        return f"""
            QLabel#DiscoveryInfoCardTitle {{
                color: {accent};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.1em;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == DISCOVERY_INFO_HIGHLIGHT:
        bg = with_alpha(resolved.link, 0.1 if resolved.is_dark else 0.06)
        border = with_alpha(resolved.link, 0.2 if resolved.is_dark else 0.14)
        return f"""
            QLabel#DiscoveryInfoHighlight {{
                color: {resolved.text_primary};
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 8px 10px;
                font-size: 12px;
                font-weight: 600;
            }}
        """
    if role == DISCOVERY_INFO_KV_KEY:
        return f"""
            QLabel#DiscoveryInfoKvKey {{
                color: {resolved.text_muted};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.04em;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == DISCOVERY_INFO_KV_VALUE:
        return f"""
            QLabel#DiscoveryInfoKvValue {{
                color: {resolved.text_primary};
                font-size: 12px;
                font-weight: 500;
                background: transparent;
                border: none;
                padding: 0;
            }}
        """
    if role == DISCOVERY_INFO_BULLET:
        return f"""
            QLabel#DiscoveryInfoBullet {{
                color: {resolved.text_secondary};
                font-size: 12px;
                font-weight: 400;
                background: transparent;
                border: none;
                padding: 2px 0 2px 2px;
            }}
        """
    if role == DISCOVERY_INFO_STATUS:
        return f"""
            QLabel#DiscoveryInfoStatus {{
                color: {resolved.text_muted};
                font-size: 11px;
                font-weight: 400;
                font-style: italic;
                background: transparent;
                border: none;
                padding: 1px 0;
            }}
        """
    if role == DISCOVERY_DIVIDER:
        color = resolved.border_subtle if resolved.is_dark else with_alpha(resolved.border, 0.28)
        return f"background-color: {color}; border: none;"
    if role == PROVIDER_STATUS_TABLE:
        object_name = kwargs.get("object_name", "KnowledgeProviderStatusTable")
        header_bg = with_alpha(resolved.surface_elevated, 0.45 if resolved.is_dark else 0.95)
        header_fg = resolved.text_secondary
        header_border = resolved.border_subtle if resolved.is_dark else resolved.border
        scroll_handle = with_alpha(resolved.text_muted, 0.35)
        return f"""
            QTableWidget#{object_name} {{
                background-color: transparent;
                border: none;
                gridline-color: transparent;
                outline: none;
                color: {resolved.text_primary};
            }}
            QTableWidget#{object_name} QAbstractScrollArea::viewport {{
                background-color: transparent;
            }}
            QTableWidget#{object_name}::item {{
                color: {resolved.text_primary};
                border: none;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 400;
            }}
            QTableWidget#{object_name} QHeaderView::section {{
                background-color: {header_bg};
                color: {header_fg};
                padding: 6px 10px;
                border: none;
                border-bottom: 1px solid {header_border};
                font-size: 11px;
                font-weight: 600;
            }}
            QTableWidget#{object_name} QTableCornerButton::section {{
                background-color: {header_bg};
                border: none;
            }}
            QTableWidget#{object_name} QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 4px 2px 4px 0;
            }}
            QTableWidget#{object_name} QScrollBar::handle:vertical {{
                background: {scroll_handle};
                border-radius: 4px;
                min-height: 24px;
            }}
            QTableWidget#{object_name} QScrollBar::add-line:vertical,
            QTableWidget#{object_name} QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """
    if role == PRESTIGE_DIALOG_SHELL:
        variant = kwargs.get("variant", "default")
        accent, confirm_fg = prestige_accent_colors(
            resolved, tone=variant, title=kwargs.get("title", "")
        )
        border = resolved.border_subtle if resolved.is_dark else resolved.border
        input_bg = resolved.surface_elevated
        return f"""
            QDialog {{
                background-color: {resolved.background};
                color: {resolved.text_primary};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel {{
                color: {resolved.text_primary};
                background: transparent;
            }}
            QLineEdit {{
                background: {input_bg};
                color: {resolved.text_primary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 10px;
            }}
            QPushButton#PrestigeDialogConfirm {{
                background-color: {accent};
                color: {confirm_fg};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 700;
            }}
            QPushButton#PrestigeDialogCancel {{
                background: transparent;
                color: {resolved.text_secondary};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 16px;
            }}
        """
    if role == PRESTIGE_DIALOG_LIST:
        row_hover = resolved.surface_hover
        muted = resolved.text_secondary
        return f"""
            QListWidget {{
                background: {resolved.surface_elevated if resolved.is_dark else resolved.surface};
                color: {resolved.text_primary};
                border: 1px solid {resolved.border_subtle if resolved.is_dark else resolved.border};
                border-radius: 8px;
            }}
            QListWidget::item {{
                padding: 8px 12px;
                color: {resolved.text_primary};
            }}
            QListWidget::item:hover {{
                background: {row_hover};
            }}
            QListWidget::item:selected {{
                background: {resolved.selection_bg};
                color: {resolved.text_primary};
            }}
            QLabel#PrestigeDialogMuted {{
                color: {muted};
            }}
        """
    if role == ONBOARDING_COACH_PANEL:
        border = (
            with_alpha(resolved.link, 0.45)
            if resolved.is_dark
            else resolved.border
        )
        body = (
            with_alpha(resolved.text_primary, 0.92)
            if resolved.is_dark
            else resolved.text_secondary
        )
        hint = resolved.warning
        step = resolved.link if resolved.is_dark else adjust_lightness(resolved.link, -0.15)
        return f"""
            QFrame#OnboardingCoachPanel {{
                background-color: {resolved.background};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#OnboardingCoachStep {{
                color: {step};
                font-size: 11px;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
            QLabel#OnboardingCoachTitle {{
                color: {resolved.text_primary};
                font-size: 15px;
                font-weight: 700;
                background: transparent;
                border: none;
            }}
            QLabel#OnboardingCoachBody {{
                color: {body};
                font-size: 13px;
                background: transparent;
                border: none;
            }}
            QLabel#OnboardingCoachHint {{
                color: {hint};
                font-size: 12px;
                font-style: italic;
                background: transparent;
                border: none;
            }}
        """
    raise ValueError(f"Unknown theme style role: {role!r}")


def apply_theme_style(widget, resolved: ResolvedTheme, role: str, **kwargs) -> None:
    widget.setStyleSheet(theme_style(resolved, role, **kwargs))
