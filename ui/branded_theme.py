"""Theme helpers for splash, bootstrap consent, and other branded startup chrome."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.tokens import ResolvedTheme

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QLabel


def qss_color(r: int, g: int, b: int, alpha: float = 1.0) -> str:
    """Hex color for Qt Style Sheets (``#RRGGBB`` or ``#RRGGBBAA``).

    Qt Style Sheets on Windows and macOS often ignore ``rgba(r,g,b,a)`` for
    ``color`` and ``background``, which leaves splash labels and surfaces black.
    """
    red = max(0, min(255, int(r)))
    green = max(0, min(255, int(g)))
    blue = max(0, min(255, int(b)))
    if alpha >= 1.0:
        return f"#{red:02x}{green:02x}{blue:02x}"
    channel_a = max(0, min(255, int(round(alpha * 255))))
    return f"#{red:02x}{green:02x}{blue:02x}{channel_a:02x}"


# Locked splash brand palette — matches pre-theme-migration splash on ``dev``.
# Not derived from user theme overrides (same policy as logo/confetti).
SPLASH_SURFACE_BG = "#12151f"
SPLASH_TITLE_COLOR = "#f8fafc"
SPLASH_HINT_COLOR = qss_color(148, 163, 184, 0.9)
SPLASH_DETAIL_COLOR = qss_color(148, 163, 184, 0.95)
SPLASH_DIVIDER_COLOR = qss_color(255, 255, 255, 0.12)
SPLASH_CHROME_ICON = "#94a3b8"
SPLASH_CHROME_BUTTON_BG = qss_color(18, 21, 31, 0.72)
SPLASH_CHROME_BUTTON_BG_HOVER = qss_color(30, 34, 48, 0.9)
SPLASH_CHROME_BUTTON_BORDER = qss_color(255, 255, 255, 0.12)
SPLASH_CHROME_BUTTON_BORDER_HOVER = qss_color(255, 255, 255, 0.2)
SPLASH_STEP_PENDING = qss_color(148, 163, 184, 0.55)
SPLASH_STEP_ACTIVE = "#c4b5fd"
SPLASH_STEP_DONE = qss_color(134, 239, 172, 0.85)
SPLASH_PROGRESS_TRACK_RGBA = (255, 255, 255, 20)
SPLASH_PROGRESS_CHUNK_HEX = "#8b5cf6"
SPLASH_PROGRESS_TEXT_RGBA = (148, 163, 184, 230)
SPLASH_SPINNER_TRACK_DARK_RGBA = (255, 255, 255, 28)
SPLASH_SPINNER_ARC_DARK_HEX = "#89b4fa"
SPLASH_SPINNER_ARC_DARK_INIT_HEX = "#8b5cf6"
SPLASH_SPINNER_TRACK_LIGHT_RGBA = (0, 0, 0, 22)
SPLASH_SPINNER_ARC_LIGHT_HEX = "#3b82f6"


def branded_theme(*, is_dark: bool = True) -> ResolvedTheme:
    """Resolved theme for splash/bootstrap surfaces (dark-branded by default)."""
    return theme_for(is_dark=is_dark)


def splash_title_label_qss() -> str:
    return f"color: {SPLASH_TITLE_COLOR}; background-color: transparent;"


def splash_hint_label_qss(*, font_size_px: int = 12) -> str:
    return (
        f"color: {SPLASH_HINT_COLOR}; background-color: transparent;"
        f" font-size: {font_size_px}px;"
    )


def splash_detail_label_qss(*, font_size_px: int = 10) -> str:
    return (
        f"color: {SPLASH_DETAIL_COLOR}; background-color: transparent;"
        f" font-size: {font_size_px}px; line-height: 1.35;"
    )


def splash_status_label_qss(*, font_size_px: int = 13) -> str:
    return (
        f"color: {SPLASH_DETAIL_COLOR}; background-color: transparent;"
        f" font-size: {font_size_px}px;"
    )


def apply_splash_label_styles(
    *,
    title: "QLabel | None" = None,
    hint: "QLabel | None" = None,
    status: "QLabel | None" = None,
    detail: "QLabel | None" = None,
) -> None:
    """Apply per-label QSS so splash text stays readable on every platform."""
    if title is not None:
        title.setStyleSheet(splash_title_label_qss())
    if hint is not None:
        hint.setStyleSheet(splash_hint_label_qss())
    if status is not None:
        status.setStyleSheet(splash_status_label_qss())
    if detail is not None:
        detail.setStyleSheet(splash_detail_label_qss())


def splash_card_surface_qss(_theme: ResolvedTheme | None = None) -> str:
    return (
        f"background-color: {SPLASH_SURFACE_BG}; border: none; border-radius: 16px;"
    )


def splash_step_list_qss(_theme: ResolvedTheme | None = None) -> str:
    return f"""
            QLabel[step_state="pending"] {{
                color: {SPLASH_STEP_PENDING};
                font-size: 11px;
            }}
            QLabel[step_state="active"] {{
                color: {SPLASH_STEP_ACTIVE};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel[step_state="done"] {{
                color: {SPLASH_STEP_DONE};
                font-size: 11px;
            }}
            """


def splash_split_card_qss(_theme: ResolvedTheme | None = None) -> str:
    surface = splash_card_surface_qss()
    return f"""
            QWidget#QubeFirstRunSplitSplashRoot {{
                background: transparent;
            }}
            QWidget#QubeFirstRunSplitSplash {{
                {surface}
            }}
            QWidget#QubeFirstRunSplashLeft,
            QWidget#QubeFirstRunConsentHost {{
                background: transparent;
            }}
            QFrame#QubeFirstRunSplitDivider {{
                background: {SPLASH_DIVIDER_COLOR};
                border: none;
                max-width: 1px;
            }}
            QLabel#QubeSplashTitle {{
                color: {SPLASH_TITLE_COLOR};
            }}
            QLabel#QubeFirstRunSplashHint {{
                color: {SPLASH_HINT_COLOR};
                font-size: 12px;
            }}
            QLabel#QubeSplashDetail {{
                color: {SPLASH_DETAIL_COLOR};
                font-size: 10px;
                line-height: 1.35;
            }}
            """


def splash_compact_card_qss(_theme: ResolvedTheme | None = None) -> str:
    surface = splash_card_surface_qss()
    return f"""
            QWidget#QubeSplashCardRoot {{
                background: transparent;
            }}
            QWidget#QubeSplashCard {{
                {surface}
            }}
            QLabel#QubeSplashTitle {{
                color: {SPLASH_TITLE_COLOR};
            }}
            QLabel#QubeSplashDetail {{
                color: {SPLASH_DETAIL_COLOR};
                font-size: 10px;
                line-height: 1.35;
            }}
            """


def early_splash_card_qss(_theme: ResolvedTheme | None = None) -> str:
    """QSS for the pre-import early splash (static logo + Loading label)."""
    surface = splash_card_surface_qss()
    return f"""
            QWidget#QubeEarlySplashCard {{
                {surface}
            }}
            QLabel#QubeEarlySplashTitle {{
                color: {SPLASH_TITLE_COLOR};
            }}
            QLabel#QubeEarlySplashStatus {{
                color: {SPLASH_DETAIL_COLOR};
                font-size: 13px;
            }}
            """


def splash_overlay_chrome_button_qss(object_name: str) -> str:
    return f"""
            QPushButton#{object_name} {{
                background: {SPLASH_CHROME_BUTTON_BG};
                border: 1px solid {SPLASH_CHROME_BUTTON_BORDER};
                border-radius: 6px;
                outline: none;
            }}
            QPushButton#{object_name}:hover {{
                background: {SPLASH_CHROME_BUTTON_BG_HOVER};
                border-color: {SPLASH_CHROME_BUTTON_BORDER_HOVER};
            }}
            QPushButton#{object_name}:focus {{
                background: {SPLASH_CHROME_BUTTON_BG};
                border: 1px solid {SPLASH_CHROME_BUTTON_BORDER};
                outline: none;
            }}
            """


def bootstrap_consent_root_qss(
    theme: ResolvedTheme,
    *,
    split_embedded: bool,
    embedded: bool,
) -> str:
    if split_embedded:
        return """
            QWidget#BootstrapConsentPanelSplit {
                background: transparent;
            }
            """
    if embedded:
        return f"""
            QWidget#BootstrapConsentPanelEmbedded {{
                background: {theme.background};
                border: 1px solid {with_alpha(theme.text_on_accent, 0.12)};
                border-radius: 16px;
            }}
            """
    return f"""
            QWidget#BootstrapConsentPanel {{
                background: {theme.background};
            }}
            """


def bootstrap_consent_stylesheet(
    theme: ResolvedTheme,
    *,
    split_embedded: bool,
    embedded: bool,
) -> str:
    """Full QSS for :class:`BootstrapConsentPanel` model picker chrome."""
    t = theme
    accent = t.accent
    return (
        bootstrap_consent_root_qss(
            theme, split_embedded=split_embedded, embedded=embedded
        )
        + f"""
            QWidget#BootstrapConsentPanelSplit QLabel,
            QWidget#BootstrapConsentPanelEmbedded QLabel,
            QWidget#BootstrapConsentPanel QLabel,
            QWidget#BootstrapModelTitleRow {{
                background: transparent;
            }}
            QLabel#BootstrapBrandTitle {{
                color: {t.text_on_accent};
                font-size: 22px;
                font-weight: 800;
            }}
            QLabel#BootstrapTitle {{
                color: {t.text_on_accent};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#BootstrapIntro,
            QLabel#BootstrapLegend {{
                color: {t.text_muted};
                font-size: 13px;
            }}
            QFrame#BootstrapCollapsiblePanel {{
                background: {with_alpha(t.text_on_accent, 0.03)};
                border: 1px solid {with_alpha(t.text_on_accent, 0.08)};
                border-radius: 10px;
            }}
            QFrame#BootstrapCollapsibleHeader {{
                background: transparent;
                border: none;
            }}
            QFrame#BootstrapCollapsibleHeader:hover QLabel#BootstrapCollapsibleHeaderText,
            QFrame#BootstrapCollapsibleHeader:hover QLabel#BootstrapCollapsibleArrow {{
                color: {t.accent_hover};
            }}
            QLabel#BootstrapCollapsibleArrow {{
                color: {t.text_muted};
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#BootstrapCollapsibleHeaderText {{
                color: {t.text_muted};
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#BootstrapCollapsibleSummary {{
                color: {t.text_secondary};
                font-size: 10px;
                line-height: 1.35;
                padding: 0 12px 8px 32px;
            }}
            QFrame#BootstrapCollapsibleBody {{
                border-top: 1px solid {with_alpha(t.text_on_accent, 0.06)};
            }}
            QLabel#BootstrapDiskSummary {{
                color: {t.text_secondary};
                font-size: 12px;
            }}
            QLabel#BootstrapDiskSummaryOver {{
                color: {with_alpha(t.error, 0.75)};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#BootstrapDiskNotice {{
                color: {t.warning};
                font-size: 11px;
                line-height: 1.35;
            }}
            QLabel#BootstrapSizeTagVerified {{
                background: {with_alpha(t.success, 0.14)};
                color: {with_alpha(t.success, 0.75)};
                border: 1px solid {with_alpha(t.success, 0.28)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapSizeTagEstimate {{
                background: {with_alpha(t.text_muted, 0.1)};
                color: {t.text_muted};
                border: 1px solid {with_alpha(t.text_muted, 0.2)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapTierTagRequired {{
                background: {with_alpha(accent, 0.16)};
                color: {t.accent_hover};
                border: 1px solid {with_alpha(accent, 0.35)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapTierTagStronglyRecommended {{
                background: {with_alpha(t.warning, 0.14)};
                color: {t.warning};
                border: 1px solid {with_alpha(t.warning, 0.28)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapTierTagRecommended {{
                background: {with_alpha(t.info, 0.14)};
                color: {with_alpha(t.info, 0.75)};
                border: 1px solid {with_alpha(t.info, 0.28)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapTierTagOptional {{
                background: {with_alpha(t.text_muted, 0.08)};
                color: {t.text_muted};
                border: 1px solid {with_alpha(t.text_muted, 0.18)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapBlockTagDisk {{
                background: {with_alpha(t.warning, 0.14)};
                color: {t.warning};
                border: 1px solid {with_alpha(t.warning, 0.3)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#BootstrapBlockTagMemory {{
                background: {with_alpha(t.error, 0.14)};
                color: {with_alpha(t.error, 0.75)};
                border: 1px solid {with_alpha(t.error, 0.3)};
                border-radius: 8px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 700;
            }}
            QFrame#BootstrapBulkBar {{
                background: {with_alpha(t.text_on_accent, 0.02)};
                border: 1px solid {with_alpha(t.text_on_accent, 0.06)};
                border-radius: 8px;
            }}
            QLabel#BootstrapBulkCaption {{
                color: {t.text_secondary};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
            QPushButton#BootstrapBulkPill {{
                background: {with_alpha(accent, 0.12)};
                color: {t.accent_hover};
                border: 1px solid {with_alpha(accent, 0.28)};
                border-radius: 12px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 14px;
                min-width: 48px;
            }}
            QPushButton#BootstrapBulkPill:hover {{
                background: {with_alpha(accent, 0.22)};
                color: {t.text_on_accent};
            }}
            QPushButton#BootstrapBulkPill:disabled {{
                color: {with_alpha(t.accent_hover, 0.35)};
                border-color: {with_alpha(accent, 0.12)};
            }}
            QLabel#BootstrapLegend {{
                font-size: 11px;
                color: {t.text_secondary};
            }}
            QLabel#BootstrapHardwareSummary {{
                color: {t.text_secondary};
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#BootstrapHfStatus {{
                color: {t.text_muted};
                font-size: 11px;
            }}
            QLabel#BootstrapModelFeasibilityNote {{
                color: {t.warning};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapModelFeasibilityDisk {{
                color: {t.warning};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapModelFeasibilityBlock {{
                color: {with_alpha(t.error, 0.75)};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapTotalLabel {{
                color: {t.accent_hover};
                font-size: 13px;
                font-weight: 600;
            }}
            QScrollArea#BootstrapScroll,
            QWidget#BootstrapScrollViewport,
            QWidget#BootstrapScrollHost,
            QWidget#BootstrapListSection {{
                background: transparent;
                border: none;
            }}
            QLabel#BootstrapBulkSep {{
                color: {with_alpha(t.text_muted, 0.45)};
                font-size: 11px;
                padding: 0 2px;
            }}
            QPushButton#BootstrapLinkBtn {{
                background: transparent;
                border: none;
                color: {t.text_secondary};
                font-size: 11px;
                font-weight: 500;
                padding: 0 2px;
            }}
            QPushButton#BootstrapLinkBtn:hover {{
                color: {t.accent_hover};
            }}
            QFrame#BootstrapModelRowDiskBlocked {{
                background: {with_alpha(t.background, 0.55)};
                border: 1px dashed {with_alpha(t.text_muted, 0.22)};
                border-radius: 10px;
            }}
            QFrame#BootstrapModelRow {{
                background: {with_alpha(t.text_on_accent, 0.04)};
                border: 1px solid {with_alpha(t.text_on_accent, 0.08)};
                border-radius: 10px;
            }}
            QFrame#BootstrapModelRowLocked {{
                background: {with_alpha(accent, 0.08)};
                border: 1px solid {with_alpha(accent, 0.22)};
                border-radius: 10px;
            }}
            QFrame#BootstrapModelRowInfo {{
                background: {with_alpha(t.text_muted, 0.06)};
                border: 1px solid {with_alpha(t.text_muted, 0.16)};
                border-radius: 10px;
            }}
            QFrame#BootstrapModelRowCaution {{
                background: {with_alpha(t.warning, 0.06)};
                border: 1px solid {with_alpha(t.warning, 0.22)};
                border-radius: 10px;
            }}
            QFrame#BootstrapModelRowCoreWarning {{
                background: {with_alpha(t.warning, 0.08)};
                border: 1px solid {with_alpha(t.warning, 0.28)};
                border-radius: 10px;
            }}
            QLabel#BootstrapModelDesc {{
                color: {t.text_secondary};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapModelDescInfo {{
                color: {t.text_muted};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapModelDescCaution {{
                color: {t.warning};
                font-size: 11px;
                margin-left: 22px;
            }}
            QLabel#BootstrapModelDescCoreWarning {{
                color: {t.warning};
                font-size: 11px;
                margin-left: 22px;
            }}
            QCheckBox#BootstrapCheckLocked {{
                color: {t.accent_hover};
            }}
            QCheckBox {{
                color: {t.text_secondary};
                font-size: 13px;
                background: transparent;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {with_alpha(t.text_on_accent, 0.22)};
                background: {with_alpha(t.background, 0.65)};
            }}
            QCheckBox::indicator:checked {{
                background: {accent};
                border-color: {t.accent_hover};
            }}
            QCheckBox::indicator:disabled {{
                background: {with_alpha(t.text_muted, 0.12)};
                border-color: {with_alpha(t.text_muted, 0.2)};
            }}
            QCheckBox::indicator:checked:disabled {{
                background: {t.accent_pressed};
                border-color: {t.accent_hover};
            }}
            QPushButton {{
                padding: 10px 16px;
                border-radius: 8px;
                font-weight: 600;
            }}
            QPushButton#BootstrapPrimaryBtn {{
                background: {accent};
                color: {t.text_on_accent};
                border: none;
            }}
            QPushButton#BootstrapPrimaryBtn:disabled {{
                background: {with_alpha(accent, 0.28)};
                color: {with_alpha(t.text_secondary, 0.45)};
            }}
            QPushButton#BootstrapSecondaryBtn {{
                background: transparent;
                color: {t.text_secondary};
                border: 1px solid {with_alpha(t.text_on_accent, 0.15)};
            }}
            QPushButton#BootstrapSecondaryBtn:disabled {{
                color: {with_alpha(t.text_secondary, 0.35)};
                border-color: {with_alpha(t.text_on_accent, 0.06)};
            }}
            """
    )


def bootstrap_scrollbar_stylesheet(theme: ResolvedTheme) -> str:
    return f"""
            QScrollBar#BootstrapScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 10px;
                margin: 0px;
            }}
            QScrollBar#BootstrapScrollBar::handle:vertical {{
                background-color: {with_alpha(theme.text_muted, 0.38)};
                border-radius: 5px;
                min-height: 30px;
            }}
            QScrollBar#BootstrapScrollBar::handle:vertical:hover {{
                background-color: {with_alpha(theme.accent_hover, 0.62)};
            }}
            QScrollBar#BootstrapScrollBar::add-line:vertical,
            QScrollBar#BootstrapScrollBar::sub-line:vertical,
            QScrollBar#BootstrapScrollBar::add-page:vertical,
            QScrollBar#BootstrapScrollBar::sub-page:vertical {{
                border: none;
                background: transparent;
                width: 0px;
                height: 0px;
            }}
            """
