"""Theme helpers for the Wakeword Test Lab dialog."""

from __future__ import annotations

from core.theme.color_utils import with_alpha
from core.theme.tokens import ResolvedTheme


def wakeword_testbed_stylesheet(theme: ResolvedTheme) -> str:
    """QSS for :class:`WakewordTestbedDialog` chrome and stateful widgets."""
    t = theme
    bg = t.background
    fg = t.text_primary
    card = t.surface_elevated if t.is_dark else t.surface
    border = t.border_subtle if t.is_dark else t.border
    subtext = with_alpha(t.text_secondary, 0.75 if t.is_dark else 1.0)
    header_subtext = with_alpha(t.text_secondary, 0.72 if t.is_dark else 1.0)
    alert = t.warning
    accent = t.accent

    def badge(bg_color: str, fg_color: str) -> str:
        return (
            f"background: {with_alpha(bg_color, 0.16)}; color: {fg_color};"
        )

    return f"""
            QFrame#WakewordLabContainer {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QFrame#WakewordGuidanceCard, QFrame#WakewordLiveCard, QFrame#WakewordResultsCard, QFrame#WakewordAdvancedCard {{
                background-color: {card};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QFrame#WakewordGuidanceCard[state="attention"] {{ border: 1px solid {t.info}; }}
            QFrame#WakewordGuidanceCard[state="cancelled"] {{ border: 1px solid {t.error}; }}
            QFrame#WakewordLiveCard[state="attention"] {{ border: 1px solid {t.info}; }}
            QFrame#WakewordResultsCard[state="success"] {{ border: 1px solid {t.success}; }}
            QFrame#WakewordResultsCard[state="caution"] {{ border: 1px solid {t.warning}; }}
            QFrame#WakewordResultsCard[state="failure"] {{ border: 1px solid {t.error}; }}
            QLabel {{ color: {fg}; }}
            QLabel#WakewordHeaderTitle {{ font-size: 18px; font-weight: 700; }}
            QLabel#WakewordHeaderSubtitle {{ color: {header_subtext}; font-size: 12px; }}
            QLabel#WakewordStageBadge {{
                font-size: 11px;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 8px;
                {badge(accent, t.accent_hover)}
            }}
            QLabel#WakewordStageBadge[state="attention"] {{ {badge(t.info, t.info)} }}
            QLabel#WakewordStageBadge[state="false_positive"] {{ {badge(t.warning, t.warning)} }}
            QLabel#WakewordStageBadge[state="success"] {{ {badge(t.success, t.success)} }}
            QLabel#WakewordStageBadge[state="warning"] {{ {badge(t.warning, t.warning)} }}
            QLabel#WakewordStageBadge[state="cancelled"] {{ {badge(t.error, t.error)} }}
            QLabel#WakewordStageBadge[state="error"] {{ {badge(t.error, t.error)} }}
            QLabel#WakewordGuidanceTitle {{ font-size: 15px; font-weight: 600; }}
            QLabel#WakewordGuidanceHint, QLabel#WakewordAdvancedLockHint {{ color: {subtext}; font-size: 12px; }}
            QLabel#WakewordAlertLabel {{ color: {alert}; font-weight: 600; }}
            QLabel#WakewordAdvancedTitle {{ color: {fg}; font-size: 12px; font-weight: 700; }}
            QLabel#WakewordResultsVerdict {{ font-size: 15px; font-weight: 700; }}
            QLabel#WakewordResultsVerdict[result_tone="success"] {{ color: {t.success}; }}
            QLabel#WakewordResultsVerdict[result_tone="caution"] {{ color: {t.warning}; }}
            QLabel#WakewordResultsVerdict[result_tone="failure"] {{ color: {t.error}; }}
            QLabel#WakewordResultsMetric {{ font-size: 13px; }}
            QLabel#WakewordResultsDetail {{ color: {subtext}; font-size: 12px; }}
            QLabel#WakewordFalsePositivePromptLabel {{
                color: {t.info};
                font-size: 12px;
                font-weight: 700;
            }}
            QLabel#WakewordFalsePositiveScriptLabel {{
                color: {fg};
                font-size: 12px;
                font-style: italic;
            }}
            QPushButton#WakewordHeaderCloseButton {{
                border: 1px solid {border};
                border-radius: 8px;
                padding: 4px;
                font-weight: 700;
            }}
            QPushButton#WakewordApplyButton[result_tone="success"] {{
                background-color: {t.success};
                color: {t.text_on_accent};
                border: 1px solid {t.success};
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QPushButton#WakewordApplyButton[result_tone="caution"] {{
                background-color: {t.warning};
                color: {t.text_primary};
                border: 1px solid {t.warning};
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QPushButton#WakewordApplyButton[result_tone="failure"] {{
                background-color: {t.error};
                color: {t.text_on_accent};
                border: 1px solid {t.error};
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }}
            QProgressBar {{
                background-color: transparent;
                border: 1px solid {border};
                border-radius: 6px;
                text-align: center;
                color: {subtext};
                min-height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {accent};
                border-radius: 5px;
            }}
            QProgressBar#WakewordInstructionBar {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 2px 0px;
                font-weight: 700;
                color: {t.info};
            }}
            QProgressBar#WakewordInstructionBar::chunk {{
                background-color: transparent;
                border-radius: 0px;
            }}
            QProgressBar#WakewordInstructionBar[state="countdown"] {{ color: {t.warning}; }}
            QProgressBar#WakewordInstructionBar[state="listening"] {{ color: {t.success}; }}
            QProgressBar#WakewordAttemptCounter {{
                background-color: transparent;
                border: 1px solid transparent;
                padding: 4px 0px 0px 0px;
            }}
            QProgressBar#WakewordAttemptCounter::chunk {{
                background-color: transparent;
                border-radius: 0px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: transparent;
                border: 1px solid {border};
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {accent};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                background: {accent};
                border: 1px solid {border};
            }}
            """
