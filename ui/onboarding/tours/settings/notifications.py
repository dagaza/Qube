"""Guided tour: Settings → Notifications."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "notifications")


def build_settings_notifications_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Notification settings",
            body=(
                "Control master alerts, quiet hours, focus behaviour, category toasts, "
                "and notification history from this page."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="master",
            title="Enable notifications",
            body=(
                "Master switch for Qube notification centre toasts and related alerts."
            ),
            target_getter=lambda h: _sv(h).notifications_enabled_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="dnd",
            title="Do Not Disturb",
            body=(
                "When on, only critical notifications break through — useful during "
                "focused work."
            ),
            target_getter=lambda h: _sv(h).notifications_dnd_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="suppress_focus",
            title="Suppress while focused",
            body=(
                "Hides info and success toasts while the Qube window is focused so "
                "non-critical alerts do not interrupt typing."
            ),
            target_getter=lambda h: _sv(h).notifications_suppress_focus_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="os_hidden",
            title="OS notifications when hidden",
            body=(
                "Allow the system notification centre to show Qube alerts when the "
                "app is minimised or running in the tray."
            ),
            target_getter=lambda h: _sv(h).notifications_os_hidden_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="sound",
            title="Alert sounds",
            body=(
                "Play a short sound with notification toasts when audio cues help you "
                "notice important events."
            ),
            target_getter=lambda h: _sv(h).notifications_sound_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview",
            title="Message previews",
            body=(
                "Include message snippets in notification banners. Turn off for more "
                "privacy on shared screens."
            ),
            target_getter=lambda h: _sv(h).notifications_preview_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="memory",
            title="Memory notifications",
            body=(
                "Optional toasts when new long-term memories are extracted — helps you "
                "catch surprising recalls early."
            ),
            target_getter=lambda h: _sv(h).notifications_memory_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="clear_history",
            title="Clear notification history",
            body=(
                "Remove stored notification history from the in-app notification centre. "
                "This does not change your alert preferences above."
            ),
            target_getter=lambda h: _sv(h).notifications_clear_history_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Notifications settings", _open),
    ]
    return OnboardingTour(host, steps)
