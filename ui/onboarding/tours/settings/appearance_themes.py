"""Guided tour: Settings → Themes."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "appearance.themes")


def build_settings_appearance_themes_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Themes settings",
            body=(
                "Customize color schemes, chat and library wallpapers, and preview "
                "changes before applying them to the running app."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="theme_picker",
            title="Choose a theme",
            body=(
                "Pick a built-in preset or a custom theme from ~/.qube/themes/. "
                "The nav moon/sun button switches light/dark within the same family "
                "when a matching variant exists."
            ),
            target_getter=lambda h: _sv(h).themes_theme_picker,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="customize",
            title="Customize colors",
            body=(
                "Adjust core colors in the draft preview. Enable auto-adjust to nudge "
                "text contrast, or open Advanced colors for surfaces and status tokens."
            ),
            target_getter=lambda h: _sv(h).themes_auto_adjust_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="wallpapers",
            title="Surface wallpapers",
            body=(
                "Decorate the chat transcript and library preview backgrounds. "
                "Wallpapers preview here until you press Apply and never change core "
                "theme tokens."
            ),
            target_getter=lambda h: _sv(h).themes_chat_wallpaper,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preview",
            title="Live preview",
            body=(
                "Miniature Conversations shell with optional More components for "
                "settings fields, memory rows, and status chips."
            ),
            target_getter=lambda h: _sv(h).themes_preview_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="apply",
            title="Apply or share",
            body=(
                "Apply commits the draft to the running app. Revert and Cancel discard "
                "draft edits; Save as custom theme or Import theme share presets."
            ),
            target_getter=lambda h: _sv(h).themes_apply_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Themes settings", _open),
    ]
    return OnboardingTour(host, steps)
