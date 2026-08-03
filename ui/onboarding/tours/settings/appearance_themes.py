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
                "Customize appearance, theme colors, and chat or library wallpapers. "
                "Each card has its own preview and action row — changes stay in draft "
                "until you press Apply on that card."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="appearance",
            title="Appearance mode",
            body=(
                "Choose Dark, Light, or Follow system. Follow system remembers the last "
                "theme you used for each polarity. Preview until you Apply on the Chat "
                "wallpaper card."
            ),
            target_getter=lambda h: _sv(h).themes_appearance_row,
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
            step_id="theme_colors",
            title="Theme colors",
            body=(
                "Adjust core and advanced color swatches for the draft preview. "
                "Enable auto-adjust to nudge text contrast, then check the contrast "
                "readout below the swatches."
            ),
            target_getter=lambda h: _sv(h).themes_auto_adjust_cb,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="colors_preview",
            title="Theme colors preview",
            body=(
                "Miniature Settings page showing how draft colors look on the mainstage "
                "canvas, sidebar, section cards, and form controls."
            ),
            target_getter=lambda h: _sv(h).themes_components_preview_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="colors_actions",
            title="Theme colors actions",
            body=(
                "Reset to default clears the color draft to this preset's defaults. "
                "Revert restores the draft to colors currently applied in the app. "
                "Cancel matches Revert. Apply commits the color draft globally."
            ),
            target_getter=lambda h: _sv(h).themes_colors_apply_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="chat_wallpaper",
            title="Chat wallpaper",
            body=(
                "Decorate the Conversations transcript background. Pick wallpaper type, "
                "readability overlay, and optional assistant message background. "
                "Wallpapers never change core theme tokens."
            ),
            target_getter=lambda h: _sv(h).themes_chat_wallpaper,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="chat_preview",
            title="Chat preview",
            body=(
                "Miniature Conversations page shell with the tools pane open."
            ),
            target_getter=lambda h: _sv(h).themes_preview_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="chat_actions",
            title="Chat wallpaper actions",
            body=(
                "Reset to default sets the chat wallpaper draft to theme default. "
                "Revert restores the chat wallpaper and theme-preset draft to what is "
                "currently applied. Apply commits those drafts to the running app."
            ),
            target_getter=lambda h: _sv(h).themes_apply_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="library_wallpaper",
            title="Library wallpaper",
            body=(
                "Decorate the library document preview background separately from chat."
            ),
            target_getter=lambda h: _sv(h).themes_library_wallpaper,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="library_preview",
            title="Library preview",
            body=(
                "Miniature Library page shell with document list, readability toolbar, "
                "and sample transcript text."
            ),
            target_getter=lambda h: _sv(h).themes_library_preview_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="library_actions",
            title="Library wallpaper actions",
            body=(
                "Reset to default sets the library wallpaper draft to theme default. "
                "Revert and Cancel restore the draft to the applied library wallpaper. "
                "Apply commits the library wallpaper draft."
            ),
            target_getter=lambda h: _sv(h).themes_library_apply_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="share",
            title="Share Themes (Pro+)",
            body=(
                "Save the current draft as a custom theme, or import, export, and "
                "share theme packs. The footer Reset to default configuration restores "
                "the entire Themes section immediately."
            ),
            target_getter=lambda h: _sv(h).themes_save_as_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Themes settings", _open),
    ]
    return OnboardingTour(host, steps)
