"""Guided tour: Settings → Help."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "help", anchor="tours")


def build_settings_help_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Help & guidance",
            body=(
                "Replay onboarding tours, open reference guides, and tune discovery "
                "options from this section."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="first_run",
            title="Local LLM setup tour",
            body=(
                "Replay the first-run walkthrough for Internal Engine, model download, "
                "wakeword setup, and @ mentions."
            ),
            target_getter=lambda h: _sv(h).replay_local_llm_tour_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer_guide",
            title="@ Composer guide",
            body=(
                "Full reference for @ tokens, mixing limits, skills, and app commands "
                "in the chat composer."
            ),
            target_getter=lambda h: _sv(h).open_composer_mention_guide_btn,
            on_enter=lambda h: open_settings_section(h, "help", anchor="composer-mentions"),
        ),
        OnboardingStep(
            step_id="hardware_hints",
            title="Hardware-aware Model Manager",
            body=(
                "When enabled, Model Manager ranks verified models and shows Good fit "
                "badges from detected RAM and VRAM."
            ),
            target_getter=lambda h: _sv(h).model_manager_hardware_suggestions_cb,
            on_enter=lambda h: open_settings_section(h, "help", anchor="discovery"),
        ),
        make_settings_tour_finish_step("Help settings", _open),
    ]
    return OnboardingTour(host, steps)
