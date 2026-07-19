"""Guided tour: Settings → Memory."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "memory")


def build_settings_memory_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Memory settings",
            body=(
                "Control how Qube extracts, promotes, and presents long-term memories "
                "across conversations."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="enrichment",
            title="Memory enrichment",
            body=(
                "When enabled, Qube may add structured context from recent turns to improve "
                "recall quality — uses extra background work."
            ),
            target_getter=lambda h: _sv(h).memory_enrichment_toggle,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="promotion",
            title="Memory promotion",
            body=(
                "Well-used facts can graduate into preference-tier memories. Confirm the "
                "first-time prompt if you enable this."
            ),
            target_getter=lambda h: _sv(h).memory_promotion_toggle,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="preset",
            title="Promotion preset",
            body=(
                "Choose how aggressively memories are promoted — Standard is a balanced default."
            ),
            target_getter=lambda h: _sv(h).memory_promotion_preset_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="consolidation",
            title="Consolidation highlights",
            body=(
                "Surfaces memories that keep reappearing so you can review duplicates or "
                "outdated facts in Memory Manager."
            ),
            target_getter=lambda h: _sv(h).memory_consolidation_toggle,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="units",
            title="Presentation units",
            body=(
                "Optional display preference for measurement units in memory summaries."
            ),
            target_getter=lambda h: _sv(h).profile_units_selector,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Memory settings", _open),
    ]
    return OnboardingTour(host, steps)
