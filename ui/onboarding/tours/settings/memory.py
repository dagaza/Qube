"""Guided tour: Settings → Memory."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import (
    dismiss_memory_settings_tour_transients,
    open_settings_section,
)
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    dismiss_memory_settings_tour_transients(host)
    open_settings_section(host, "memory")


def _refresh_tour_layout(host) -> None:
    from PyQt6.QtCore import QTimer

    refresh = getattr(host, "refresh_active_tour_layout", None)
    if refresh is not None:
        QTimer.singleShot(180, refresh)


def _open_advanced_toggle(host) -> None:
    _open(host)
    open_settings_section(host, "memory", anchor="advanced_memory")
    _sv(host).begin_memory_advanced_tutorial_preview(reveal_panel=False)
    _refresh_tour_layout(host)


def _open_advanced_panel(host) -> None:
    _open(host)
    open_settings_section(host, "memory", anchor="advanced_memory")
    _sv(host).begin_memory_advanced_tutorial_preview()
    _refresh_tour_layout(host)


def build_settings_memory_tour(host) -> OnboardingTour:
    def _on_finished() -> None:
        dismiss_memory_settings_tour_transients(host)

    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Memory settings",
            body=(
                "Control how Qube extracts and retains long-term memories. Simple mode "
                "shows enrichment; unlock advanced settings for promotion and consolidation."
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
            step_id="advanced_memory_toggle",
            title="Advanced memory settings",
            body=(
                "Unlock promotion and consolidation workers. Both stay off until you "
                "enable them — most users can leave this collapsed."
            ),
            target_getter=lambda h: _sv(h).advanced_memory_toggle,
            on_enter=_open_advanced_toggle,
        ),
        OnboardingStep(
            step_id="promotion",
            title="Memory promotion",
            body=(
                "Well-used facts can graduate into preference-tier memories. Confirm the "
                "first-time prompt if you enable this."
            ),
            target_getter=lambda h: _sv(h).memory_promotion_toggle,
            on_enter=_open_advanced_panel,
        ),
        OnboardingStep(
            step_id="preset",
            title="Promotion preset",
            body=(
                "Choose how aggressively memories are promoted — Standard is a balanced default."
            ),
            target_getter=lambda h: _sv(h).memory_promotion_preset_selector,
            on_enter=_open_advanced_panel,
        ),
        OnboardingStep(
            step_id="consolidation",
            title="Consolidation highlights",
            body=(
                "Surfaces memories that keep reappearing so you can review duplicates or "
                "outdated facts in Memory Manager."
            ),
            target_getter=lambda h: _sv(h).memory_consolidation_toggle,
            on_enter=_open_advanced_panel,
        ),
        make_settings_tour_finish_step("Memory settings", _open),
    ]
    return OnboardingTour(host, steps, on_finished=_on_finished)
