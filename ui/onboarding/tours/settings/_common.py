"""Shared helpers for settings section guided tours."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, StepCallback


def make_settings_tour_finish_step(
    section_title: str,
    on_enter: StepCallback,
) -> OnboardingStep:
    return OnboardingStep(
        step_id="tour_complete",
        title="Congratulations!",
        body=(
            f"Congratulations for finishing the {section_title} guide. Reopen it "
            "anytime from the ? button beside the section title."
        ),
        on_enter=on_enter,
    )
