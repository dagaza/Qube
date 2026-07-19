"""Guided tour: Settings → Contact & Feedback."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "contact.feedback")


def build_settings_contact_feedback_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Contact & Feedback",
            body=(
                "Send bug reports or feature requests to the Qube team. Logs may be "
                "attached when you report an issue."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="bug",
            title="Report a bug",
            body=(
                "Opens a form to describe what went wrong. Include steps to reproduce "
                "when possible."
            ),
            target_getter=lambda h: _sv(h).report_bug_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="feature",
            title="Request a feature",
            body=(
                "Share ideas for improvements. Feature requests help prioritise the "
                "roadmap."
            ),
            target_getter=lambda h: _sv(h).request_feature_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Contact & Feedback settings", _open),
    ]
    return OnboardingTour(host, steps)
