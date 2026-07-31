"""Guided tour: Settings → Privacy & data."""

from __future__ import annotations

from core.diagnostic_logs import iter_diagnostic_logs_by_category
from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step
from ui.onboarding.tours.settings._diagnostic_logs import diagnostic_log_tour_steps


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "privacy.data", anchor="overview")


def _open_web(host) -> None:
    open_settings_section(host, "privacy.data", anchor="web_discovery_privacy")


def _open_session_audit(host) -> None:
    open_settings_section(host, "privacy.data", anchor="session_audit")


def build_settings_privacy_data_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Privacy & data",
            body=(
                "Control web discovery privacy, Hybrid Internet Mode, and audit logs "
                "that may contain queries, prompts, or retrieval traces on disk."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="overview",
            title="Data & egress overview",
            body=(
                "Summary of what Qube stores locally and where to audit integrations "
                "and Telemetry session summaries."
            ),
            target_getter=lambda h: _sv(h).privacy_data_overview_hint,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="session_audit",
            title="Session audit",
            body=(
                "Jump to Telemetry for live web discovery budgets and integration "
                "calls for the active conversation."
            ),
            target_getter=lambda h: _sv(h).privacy_data_session_audit_hint,
            on_enter=_open_session_audit,
        ),
        OnboardingStep(
            step_id="privacy_tier",
            title="Web discovery privacy tier",
            body=(
                "Choose how @internet and Hybrid Internet Mode balance privacy vs "
                "optional API fallbacks. Advanced provider setup stays on Knowledge."
            ),
            target_getter=lambda h: _sv(h).privacy_data_privacy_tier_selector,
            on_enter=_open_web,
        ),
        OnboardingStep(
            step_id="hybrid_internet",
            title="Hybrid Internet Mode",
            body=(
                "Mirror of the Conversations tools panel toggle — Qube may auto-route "
                "to web search when context warrants it."
            ),
            target_getter=lambda h: _sv(h).privacy_data_internet_hybrid_toggle,
            on_enter=_open_web,
        ),
        OnboardingStep(
            step_id="what_leaves_device",
            title="What leaves your device",
            body=(
                "Plain-language summary of outbound web discovery traffic for the "
                "current privacy tier — updates when you change tier."
            ),
            target_getter=lambda h: _sv(h).privacy_data_what_leaves_card,
            on_enter=_open_web,
        ),
        *diagnostic_log_tour_steps(
            "privacy.data",
            iter_diagnostic_logs_by_category("audit"),
        ),
        make_settings_tour_finish_step("Privacy & data", _open),
    ]
    return OnboardingTour(host, steps)
