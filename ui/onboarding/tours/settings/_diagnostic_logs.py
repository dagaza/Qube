"""Shared guided-tour steps for diagnostic log cards."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ui.components.onboarding_tour import OnboardingStep
from ui.onboarding.tour_helpers import open_settings_section


def _sv(host):
    return host.settings_view


def diagnostic_log_tour_steps(
    section_id: str,
    specs: Iterable,
) -> list[OnboardingStep]:
    steps: list[OnboardingStep] = []

    def _open_anchor(host, anchor: str) -> None:
        open_settings_section(host, section_id, anchor=anchor)

    for spec in specs:
        log_id = spec.id
        if spec.supports_recording_toggle:
            steps.append(
                OnboardingStep(
                    step_id=f"{log_id}_recording",
                    title=f"{spec.title} — recording",
                    body=(
                        spec.recording_toggle_label
                        or "Enable file recording for this log. Existing lines stay on "
                        "disk until you clear the log."
                    ),
                    target_getter=lambda h, lid=log_id: _sv(h).diagnostic_log_recording_toggles.get(
                        lid
                    ),
                    on_enter=lambda h, lid=log_id: _open_anchor(h, lid),
                )
            )
        if spec.supports_redaction_toggle:
            steps.append(
                OnboardingStep(
                    step_id=f"{log_id}_redaction",
                    title=f"{spec.title} — redaction",
                    body=(
                        spec.redaction_toggle_label
                        or "Redact sensitive query text in new log entries before sharing."
                    ),
                    target_getter=lambda h, lid=log_id: _sv(h).diagnostic_log_redaction_toggles.get(
                        lid
                    ),
                    on_enter=lambda h, lid=log_id: _open_anchor(h, lid),
                )
            )
        steps.append(
            OnboardingStep(
                step_id=f"{log_id}_view",
                title=f"View {spec.title.lower()}",
                body=(
                    f"Open an in-app viewer for {spec.title.lower()} with refresh and "
                    "live tail while troubleshooting."
                ),
                target_getter=lambda h, lid=log_id: _sv(h).diagnostic_log_view_buttons.get(
                    lid
                ),
                on_enter=lambda h, lid=log_id: _open_anchor(h, lid),
            )
        )
        steps.append(
            OnboardingStep(
                step_id=f"{log_id}_clear",
                title=f"Clear {spec.title.lower()}",
                body=(
                    "Delete all contents of this log file and any rotated backups. "
                    "New entries resume automatically when recording is enabled."
                ),
                target_getter=lambda h, lid=log_id: _sv(h).diagnostic_log_clear_buttons.get(
                    lid
                ),
                on_enter=lambda h, lid=log_id: _open_anchor(h, lid),
            )
        )
    return steps
