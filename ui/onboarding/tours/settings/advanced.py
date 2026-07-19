"""Guided tour: Settings → Advanced."""

from __future__ import annotations

from core.diagnostic_logs import iter_diagnostic_logs
from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "advanced", anchor="json")


def _open_anchor(host, anchor: str) -> None:
    open_settings_section(host, "advanced", anchor=anchor)


def _log_toggle(host, log_id: str):
    return _sv(host).diagnostic_log_recording_toggles.get(log_id)


def _log_view_btn(host, log_id: str):
    return _sv(host).diagnostic_log_view_buttons.get(log_id)


def _log_clear_btn(host, log_id: str):
    return _sv(host).diagnostic_log_clear_buttons.get(log_id)


def _diagnostic_log_steps() -> list[OnboardingStep]:
    steps: list[OnboardingStep] = []
    for spec in iter_diagnostic_logs():
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
                    target_getter=lambda h, lid=log_id: _log_toggle(h, lid),
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
                target_getter=lambda h, lid=log_id: _log_view_btn(h, lid),
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
                target_getter=lambda h, lid=log_id: _log_clear_btn(h, lid),
                on_enter=lambda h, lid=log_id: _open_anchor(h, lid),
            )
        )
    return steps


def build_settings_advanced_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Advanced settings",
            body=(
                "Power-user tools: edit raw JSON settings, open the logs folder, and "
                "manage dedicated diagnostic log files. Proceed carefully."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="json",
            title="JSON settings editor",
            body=(
                "Inspect or override keys not exposed in the UI. Invalid JSON is rejected "
                "before save; reload when settings.json changes on disk."
            ),
            target_getter=lambda h: _sv(h).open_settings_json_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="logs_folder",
            title="Open logs folder",
            body=(
                "Reveal the logs directory in your file manager for all rotating debug "
                "files Qube writes under ~/.qube/logs."
            ),
            target_getter=lambda h: _sv(h).open_logs_folder_btn,
            on_enter=lambda h: _open_anchor(h, "logs"),
        ),
        *_diagnostic_log_steps(),
        make_settings_tour_finish_step("Advanced settings", _open),
    ]
    return OnboardingTour(host, steps)
