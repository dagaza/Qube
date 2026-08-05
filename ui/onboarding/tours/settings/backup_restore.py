"""Guided tour: Settings → Backup & restore."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "system.backup", anchor="overview")


def _open_automatic(host) -> None:
    open_settings_section(host, "system.backup", anchor="automatic")


def _open_manual(host) -> None:
    open_settings_section(host, "system.backup", anchor="manual")


def build_settings_system_backup_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Backup & restore",
            body=(
                "Save and recover essential Qube state locally — conversations, library "
                "indexes, memory vectors, settings, and knowledge configuration. Model "
                "weights are never included."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="overview",
            title="What gets backed up",
            body=(
                "Essential state lives under your Qube user data folder. Manual and "
                "automatic backups exclude downloaded models, logs, and caches."
            ),
            target_getter=lambda h: _sv(h).state_backup_overview_hint,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="automatic_enabled",
            title="Automatic backup",
            body=(
                "Opt in to run a local backup on startup when the interval has elapsed. "
                "Disabled by default — nothing is copied until you turn this on."
            ),
            target_getter=lambda h: _sv(h).state_backup_auto_enabled_toggle,
            on_enter=_open_automatic,
        ),
        OnboardingStep(
            step_id="interval",
            title="Backup interval",
            body=(
                "Choose how often Qube should save a new automatic archive — for "
                "example every 30 days."
            ),
            target_getter=lambda h: _sv(h).state_backup_interval_selector,
            on_enter=_open_automatic,
        ),
        OnboardingStep(
            step_id="retention",
            title="Keep recent backups",
            body=(
                "Older automatic archives under backups/auto/ are deleted when this "
                "limit is exceeded."
            ),
            target_getter=lambda h: _sv(h).state_backup_retention_spin,
            on_enter=_open_automatic,
        ),
        OnboardingStep(
            step_id="wallpapers",
            title="Include wallpapers",
            body=(
                "Optional: include imported wallpaper images in automatic backups. "
                "Leave off to keep archives smaller."
            ),
            target_getter=lambda h: _sv(h).state_backup_include_wallpapers_cb,
            on_enter=_open_automatic,
        ),
        OnboardingStep(
            step_id="status",
            title="Last automatic run",
            body=(
                "Shows when the last automatic backup ran, whether it succeeded, and "
                "where the archive was saved."
            ),
            target_getter=lambda h: _sv(h).state_backup_status_hint,
            on_enter=_open_automatic,
        ),
        OnboardingStep(
            step_id="manual_create",
            title="Manual backup & restore",
            body=(
                "Create a backup now or restore from an archive. Restore saves a "
                "pre-restore safety snapshot first, then requires restarting Qube."
            ),
            target_getter=lambda h: _sv(h).state_backup_create_btn,
            on_enter=_open_manual,
        ),
        OnboardingStep(
            step_id="open_folder",
            title="Backups folder",
            body=(
                "Open your backups directory for manual archives, automatic archives, "
                "and pre-restore safety snapshots."
            ),
            target_getter=lambda h: _sv(h).state_backup_open_backups_btn,
            on_enter=_open_manual,
        ),
        make_settings_tour_finish_step("Backup & restore", _open),
    ]
    return OnboardingTour(host, steps)
