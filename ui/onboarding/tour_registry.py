"""Registry of page guided tour builders keyed by stable tour id."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ui.components.onboarding_tour import OnboardingTour

TourBuilder = Callable[["QWidget"], "OnboardingTour"]

_TOUR_BUILDERS: dict[str, TourBuilder] = {}

# Human-readable names for tours and UI labels.
TOUR_DISPLAY_NAMES: dict[str, str] = {
    "conversations": "Conversations",
    "library": "Library",
    "memory_manager": "Memory Manager",
    "model_manager": "Model Manager",
    "telemetry": "Advanced Telemetry",
    "settings.voice_audio": "Voice & Audio settings",
    "settings.ai_models": "AI & Models settings",
    "settings.memory": "Memory settings",
    "settings.knowledge": "Knowledge settings",
    "settings.integrations": "Integrations settings",
    "settings.general": "General settings",
    "settings.appearance_themes": "Themes settings",
    "settings.companion_desktop": "Desktop Companion settings",
    "settings.notifications": "Notifications settings",
    "settings.help": "Help settings",
    "settings.about": "About settings",
    "settings.contact_feedback": "Contact & Feedback settings",
    "settings.privacy_data": "Privacy & data settings",
    "settings.system_backup": "Backup & restore settings",
    "settings.diagnostics": "Diagnostics settings",
    "settings.license": "License settings",
    "settings.advanced": "Advanced settings",
}


def register_tour(tour_id: str, builder: TourBuilder) -> None:
    _TOUR_BUILDERS[tour_id] = builder


def get_tour_builder(tour_id: str) -> TourBuilder | None:
    return _TOUR_BUILDERS.get(tour_id)


def build_tour(tour_id: str, host) -> "OnboardingTour | None":
    builder = get_tour_builder(tour_id)
    if builder is None:
        return None
    return builder(host)


def list_registered_tour_ids() -> list[str]:
    return sorted(_TOUR_BUILDERS.keys())


def tour_display_name(tour_id: str) -> str:
    return TOUR_DISPLAY_NAMES.get(tour_id, tour_id.replace("_", " ").replace(".", " — "))


def settings_section_tour_id(section_id: str) -> str:
    """Map settings section id (e.g. voice.audio) to tour registry id."""
    return f"settings.{section_id.replace('.', '_')}"


def _register_all_tours() -> None:
    from ui.onboarding.tours.conversations import build_conversations_tour
    from ui.onboarding.tours.library import build_library_tour
    from ui.onboarding.tours.memory_manager import build_memory_manager_tour
    from ui.onboarding.tours.model_manager import build_model_manager_tour
    from ui.onboarding.tours.telemetry import build_telemetry_tour
    from ui.onboarding.tours.settings.advanced import build_settings_advanced_tour
    from ui.onboarding.tours.settings.backup_restore import (
        build_settings_system_backup_tour,
    )
    from ui.onboarding.tours.settings.diagnostics import build_settings_diagnostics_tour
    from ui.onboarding.tours.settings.license import build_settings_license_tour
    from ui.onboarding.tours.settings.privacy_data import build_settings_privacy_data_tour
    from ui.onboarding.tours.settings.appearance_themes import (
        build_settings_appearance_themes_tour,
    )
    from ui.onboarding.tours.settings.ai_models import build_settings_ai_models_tour
    from ui.onboarding.tours.settings.companion_desktop import (
        build_settings_companion_desktop_tour,
    )
    from ui.onboarding.tours.settings.about import build_settings_about_tour
    from ui.onboarding.tours.settings.contact_feedback import (
        build_settings_contact_feedback_tour,
    )
    from ui.onboarding.tours.settings.general import build_settings_general_tour
    from ui.onboarding.tours.settings.help import build_settings_help_tour
    from ui.onboarding.tours.settings.integrations import build_settings_integrations_tour
    from ui.onboarding.tours.settings.knowledge import build_settings_knowledge_tour
    from ui.onboarding.tours.settings.memory import build_settings_memory_tour
    from ui.onboarding.tours.settings.notifications import (
        build_settings_notifications_tour,
    )
    from ui.onboarding.tours.settings.voice_audio import build_settings_voice_audio_tour

    register_tour("conversations", build_conversations_tour)
    register_tour("library", build_library_tour)
    register_tour("memory_manager", build_memory_manager_tour)
    register_tour("model_manager", build_model_manager_tour)
    register_tour("telemetry", build_telemetry_tour)
    register_tour("settings.voice_audio", build_settings_voice_audio_tour)
    register_tour("settings.ai_models", build_settings_ai_models_tour)
    register_tour("settings.memory", build_settings_memory_tour)
    register_tour("settings.knowledge", build_settings_knowledge_tour)
    register_tour("settings.integrations", build_settings_integrations_tour)
    register_tour("settings.general", build_settings_general_tour)
    register_tour("settings.appearance_themes", build_settings_appearance_themes_tour)
    register_tour("settings.companion_desktop", build_settings_companion_desktop_tour)
    register_tour("settings.notifications", build_settings_notifications_tour)
    register_tour("settings.help", build_settings_help_tour)
    register_tour("settings.about", build_settings_about_tour)
    register_tour("settings.contact_feedback", build_settings_contact_feedback_tour)
    register_tour("settings.privacy_data", build_settings_privacy_data_tour)
    register_tour("settings.system_backup", build_settings_system_backup_tour)
    register_tour("settings.diagnostics", build_settings_diagnostics_tour)
    register_tour("settings.license", build_settings_license_tour)
    register_tour("settings.advanced", build_settings_advanced_tour)


_register_all_tours()
