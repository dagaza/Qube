"""Settings handler mixins extracted from SettingsView."""

from ui.views.settings.handlers.ai_models import AiModelsHandlersMixin
from ui.views.settings.handlers.bootstrap_downloads import BootstrapDownloadsHandlersMixin
from ui.views.settings.handlers.companion import CompanionHandlersMixin
from ui.views.settings.handlers.diagnostics import DiagnosticsHandlersMixin
from ui.views.settings.handlers.generation import GenerationMixin
from ui.views.settings.handlers.knowledge import KnowledgeHandlersMixin
from ui.views.settings.handlers.licensing import LicenseHandlersMixin
from ui.views.settings.handlers.uninstall import UninstallHandlersMixin
from ui.views.settings.handlers.memory import MemoryHandlersMixin
from ui.views.settings.handlers.persistence import PersistenceHandlersMixin
from ui.views.settings.handlers.prestige_menu import PrestigeMenuMixin
from ui.views.settings.handlers.styling import StylingMixin
from ui.views.settings.handlers.support import SupportHandlersMixin
from ui.views.settings.handlers.themes import ThemesHandlersMixin
from ui.views.settings.handlers.voice import VoiceHandlersMixin

__all__ = [
    "AiModelsHandlersMixin",
    "BootstrapDownloadsHandlersMixin",
    "CompanionHandlersMixin",
    "DiagnosticsHandlersMixin",
    "GenerationMixin",
    "KnowledgeHandlersMixin",
    "LicenseHandlersMixin",
    "UninstallHandlersMixin",
    "MemoryHandlersMixin",
    "PersistenceHandlersMixin",
    "PrestigeMenuMixin",
    "StylingMixin",
    "SupportHandlersMixin",
    "ThemesHandlersMixin",
    "VoiceHandlersMixin",
]
