"""Settings view package."""

__all__ = ["SettingsView"]


def __getattr__(name: str):
    if name == "SettingsView":
        from ui.views.settings.settings_view import SettingsView

        return SettingsView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
