"""Settings handler mixin: license import and status."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog

from core.licensing.store import (
    import_license_from_path,
    license_summary,
    remove_license,
)
from ui.views.settings.license_status_ui import sync_license_status_presentation
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.SettingsLicense")

_LICENSE_FILE_FILTER = (
    "Qube license (*.qube-license *.json);;Qube license (*.qube-license);;"
    "JSON (*.json);;All files (*)"
)


class LicenseHandlersMixin:
    def _play_license_import_celebration(self) -> None:
        """Border fireworks around the License card (same effect as composer @ discovery)."""
        from PyQt6.QtCore import QTimer

        from ui.components.celebration_burst import show_border_fireworks

        anchor = getattr(self, "license_section_card", None)
        if anchor is None:
            anchor = getattr(self, "import_license_btn", None)
        if anchor is None or not anchor.isVisible():
            return

        stack = getattr(self, "settings_section_stack", None)
        overlay_parent = stack.currentWidget() if stack is not None else None
        if overlay_parent is None:
            win = self.window()
            overlay_parent = win if win is not None else self

        def _start() -> None:
            show_border_fireworks(
                anchor,
                overlay_parent=overlay_parent,
                duration_ms=3200,
            )

        QTimer.singleShot(80, _start)

    def _refresh_license_status_ui(self) -> None:
        summary = license_summary()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        sync_license_status_presentation(self, summary, is_dark=is_dark)
        cached = bool(summary.get("cached"))
        remove_btn = getattr(self, "remove_license_btn", None)
        if remove_btn is not None:
            remove_btn.setEnabled(cached)

    def _on_import_license_clicked(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        path, _ = QFileDialog.getOpenFileName(
            self.window(),
            "Import Qube license",
            str(Path.home()),
            _LICENSE_FILE_FILTER,
        )
        if not path:
            return
        result = import_license_from_path(Path(path))
        if result.ok and result.document is not None:
            tier = result.document.tier.value.title()
            PrestigeDialog(
                self.window(),
                "License imported",
                f"Recorded a {tier} license locally. "
                "Pro and Team capabilities are now available on this device.",
                is_dark=is_dark,
            ).exec()
            logger.info("Imported license from %s (tier=%s)", path, result.document.tier.value)
            self._play_license_import_celebration()
        else:
            PrestigeDialog(
                self.window(),
                "Import failed",
                result.error or "The license file could not be imported.",
                is_dark=is_dark,
            ).exec()
            logger.warning("License import failed for %s: %s", path, result.error)
        self._refresh_license_status_ui()
        if hasattr(self, "_sync_library_pro_features"):
            self._sync_library_pro_features()
        if hasattr(self, "_sync_share_themes_pro_features"):
            self._sync_share_themes_pro_features()
        if hasattr(self, "_sync_custom_model_paths_pro_features"):
            self._sync_custom_model_paths_pro_features()
        if hasattr(self, "_sync_wakeword_pro_features"):
            self._sync_wakeword_pro_features()
        if hasattr(self, "_sync_mcp_filesystem_pro_features"):
            self._sync_mcp_filesystem_pro_features()
        if hasattr(self, "_sync_deep_research_profile_selector"):
            self._sync_deep_research_profile_selector()

    def _on_remove_license_clicked(self) -> None:
        summary = license_summary()
        if not summary.get("cached"):
            self._refresh_license_status_ui()
            if hasattr(self, "_sync_library_pro_features"):
                self._sync_library_pro_features()
            if hasattr(self, "_sync_share_themes_pro_features"):
                self._sync_share_themes_pro_features()
            if hasattr(self, "_sync_custom_model_paths_pro_features"):
                self._sync_custom_model_paths_pro_features()
            if hasattr(self, "_sync_wakeword_pro_features"):
                self._sync_wakeword_pro_features()
            if hasattr(self, "_sync_mcp_filesystem_pro_features"):
                self._sync_mcp_filesystem_pro_features()
            if hasattr(self, "_sync_deep_research_profile_selector"):
                self._sync_deep_research_profile_selector()
            return

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Remove license?",
            "Remove the cached license from this device? "
            "You can import the same file again later.",
            is_dark=is_dark,
            tone="danger",
            dialog_width=480,
            confirm_text="REMOVE",
        )
        if not dlg.exec():
            return

        removed = remove_license()
        if removed:
            PrestigeDialog(
                self.window(),
                "License removed",
                "The cached license was deleted from this device.",
                is_dark=is_dark,
            ).exec()
            logger.info("Removed cached license")
        self._refresh_license_status_ui()
        if hasattr(self, "_sync_library_pro_features"):
            self._sync_library_pro_features()
        if hasattr(self, "_sync_share_themes_pro_features"):
            self._sync_share_themes_pro_features()
        if hasattr(self, "_sync_custom_model_paths_pro_features"):
            self._sync_custom_model_paths_pro_features()
        if hasattr(self, "_sync_wakeword_pro_features"):
            self._sync_wakeword_pro_features()
        if hasattr(self, "_sync_mcp_filesystem_pro_features"):
            self._sync_mcp_filesystem_pro_features()
        if hasattr(self, "_sync_deep_research_profile_selector"):
            self._sync_deep_research_profile_selector()
