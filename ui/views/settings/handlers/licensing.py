"""Settings handler mixin: license import and status."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog

from core.licensing.store import (
    format_license_status_text,
    import_license_from_path,
    license_summary,
    remove_license,
)
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.SettingsLicense")

_LICENSE_FILE_FILTER = (
    "Qube license (*.qube-license *.json);;Qube license (*.qube-license);;"
    "JSON (*.json);;All files (*)"
)


class LicenseHandlersMixin:
    def _refresh_license_status_ui(self) -> None:
        status_lbl = getattr(self, "license_status_lbl", None)
        if status_lbl is None:
            return
        summary = license_summary()
        status_lbl.setText(format_license_status_text(summary))
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
                "Feature gating is not active during the MIT launch period.",
                is_dark=is_dark,
            ).exec()
            logger.info("Imported license from %s (tier=%s)", path, result.document.tier.value)
        else:
            PrestigeDialog(
                self.window(),
                "Import failed",
                result.error or "The license file could not be imported.",
                is_dark=is_dark,
            ).exec()
            logger.warning("License import failed for %s: %s", path, result.error)
        self._refresh_license_status_ui()

    def _on_remove_license_clicked(self) -> None:
        summary = license_summary()
        if not summary.get("cached"):
            self._refresh_license_status_ui()
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
