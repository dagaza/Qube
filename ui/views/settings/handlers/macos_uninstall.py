"""macOS uninstall handlers for Settings → Help."""

from __future__ import annotations

from core.macos_uninstall import is_macos_uninstall_available, request_macos_uninstall
from ui.components.prestige_dialog import PrestigeDialog


class MacOSUninstallHandlersMixin:
    def _confirm_and_uninstall_qube(self, *, keep_user_data: bool) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if keep_user_data:
            title = "Remove Qube app only?"
            message = (
                "Qube will quit and remove the application from /Applications "
                "(or ~/Applications).\n\n"
                "Your data in ~/.qube — models, library indexes, memory, and "
                "settings — will be kept."
            )
            confirm_text = "REMOVE APP"
        else:
            title = "Uninstall Qube?"
            message = (
                "Qube will quit and remove the application plus all local data, "
                "including:\n"
                "• ~/.qube (models, library, memory, logs, settings)\n"
                "• macOS preference and cache files\n\n"
                "This cannot be undone. Export a knowledge pack first if you "
                "need a backup."
            )
            confirm_text = "UNINSTALL"

        dlg = PrestigeDialog(
            self.window(),
            title,
            message,
            is_dark=is_dark,
            tone="danger",
            dialog_width=520,
            confirm_text=confirm_text,
        )
        if not dlg.exec():
            return

        ok, detail = request_macos_uninstall(keep_user_data=keep_user_data)
        if ok:
            return

        PrestigeDialog(
            self.window(),
            "Uninstall unavailable",
            detail,
            is_dark=is_dark,
        ).exec()

    def _on_uninstall_qube_all_data_clicked(self) -> None:
        if not is_macos_uninstall_available():
            return
        self._confirm_and_uninstall_qube(keep_user_data=False)

    def _on_uninstall_qube_keep_data_clicked(self) -> None:
        if not is_macos_uninstall_available():
            return
        self._confirm_and_uninstall_qube(keep_user_data=True)
