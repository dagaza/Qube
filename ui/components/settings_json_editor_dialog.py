"""In-app editor for ``~/.qube/settings.json`` with validation and external change detection."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from core.settings_store import (
    default_user_settings_path,
    get_settings_store,
    open_user_settings_in_editor,
)
from ui.components.brand_buttons import apply_brand_primary
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.SettingsJsonEditor")


class SettingsJsonEditorDialog(QDialog):
    settings_applied = pyqtSignal(set)

    def __init__(self, parent=None, *, is_dark: bool | None = None):
        super().__init__(parent)
        if is_dark is None:
            is_dark = getattr(parent.window() if parent else None, "_is_dark_theme", True)
        self._is_dark = is_dark
        self._dirty = False
        self._disk_mtime: float | None = None
        self._suppress_dirty = False
        self._external_prompt_open = False

        self.setWindowTitle("settings.json")
        self.setModal(False)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(760, 580)

        self._build_ui()
        self._apply_theme_styles()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._poll_disk_changes)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_watcher_file_changed)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._poll_disk_changes)

        self._editor.textChanged.connect(self._on_editor_text_changed)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        self.container = QFrame()
        self.container.setObjectName("SettingsJsonEditorContainer")
        root = QVBoxLayout(self.container)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(12)

        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self.header_title_lbl = QLabel("USER SETTINGS")
        self.header_title_lbl.setObjectName("SettingsJsonEditorTitle")
        self.path_lbl = QLabel(str(default_user_settings_path()))
        self.path_lbl.setObjectName("SettingsJsonEditorPath")
        self.path_lbl.setWordWrap(True)
        title_col.addWidget(self.header_title_lbl)
        title_col.addWidget(self.path_lbl)
        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("SettingsJsonEditorClose")
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(self.close)
        header_row.addLayout(title_col, 1)
        header_row.addWidget(self.close_btn, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header_row)

        self.external_banner = QLabel("")
        self.external_banner.setObjectName("SettingsJsonEditorBanner")
        self.external_banner.setWordWrap(True)
        self.external_banner.hide()
        root.addWidget(self.external_banner)

        self._editor = QPlainTextEdit()
        self._editor.setObjectName("SettingsJsonEditorText")
        self._editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._editor.setTabStopDistance(24)
        font = QFont("Cascadia Mono")
        if not font.exactMatch():
            font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setPointSize(10)
        self._editor.setFont(font)
        self._editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root.addWidget(self._editor, stretch=1)

        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("SettingsJsonEditorStatus")
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self.format_btn = QPushButton("Format")
        self.format_btn.setToolTip("Beautify JSON (indent, sorted keys).")
        self.format_btn.clicked.connect(self._on_format_clicked)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip("Reload from disk.")
        self.reload_btn.clicked.connect(
            lambda: self._reload_from_disk(force=True, apply_runtime=True)
        )
        self.external_btn = QPushButton("Open externally")
        self.external_btn.setToolTip("Open in your default system editor.")
        self.external_btn.clicked.connect(self._on_open_external)
        self.save_btn = QPushButton("Save")
        apply_brand_primary(self.save_btn, icon_name="fa5s.save")
        self.save_btn.setToolTip("Validate, save to disk, and apply settings.")
        self.save_btn.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(self.format_btn)
        btn_row.addWidget(self.reload_btn)
        btn_row.addWidget(self.external_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.save_btn)
        root.addLayout(btn_row)

        outer.addWidget(self.container)

    def _apply_theme_styles(self) -> None:
        is_dark = self._is_dark
        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.12)" if is_dark else "#cbd5e1"
        surface = "#181825" if is_dark else "#f8fafc"
        banner_bg = "#45475a" if is_dark else "#e2e8f0"
        ok_color = "#a6e3a1" if is_dark else "#15803d"
        err_color = "#f38ba8" if is_dark else "#b91c1c"
        self._ok_color = ok_color
        self._err_color = err_color

        self.container.setStyleSheet(
            f"""
            QFrame#SettingsJsonEditorContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel#SettingsJsonEditorTitle {{
                color: {accent};
                font-weight: bold;
                font-size: 11px;
                letter-spacing: 2px;
            }}
            QLabel#SettingsJsonEditorPath {{
                color: {fg};
                font-size: 12px;
                opacity: 0.85;
            }}
            QLabel#SettingsJsonEditorBanner {{
                background: {banner_bg};
                color: {fg};
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLabel#SettingsJsonEditorStatus {{
                color: {fg};
                font-size: 12px;
                min-height: 18px;
            }}
            QPlainTextEdit#SettingsJsonEditorText {{
                background: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 12px 14px;
                selection-background-color: {accent};
            }}
            QPushButton#SettingsJsonEditorClose {{
                background: transparent;
                color: {fg};
                border: 1px solid {border};
                border-radius: 8px;
                font-size: 14px;
            }}
            QPushButton#SettingsJsonEditorClose:hover {{
                background: rgba(255, 255, 255, 0.06);
            }}
        """
        )
        btn_style = f"""
            QPushButton {{
                padding: 10px 16px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 12px;
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        for btn in (self.format_btn, self.reload_btn, self.external_btn):
            btn.setStyleSheet(btn_style)

    def refresh_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._apply_theme_styles()
        self._update_validation_status()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._ensure_watched()
        self._poll_timer.start()
        if not self._editor.toPlainText().strip():
            self.load_from_disk()

    def closeEvent(self, event) -> None:
        if self._dirty:
            dlg = PrestigeDialog(
                self,
                "Unsaved changes",
                "Close the editor without saving your changes to settings.json?",
                is_dark=self._is_dark,
            )
            if not dlg.exec():
                event.ignore()
                return
        self._poll_timer.stop()
        super().closeEvent(event)

    def load_from_disk(self) -> None:
        store = get_settings_store()
        text = store.read_file_text()
        formatted, err = store.format_json_text(text)
        self._set_editor_text(formatted if err is None else text)
        self._refresh_disk_mtime()
        self._dirty = False
        self.external_banner.hide()
        self._update_validation_status(err)

    def _ensure_watched(self) -> None:
        path = str(default_user_settings_path())
        if path not in self._watcher.files():
            get_settings_store().ensure_user_settings_file()
            self._watcher.addPath(path)
        parent = str(default_user_settings_path().parent)
        if parent not in self._watcher.directories():
            self._watcher.addPath(parent)

    def _refresh_disk_mtime(self) -> None:
        path = default_user_settings_path()
        if path.is_file():
            try:
                self._disk_mtime = path.stat().st_mtime
            except OSError:
                self._disk_mtime = None
        else:
            self._disk_mtime = None

    def _set_editor_text(self, text: str) -> None:
        self._suppress_dirty = True
        self._editor.setPlainText(text)
        self._suppress_dirty = False

    def _on_editor_text_changed(self) -> None:
        if self._suppress_dirty:
            return
        self._dirty = True
        self.external_banner.hide()
        self._update_validation_status()

    def _update_validation_status(self, extra_error: str | None = None) -> None:
        if extra_error:
            self.status_lbl.setText(extra_error)
            self.status_lbl.setStyleSheet(f"color: {self._err_color};")
            return
        validation = get_settings_store().validate_json_text(self._editor.toPlainText())
        if validation.ok:
            msg = "Valid JSON"
            if validation.skipped_keys:
                msg += f" — unknown keys ignored on save: {', '.join(validation.skipped_keys[:4])}"
                if len(validation.skipped_keys) > 4:
                    msg += ", …"
            if self._dirty:
                msg += " (unsaved)"
            self.status_lbl.setText(msg)
            self.status_lbl.setStyleSheet(f"color: {self._ok_color};")
        else:
            self.status_lbl.setText(validation.error or "Invalid JSON")
            self.status_lbl.setStyleSheet(f"color: {self._err_color};")

    def _on_format_clicked(self) -> None:
        store = get_settings_store()
        formatted, err = store.format_json_text(self._editor.toPlainText())
        if err:
            self._update_validation_status(err)
            return
        self._set_editor_text(formatted)
        self._dirty = True
        self._update_validation_status()

    def _on_save_clicked(self) -> None:
        store = get_settings_store()
        result = store.save_from_json_text(self._editor.toPlainText())
        if not result.ok:
            self._update_validation_status(result.parse_error)
            PrestigeDialog(
                self,
                "Cannot save",
                result.parse_error or "Fix JSON errors before saving.",
                is_dark=self._is_dark,
            ).exec()
            return
        self._set_editor_text(store.read_file_text())
        self._refresh_disk_mtime()
        self._dirty = False
        self.external_banner.hide()
        self._update_validation_status()
        if result.skipped_keys:
            logger.info("Ignored unknown settings keys on save: %s", result.skipped_keys)
        if result.changed_keys:
            self.settings_applied.emit(set(result.changed_keys))
        self.status_lbl.setText("Saved and applied.")

    def _on_open_external(self) -> None:
        open_user_settings_in_editor()

    def _on_watcher_file_changed(self, _path: str) -> None:
        self._debounce.start()

    def _poll_disk_changes(self) -> None:
        path = default_user_settings_path()
        if not path.is_file():
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if self._disk_mtime is not None and mtime == self._disk_mtime:
            return
        if self._external_prompt_open:
            return
        if self._dirty:
            self._external_prompt_open = True
            dlg = PrestigeDialog(
                self,
                "File changed on disk",
                f"{path.name} was modified outside this editor. "
                "Reload from disk? Unsaved changes in the editor will be lost.",
                is_dark=self._is_dark,
            )
            self._external_prompt_open = False
            if dlg.exec():
                self._reload_from_disk(force=True)
            else:
                self.external_banner.setText(
                    "Disk copy differs — save here or choose Reload to discard local edits."
                )
                self.external_banner.show()
                self._refresh_disk_mtime()
            return
        self._reload_from_disk(force=False, apply_runtime=True)

    def _reload_from_disk(self, *, force: bool, apply_runtime: bool = False) -> None:
        before = get_settings_store().effective_snapshot()
        result = get_settings_store().reload_from_disk()
        if not result.ok:
            PrestigeDialog(
                self,
                "Invalid settings.json",
                result.parse_error or "The file on disk could not be parsed.",
                is_dark=self._is_dark,
            ).exec()
            self._update_validation_status(result.parse_error)
            return
        self.load_from_disk()
        if apply_runtime or force:
            after = get_settings_store().effective_snapshot()
            changed = [
                key
                for key in get_settings_store().schema
                if before.get(key) != after.get(key)
            ]
            if changed:
                self.settings_applied.emit(set(changed))
