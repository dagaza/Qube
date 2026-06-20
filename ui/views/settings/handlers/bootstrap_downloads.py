"""Settings handlers for downloading missing bootstrap base models (#47–#49)."""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from core.app_settings import set_sidecar_enabled, set_sidecar_model_path
from core.bootstrap_download import resolve_model_destination
from core.bootstrap_manifest import BOOTSTRAP_MODELS, BootstrapModelId
from core.bootstrap_missing_models import (
    cognition_model_present,
    embedding_model_available,
    stt_model_available,
    tts_model_available,
)
from ui.components.brand_buttons import apply_brand_primary
from ui.components.prestige_dialog import PrestigeDialog

logger = logging.getLogger("Qube.UI.Settings.BootstrapDownloads")


def make_bootstrap_download_row(
    host,
    *,
    row_attr: str,
    label_attr: str,
    button_attr: str,
    handler_name: str,
    label_text: str,
    button_text: str,
) -> QWidget:
    row = QWidget()
    layout = QVBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(label_text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #f59e0b; font-size: 12px;")
    btn = QPushButton(button_text)
    apply_brand_primary(btn)
    btn.clicked.connect(getattr(host, handler_name))
    layout.addWidget(label)
    layout.addWidget(btn)
    setattr(host, row_attr, row)
    setattr(host, label_attr, label)
    setattr(host, button_attr, btn)
    return row


class BootstrapDownloadsHandlersMixin:
    """Download missing bootstrap models from Settings subsections."""

    def _download_bootstrap_stt(self) -> None:
        self._start_bootstrap_model_download(BootstrapModelId.WHISPER_SMALL)

    def _download_bootstrap_tts(self) -> None:
        self._start_bootstrap_model_download(BootstrapModelId.KOKORO_TTS)

    def _download_bootstrap_embedding(self) -> None:
        self._start_bootstrap_model_download(BootstrapModelId.NOMIC_EMBED)

    def _download_bootstrap_cognition(self) -> None:
        self._start_bootstrap_model_download(BootstrapModelId.SIDECAR_QWEN17)

    def _start_bootstrap_model_download(self, model_id: BootstrapModelId) -> None:
        if getattr(self, "_bootstrap_model_download_worker", None) is not None:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Download busy",
                "A model download is already in progress.",
                is_dark=is_dark,
            ).exec()
            return

        from workers.bootstrap_model_download_worker import BootstrapModelDownloadWorker

        spec = BOOTSTRAP_MODELS[model_id]
        worker = BootstrapModelDownloadWorker(model_id)
        self._bootstrap_model_download_worker = worker

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._bootstrap_download_dialog = PrestigeDialog(
            self.window(),
            f"Downloading {spec.label}",
            "Working… this may take a while on first download.",
            is_dark=is_dark,
            show_cancel=False,
        )
        self._bootstrap_download_dialog.show()

        def _apply_post_download() -> None:
            if model_id == BootstrapModelId.NOMIC_EMBED:
                dest = resolve_model_destination(model_id)
                if dest is not None and dest.is_file():
                    from core.app_settings import set_embedding_model_path

                    set_embedding_model_path(str(dest))
                if hasattr(self, "_reload_embedder_from_settings"):
                    self._reload_embedder_from_settings()
                if hasattr(self, "embedding_model_changed"):
                    self.embedding_model_changed.emit()
                self._sync_active_embedding_label()
                self._refresh_embedding_gguf_list()
            elif model_id == BootstrapModelId.WHISPER_SMALL:
                if hasattr(self, "_reload_stt_from_settings"):
                    self._reload_stt_from_settings()
                if hasattr(self, "stt_model_changed"):
                    self.stt_model_changed.emit()
                self._sync_active_stt_label()
                self._refresh_stt_model_list()
            elif model_id == BootstrapModelId.KOKORO_TTS:
                if hasattr(self, "_reload_tts_from_settings"):
                    self._reload_tts_from_settings()
                if hasattr(self, "tts_model_changed"):
                    self.tts_model_changed.emit()
                self._sync_active_tts_label()
                self._refresh_tts_model_list()
            elif model_id in {BootstrapModelId.SIDECAR_QWEN17, BootstrapModelId.SIDECAR_QWEN05}:
                dest = resolve_model_destination(model_id)
                if dest is not None and dest.is_file():
                    set_sidecar_enabled(True)
                    set_sidecar_model_path(str(dest))
                if hasattr(self, "_reload_sidecar_from_settings"):
                    self._reload_sidecar_from_settings()
                if hasattr(self, "cognition_model_changed"):
                    self.cognition_model_changed.emit()
                self._sync_active_cognition_label()
                self._refresh_cognition_gguf_list()
            self._sync_bootstrap_download_visibility()

        def _on_ok(used_mock: bool) -> None:
            try:
                dlg = getattr(self, "_bootstrap_download_dialog", None)
                if dlg is not None:
                    dlg.accept()
            except Exception:
                pass
            self._bootstrap_model_download_worker = None
            if not used_mock:
                _apply_post_download()
            else:
                self._sync_bootstrap_download_visibility()
            if used_mock:
                body = (
                    f"Mock download finished for {spec.label}. No files were written — "
                    "guards and notifications will stay until a real download completes. "
                    "Unset QUBE_BOOTSTRAP_MOCK_DOWNLOAD or use QUBE_BOOTSTRAP_REAL_DOWNLOAD=1."
                )
                title = "Mock download complete"
            else:
                body = f"{spec.label} is ready to use."
                title = "Download complete"
            PrestigeDialog(
                self.window(),
                title,
                body,
                is_dark=is_dark,
            ).exec()

        def _on_failed(err: str) -> None:
            try:
                dlg = getattr(self, "_bootstrap_download_dialog", None)
                if dlg is not None:
                    dlg.reject()
            except Exception:
                pass
            self._bootstrap_model_download_worker = None
            PrestigeDialog(
                self.window(),
                "Download failed",
                str(err or "Model download failed."),
                is_dark=is_dark,
                tone="danger",
            ).exec()

        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)
        worker.start()

    def _sync_bootstrap_download_visibility(self) -> None:
        rows = (
            ("stt_bootstrap_download_row", stt_model_available),
            ("tts_bootstrap_download_row", tts_model_available),
            ("embedding_bootstrap_download_row", embedding_model_available),
            ("cognition_bootstrap_download_row", cognition_model_present),
        )
        for attr, available_fn in rows:
            row = getattr(self, attr, None)
            if row is not None:
                row.setVisible(not available_fn())
