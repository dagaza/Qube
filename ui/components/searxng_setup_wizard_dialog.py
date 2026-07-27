"""SearXNG setup wizard — detect, test, and configure a local instance."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from core.app_settings import get_discovery_searxng_base_url
from core.knowledge.credentials import resolve_credential
from core.knowledge.discovery.searxng import SEARXNG_DISCOVERY_PROVIDER_ID
from core.knowledge.discovery.searxng_wizard import (
    SEARXNG_DOCKER_RUN_HINT,
    SearXNGProbeResult,
    docker_cli_available,
    docker_searxng_container_running,
    normalize_searxng_base_url,
    probe_searxng_base_url,
    scan_local_searxng_candidates,
)
from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_DIALOG_INPUT,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_MUTED_LABEL,
    PRESTIGE_SOURCE_CONTAINER,
)
from ui.components.prestige_dialog import _center_dialog_on_host, _resolve_is_dark_from_parent

_DIALOG_WIDTH = 560


class _SearXNGScanWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, *, api_key: str | None, parent=None) -> None:
        super().__init__(parent)
        self._api_key = api_key

    def run(self) -> None:
        hits = scan_local_searxng_candidates(api_key=self._api_key)
        self.finished.emit(hits)


class _SearXNGTestWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._api_key = api_key

    def run(self) -> None:
        result = probe_searxng_base_url(self._base_url, api_key=self._api_key)
        self.finished.emit(result)


class SearXNGSetupWizardDialog(QDialog):
    """Modal wizard for SearXNG URL detection, connectivity test, and apply."""

    def __init__(self, host, *, is_dark: bool | None = None, parent=None) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)
        self._host = host
        self._is_dark = is_dark
        self._scan_worker: _SearXNGScanWorker | None = None
        self._test_worker: _SearXNGTestWorker | None = None
        self._last_probe_ok = False

        theme = theme_for(is_dark=is_dark)
        hover_bg = with_alpha(theme.text_primary, 0.05)

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(_DIALOG_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("SearXNGSetupWizardContainer")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="SearXNGSetupWizardContainer",
            )
            + """
            QLabel {
                color: inherit;
                background: transparent;
                border: none;
            }
            """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SET UP SEARXNG")
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
        )
        inner.addWidget(header)

        intro = QLabel(
            "Point Qube at your self-hosted SearXNG instance. Queries go to your "
            "server — upstream engines depend on your SearXNG configuration."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(theme.style(PRESTIGE_BODY_LABEL, font_size="14px"))
        inner.addWidget(intro)

        detect_row = QHBoxLayout()
        detect_row.setSpacing(10)
        self.detect_btn = QPushButton("Detect local")
        self.detect_btn.setToolTip(
            "Scan common localhost ports for a running SearXNG JSON API."
        )
        self.detect_btn.clicked.connect(self._on_detect_clicked)
        detect_row.addWidget(self.detect_btn)
        self.detect_status = QLabel("")
        self.detect_status.setWordWrap(True)
        self.detect_status.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="12px"))
        detect_row.addWidget(self.detect_status, stretch=1)
        inner.addLayout(detect_row)

        self.detected_combo = QComboBox()
        self.detected_combo.setVisible(False)
        self.detected_combo.currentIndexChanged.connect(self._on_detected_selected)
        inner.addWidget(self.detected_combo)

        url_label = QLabel("Base URL")
        url_label.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="11px"))
        inner.addWidget(url_label)

        self.url_field = QLineEdit()
        self.url_field.setPlaceholderText("http://127.0.0.1:8080")
        self.url_field.setToolTip(
            "SearXNG instance root URL (JSON search at /search?format=json)."
        )
        self.url_field.setStyleSheet(theme.style(PRESTIGE_DIALOG_INPUT))
        configured = normalize_searxng_base_url(get_discovery_searxng_base_url())
        if configured:
            self.url_field.setText(configured)
        inner.addWidget(self.url_field)

        key_label = QLabel("API key (optional)")
        key_label.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="11px"))
        inner.addWidget(key_label)

        self.api_key_field = QLineEdit()
        self.api_key_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_field.setPlaceholderText("Bearer token if your instance requires auth")
        self.api_key_field.setStyleSheet(theme.style(PRESTIGE_DIALOG_INPUT))
        secret = (resolve_credential(SEARXNG_DISCOVERY_PROVIDER_ID).secret or "").strip()
        if secret:
            self.api_key_field.setText(secret)
        inner.addWidget(self.api_key_field)

        test_row = QHBoxLayout()
        test_row.setSpacing(10)
        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self._on_test_clicked)
        test_row.addWidget(self.test_btn)
        self.test_status = QLabel("Run a test before saving.")
        self.test_status.setWordWrap(True)
        self.test_status.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="12px"))
        test_row.addWidget(self.test_status, stretch=1)
        inner.addLayout(test_row)

        self.docker_hint = QLabel("")
        self.docker_hint.setWordWrap(True)
        self.docker_hint.setStyleSheet(theme.style(PRESTIGE_MUTED_LABEL, font_size="12px"))
        self._refresh_docker_hint()
        inner.addWidget(self.docker_hint)

        self.switch_tier_cb = QCheckBox("Switch privacy tier to Self-hosted SearXNG")
        self.switch_tier_cb.setChecked(True)
        self.switch_tier_cb.setToolTip(
            "Recommended after a successful test so @internet uses your instance."
        )
        inner.addWidget(self.switch_tier_cb)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setStyleSheet(
            theme.style(PRESTIGE_GHOST_BUTTON)
            + f"QPushButton:hover {{ background: {hover_bg}; }}"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.save_btn = QPushButton("SAVE")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(theme.style(PRESTIGE_DIALOG_CONFIRM, accent=theme.link))
        self.save_btn.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(self.save_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)
        self._style_secondary_buttons(theme, hover_bg)

    def _style_secondary_buttons(self, theme, hover_bg: str) -> None:
        ghost = (
            theme.style(PRESTIGE_GHOST_BUTTON)
            + f"QPushButton:hover {{ background: {hover_bg}; }}"
        )
        for btn in (self.detect_btn, self.test_btn):
            btn.setStyleSheet(ghost)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 — Qt API
        super().showEvent(event)
        _center_dialog_on_host(self)

    def _api_key_value(self) -> str | None:
        value = self.api_key_field.text().strip()
        return value or None

    def _refresh_docker_hint(self) -> None:
        if docker_searxng_container_running():
            self.docker_hint.setText(
                "Docker: a container named like “searxng” appears to be running."
            )
            return
        if docker_cli_available():
            self.docker_hint.setText(
                "Docker is available but no local SearXNG responded. Example:\n"
                f"{SEARXNG_DOCKER_RUN_HINT}"
            )
            return
        self.docker_hint.setText(
            "Install SearXNG locally or via Docker, enable JSON search, then Detect local."
        )

    def _set_busy(self, busy: bool) -> None:
        self.detect_btn.setEnabled(not busy)
        self.test_btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy and self._last_probe_ok)

    def _on_detect_clicked(self) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        self.detect_status.setText("Scanning localhost…")
        self.detected_combo.clear()
        self.detected_combo.setVisible(False)
        self._set_busy(True)
        self._scan_worker = _SearXNGScanWorker(api_key=self._api_key_value(), parent=self)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.start()

    def _on_scan_finished(self, hits: object) -> None:
        self._set_busy(False)
        results = list(hits) if isinstance(hits, list) else []
        ok_hits = [r for r in results if isinstance(r, SearXNGProbeResult) and r.ok]
        if not ok_hits:
            self.detect_status.setText("No local SearXNG instance found on common ports.")
            self._refresh_docker_hint()
            return

        self.detect_status.setText(
            f"Found {len(ok_hits)} instance(s) — pick one or edit the URL below."
        )
        self.detected_combo.clear()
        for hit in ok_hits:
            self.detected_combo.addItem(hit.base_url, hit.base_url)
        self.detected_combo.setVisible(len(ok_hits) > 1)
        self.url_field.setText(ok_hits[0].base_url)
        self._apply_probe_result(ok_hits[0], from_test=False)

    def _on_detected_selected(self, index: int) -> None:
        if index < 0:
            return
        url = self.detected_combo.itemData(index)
        if url:
            self.url_field.setText(str(url))

    def _on_test_clicked(self) -> None:
        if self._test_worker is not None and self._test_worker.isRunning():
            return
        self.test_status.setText("Testing…")
        self._set_busy(True)
        self._test_worker = _SearXNGTestWorker(
            base_url=self.url_field.text(),
            api_key=self._api_key_value(),
            parent=self,
        )
        self._test_worker.finished.connect(self._on_test_finished)
        self._test_worker.start()

    def _on_test_finished(self, result: object) -> None:
        self._set_busy(False)
        if not isinstance(result, SearXNGProbeResult):
            self.test_status.setText("Test failed unexpectedly.")
            self._last_probe_ok = False
            self.save_btn.setEnabled(False)
            return
        self._apply_probe_result(result, from_test=True)

    def _apply_probe_result(self, result: SearXNGProbeResult, *, from_test: bool) -> None:
        self._last_probe_ok = result.ok
        target = self.test_status if from_test else self.detect_status
        target.setText(result.message)
        if result.ok and result.base_url:
            self.url_field.setText(result.base_url)
        self.save_btn.setEnabled(result.ok)

    def _on_save_clicked(self) -> None:
        if not self._last_probe_ok:
            return
        normalized = normalize_searxng_base_url(self.url_field.text())
        if not normalized:
            self.test_status.setText("Enter a valid base URL before saving.")
            return
        self._saved_url = normalized
        self._saved_api_key = self._api_key_value()
        self._saved_switch_tier = self.switch_tier_cb.isChecked()
        self.accept()

    @property
    def saved_url(self) -> str:
        return getattr(self, "_saved_url", "")

    @property
    def saved_api_key(self) -> str | None:
        return getattr(self, "_saved_api_key", None)

    @property
    def saved_switch_tier(self) -> bool:
        return bool(getattr(self, "_saved_switch_tier", False))


def open_searxng_setup_wizard(
    host,
    *,
    is_dark: bool | None = None,
    parent=None,
) -> bool:
    """Show SearXNG wizard; returns True when the user saved settings."""
    dlg = SearXNGSetupWizardDialog(
        host,
        is_dark=is_dark,
        parent=parent or (host.window() if hasattr(host, "window") else None),
    )
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False

    from core.app_settings import (
        KEY_DISCOVERY_PRIVACY_TIER,
        KEY_DISCOVERY_SEARXNG_BASE_URL,
        KEY_KNOWLEDGE_PROVIDER_CREDENTIALS,
        set_discovery_privacy_tier,
        set_discovery_searxng_base_url,
    )
    from core.knowledge.credentials import clear_provider_api_key, set_provider_api_key
    from core.knowledge.discovery.privacy_policy import TIER_SEARXNG
    from ui.views.settings.sections.knowledge_web_discovery import (
        sync_web_discovery_policy_section,
    )

    set_discovery_searxng_base_url(dlg.saved_url)
    if dlg.saved_api_key:
        set_provider_api_key(SEARXNG_DISCOVERY_PROVIDER_ID, dlg.saved_api_key)
    else:
        clear_provider_api_key(SEARXNG_DISCOVERY_PROVIDER_ID)

    changed_keys = {KEY_DISCOVERY_SEARXNG_BASE_URL, KEY_KNOWLEDGE_PROVIDER_CREDENTIALS}
    if dlg.saved_switch_tier:
        set_discovery_privacy_tier(TIER_SEARXNG)
        changed_keys.add(KEY_DISCOVERY_PRIVACY_TIER)

    sync_web_discovery_policy_section(host)
    if hasattr(host, "_emit_external_settings_changed"):
        host._emit_external_settings_changed(*changed_keys)
    return True
