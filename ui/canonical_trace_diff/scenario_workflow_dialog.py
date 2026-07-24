"""Non-blocking workflow: Qube pathway test, then external pathway test with user gates."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QProcess, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.conversation_replay import Scenario
from core.scenario_loader import load_scenario
from core.scenario_workflow import (
    ReadinessFn,
    QubeRunnerFn,
    SessionComparerFn,
    build_external_replay_command,
    resolve_external_api_url,
    suggested_external_model_name,
)
from core.theme.accessors import theme_for
from ui.canonical_trace_diff.trace_diff_theme import scenario_workflow_surface_stylesheet
from ui.components.prestige_dialog import _resolve_is_dark_from_parent

logger = logging.getLogger("Qube.ScenarioWorkflowDialog")


class ScenarioComparisonWorkflowDialog(QDialog):
    """Two-phase scenario comparison with explicit user confirmation gates."""

    qube_phase_completed = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None,
        *,
        scenario_path: str,
        repo_root: Path | str,
        qube_ready: ReadinessFn,
        run_qube: QubeRunnerFn,
        compare_sessions: SessionComparerFn | None = None,
        model_hint: Callable[[Scenario], str] | None = None,
        single_phase: bool = False,
    ) -> None:
        super().__init__(parent)
        self._scenario_path = str(scenario_path)
        self._repo_root = Path(repo_root)
        self._qube_ready = qube_ready
        self._run_qube = run_qube
        self._compare_sessions = compare_sessions
        self._model_hint = model_hint
        self._single_phase = bool(single_phase)
        self._scenario = load_scenario(self._scenario_path)
        self._phase = "qube_gate"
        self._qube_session_path = ""
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._refresh_readiness)

        self.setWindowTitle("Scenario comparison workflow")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumWidth(520)
        self._qube_completed = False
        self._is_dark = _resolve_is_dark_from_parent(parent)
        self._build_ui()
        self._apply_theme()
        self._refresh_readiness()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._surface = QFrame()
        self._surface.setObjectName("ScenarioWorkflowSurface")
        surface_l = QVBoxLayout(self._surface)
        surface_l.setSpacing(10)

        self._title = QLabel("Phase 1 — Qube pathway")
        self._title.setObjectName("ViewTitle")
        surface_l.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        self._body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        surface_l.addWidget(self._body)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setObjectName("ViewSubtitle")
        surface_l.addWidget(self._status)

        layout.addWidget(self._surface)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self._btn_cancel = QPushButton("Hide for now")
        self._btn_cancel.setToolTip(
            "Close this panel without cancelling the workflow. "
            "It reopens when a model is ready or from Run comparison workflow…"
        )
        self._btn_cancel.clicked.connect(self._hide_for_now)
        buttons.addWidget(self._btn_cancel)
        self._btn_primary = QPushButton("Start Qube pathway test")
        self._btn_primary.setDefault(True)
        self._btn_primary.clicked.connect(self._on_primary_clicked)
        buttons.addWidget(self._btn_primary)
        layout.addLayout(buttons)

        self._set_qube_gate_copy()

    def refresh_theme(self, is_dark: bool | None = None) -> None:
        if is_dark is not None:
            self._is_dark = is_dark
        else:
            self._is_dark = _resolve_is_dark_from_parent(self.parent())
        self._apply_theme()

    def _apply_theme(self) -> None:
        theme = theme_for(is_dark=self._is_dark)
        self._surface.setStyleSheet(scenario_workflow_surface_stylesheet(theme))

    def _model_name(self) -> str:
        if self._model_hint is not None:
            hinted = str(self._model_hint(self._scenario) or "").strip()
            if hinted:
                return hinted
        return suggested_external_model_name(self.parent(), self._scenario)

    def _set_qube_gate_copy(self) -> None:
        self._title.setText("Phase 1 — Qube pathway")
        self._body.setText(
            "Run the scenario through Qube's full pipeline (Harmony, stops, native engine).\n\n"
            "Use the main Qube window to load a model in the toolbar — this panel does not "
            "block the app. When a model is ready, click Start below."
        )
        self._btn_cancel.setText("Hide for now")
        self._btn_cancel.setEnabled(True)

    def _hide_for_now(self) -> None:
        if self._phase == "qube_gate":
            self.hide()
            return
        self.reject()

    def qube_phase_done(self) -> bool:
        return self._qube_completed

    def _set_external_gate_copy(self) -> None:
        model = self._model_name() or "(same model as Qube run)"
        api_url = resolve_external_api_url(self._scenario)
        self._title.setText("Phase 2 — External pathway")
        self._body.setText(
            "The Qube session has been saved.\n\n"
            f"Session: {self._qube_session_path or '—'}\n\n"
            "Before the external test:\n"
            "1. Close Qube completely (free VRAM for LM Studio).\n"
            "2. Start LM Studio and load the model.\n"
            f"3. Model name in LM Studio should match: {model}\n"
            f"4. API endpoint: {api_url}\n\n"
            "When LM Studio is ready, confirm below. A background runner will wait for "
            "the API, replay the scenario on the external HTTP path, and write a diff."
        )
        self._btn_primary.setText("Run external pathway test")
        self._btn_primary.setEnabled(True)
        self._status.setText(
            "You can close this dialog after starting the external runner; it continues in the background."
        )

    def _refresh_readiness(self) -> None:
        if self._phase != "qube_gate":
            return
        ready, message = self._qube_ready()
        self._btn_primary.setEnabled(bool(ready))
        if ready:
            self._status.setText(
                "Model ready — click to start the Qube pathway test. "
                "(This panel may have opened in the background while you loaded the model.)"
            )
            if not self.isVisible() and not self._qube_completed:
                self.show()
                self.raise_()
        else:
            self._status.setText(message or "Waiting for a loaded model…")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._phase == "qube_gate":
            if not self._poll_timer.isActive():
                self._poll_timer.start()
            self._refresh_readiness()

    def closeEvent(self, event) -> None:
        if self._phase == "qube_gate" and not self._qube_completed:
            event.ignore()
            self.hide()
            return
        self._poll_timer.stop()
        super().closeEvent(event)

    def _on_primary_clicked(self) -> None:
        if self._phase == "qube_gate":
            self._start_qube_phase()
        elif self._phase == "external_gate":
            self._start_external_phase()

    def _start_qube_phase(self) -> None:
        ready, message = self._qube_ready()
        if not ready:
            self._status.setText(message or "Model is not ready yet.")
            return
        self._poll_timer.stop()
        self._phase = "qube_running"
        self._btn_primary.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        self._status.setText("Running Qube pathway replay…")
        try:
            self._qube_session_path = str(self._run_qube(self._scenario_path) or "").strip()
        except Exception as exc:
            logger.exception("Qube pathway replay failed")
            self._phase = "qube_gate"
            self._btn_primary.setEnabled(False)
            self._btn_cancel.setEnabled(True)
            self._poll_timer.start()
            self._status.setText(f"Qube replay failed: {exc}")
            return

        self._qube_completed = True
        self.qube_phase_completed.emit()

        if self._single_phase:
            self._phase = "done"
            self._title.setText("Qube pathway complete")
            self._body.setText(f"Session saved:\n{self._qube_session_path or '—'}")
            self._btn_primary.setText("Close")
            self._btn_primary.setEnabled(True)
            self._btn_primary.clicked.disconnect()
            self._btn_primary.clicked.connect(self.accept)
            self._status.setText("")
            return

        self._phase = "external_gate"
        self._btn_cancel.setText("Close")
        self._btn_cancel.setToolTip("")
        self._btn_cancel.setEnabled(True)
        self._set_external_gate_copy()

    def _start_external_phase(self) -> None:
        model = self._model_name()
        api_url = resolve_external_api_url(self._scenario)
        cmd = build_external_replay_command(
            self._scenario_path,
            repo_root=self._repo_root,
            model=model,
            api_url=api_url,
            qube_session_path=self._qube_session_path,
        )
        started = QProcess.startDetached(cmd[0], cmd[1:], str(self._repo_root))
        if not started:
            self._status.setText("Failed to start external runner process.")
            return
        self._phase = "external_launched"
        self._btn_primary.setText("Close")
        self._btn_primary.setEnabled(True)
        self._btn_primary.clicked.disconnect()
        self._btn_primary.clicked.connect(self.accept)
        diff_hint = ""
        if self._qube_session_path:
            diff_hint = (
                "\n\nWhen the runner finishes, open the diff under debug/replay_diffs/ "
                "or relaunch Qube with --compare-sessions."
            )
        self._status.setText(
            "External runner started in the background.\n"
            "Close Qube now, start LM Studio, and load the model.\n"
            f"Command: {' '.join(cmd)}"
            f"{diff_hint}"
        )


def create_scenario_comparison_workflow(
    parent: QWidget | None,
    *,
    scenario_path: str,
    repo_root: Path | str,
    qube_ready: ReadinessFn,
    run_qube: QubeRunnerFn,
    compare_sessions: SessionComparerFn | None = None,
    model_hint: Callable[[Scenario], str] | None = None,
    single_phase: bool = False,
) -> ScenarioComparisonWorkflowDialog:
    return ScenarioComparisonWorkflowDialog(
        parent,
        scenario_path=scenario_path,
        repo_root=repo_root,
        qube_ready=qube_ready,
        run_qube=run_qube,
        compare_sessions=compare_sessions,
        model_hint=model_hint,
        single_phase=single_phase,
    )


def open_scenario_comparison_workflow(
    parent: QWidget | None,
    *,
    scenario_path: str,
    repo_root: Path | str,
    qube_ready: ReadinessFn,
    run_qube: QubeRunnerFn,
    compare_sessions: SessionComparerFn | None = None,
    model_hint: Callable[[Scenario], str] | None = None,
    single_phase: bool = False,
) -> ScenarioComparisonWorkflowDialog:
    """Show a non-modal workflow panel (does not block the main Qube window)."""
    dialog = create_scenario_comparison_workflow(
        parent,
        scenario_path=scenario_path,
        repo_root=repo_root,
        qube_ready=qube_ready,
        run_qube=run_qube,
        compare_sessions=compare_sessions,
        model_hint=model_hint,
        single_phase=single_phase,
    )
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
