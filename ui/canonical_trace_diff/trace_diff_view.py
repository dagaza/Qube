"""Canonical trace diff debugger UI for Qube."""
from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any, Literal, Optional

from PyQt6.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QTextBrowser,
    QFileDialog,
)

from core.canonical_fingerprint import fingerprint_text
from core.canonical_trace_diff import CanonicalTrace, coerce_canonical_trace, find_first_divergence
from core.golden_trace_capture import load_golden_trace
from ui.canonical_trace_diff.collapse_timeline import CollapseTimelineWidget
from ui.canonical_trace_diff.diff_compute import (
    diff_json_trees,
    json_pretty,
    sentence_diff_html,
    word_diff_html,
)

logger = logging.getLogger("Qube.UI.CanonicalTraceDiff")

ViewMode = Literal["diff", "normalized", "raw"]

_DIFF_STYLESHEET = """
.diff-match { color: #86efac; }
.diff-mod { color: #fde047; background: rgba(253,224,71,0.12); }
.diff-miss { color: #fca5a5; background: rgba(248,113,113,0.15); }
.diff-extra { color: #93c5fd; background: rgba(147,197,253,0.12); }
.diff-truncated { color: #94a3b8; font-style: italic; }
.divergence-marker { color: #f97316; font-weight: 600; }
"""

_STATUS_COLORS = {
    "match": "#166534",
    "modified": "#854d0e",
    "missing": "#991b1b",
    "extra": "#1e40af",
}


class _DiffSignals(QObject):
    finished = pyqtSignal(str, str, str)  # section_key, left_html, right_html


class _PromptDiffTask(QRunnable):
    def __init__(self, section_key: str, left: str, right: str, signals: _DiffSignals) -> None:
        super().__init__()
        self.section_key = section_key
        self.left = left
        self.right = right
        self.signals = signals

    def run(self) -> None:
        try:
            if self.section_key == "prompt":
                lh, rh, _ = word_diff_html(self.left, self.right)
            else:
                lh, rh, _ = sentence_diff_html(self.left, self.right)
            self.signals.finished.emit(self.section_key, lh, rh)
        except Exception:
            logger.debug("[CanonicalTraceDiff] async diff failed", exc_info=True)


class _DivergenceRail(QWidget):
    """Vertical indicator for first pipeline divergence level."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._level: Optional[str] = None
        self.setFixedWidth(28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_level(self, level: Optional[str]) -> None:
        self._level = level
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        from PyQt6.QtGui import QPainter, QColor

        super().paintEvent(event)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0f172a"))
        if not self._level:
            return
        colors = {
            "REQUEST": QColor("#ef4444"),
            "PROMPT": QColor("#f59e0b"),
            "OUTPUT": QColor("#3b82f6"),
        }
        color = colors.get(self._level, QColor("#64748b"))
        slot_h = self.height() // 3
        idx = {"REQUEST": 0, "PROMPT": 1, "OUTPUT": 2}.get(self._level, 0)
        y = idx * slot_h + 4
        p.fillRect(4, y, 20, max(24, slot_h - 8), color)
        p.end()


class _TraceSidePanel(QWidget):
    """Baseline or current trace with collapsible sections."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._sections: dict[str, QGroupBox] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        head = QLabel(title)
        head.setObjectName("ViewSubtitle")
        root.addWidget(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setSpacing(8)
        for key, label in (
            ("request", "Request"),
            ("prompt", "Prompt"),
            ("output", "Output"),
            ("metadata", "Metadata"),
        ):
            box = QGroupBox(label)
            box.setCheckable(True)
            box.setChecked(True)
            lay = QVBoxLayout(box)
            if key in ("request", "metadata"):
                tree = QTreeWidget()
                tree.setHeaderLabels(["Path", "Value"])
                tree.setAlternatingRowColors(True)
                tree.setMinimumHeight(160)
                lay.addWidget(tree)
                self._sections[key] = box
                setattr(self, f"_{key}_tree", tree)
            elif key == "prompt":
                meta = QLabel("")
                meta.setWordWrap(True)
                meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                browser = QTextBrowser()
                browser.setOpenExternalLinks(False)
                browser.setMinimumHeight(200)
                lay.addWidget(meta)
                lay.addWidget(browser)
                self._sections[key] = box
                self._prompt_meta = meta
                self._prompt_browser = browser
            else:
                browser = QTextBrowser()
                browser.setOpenExternalLinks(False)
                browser.setMinimumHeight(160)
                lay.addWidget(browser)
                self._sections[key] = box
                self._output_browser = browser
            self._inner_layout.addWidget(box)
        self._inner_layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        self._raw_edit = QPlainTextEdit()
        self._raw_edit.setReadOnly(True)
        self._raw_edit.setVisible(False)
        mono = QFont("monospace")
        if not mono.exactMatch():
            mono = QFont("Consolas", 10)
        self._raw_edit.setFont(mono)
        root.addWidget(self._raw_edit)

    def section_boxes(self) -> list[QGroupBox]:
        return list(self._sections.values())

    def set_mode(self, mode: ViewMode) -> None:
        show_raw = mode == "raw"
        self._raw_edit.setVisible(show_raw)
        for box in self._sections.values():
            box.setVisible(not show_raw)

    def set_raw_json(self, trace: CanonicalTrace) -> None:
        self._raw_edit.setPlainText(json_pretty(trace.to_dict()))

    def populate_request_tree(
        self,
        rows: list[dict[str, Any]],
        *,
        side: Literal["baseline", "current"],
    ) -> None:
        tree: QTreeWidget = self._request_tree
        tree.clear()
        for row in rows:
            path = str(row.get("path") or "")
            status = str(row.get("status") or "match")
            value = row.get("baseline") if side == "baseline" else row.get("current")
            if side == "baseline" and status == "extra":
                continue
            if side == "current" and status == "missing":
                continue
            display = json.dumps(value, ensure_ascii=False, default=str) if value is not None else "—"
            item = QTreeWidgetItem([path, display])
            color = _STATUS_COLORS.get(status if status != "extra" or side == "current" else "extra", "#334155")
            if status == "match":
                color = _STATUS_COLORS["match"]
            item.setBackground(0, _qt_color(color, 0.25))
            item.setBackground(1, _qt_color(color, 0.18))
            tree.addTopLevelItem(item)
        tree.expandAll()

    def populate_metadata_tree(
        self,
        metadata: dict[str, Any],
        rows: list[dict[str, Any]],
        *,
        side: Literal["baseline", "current"],
    ) -> None:
        tree: QTreeWidget = self._metadata_tree
        tree.clear()
        for row in rows:
            path = str(row.get("path") or "")
            status = str(row.get("status") or "match")
            if side == "baseline" and status == "extra":
                continue
            if side == "current" and status == "missing":
                continue
            value = row.get("baseline") if side == "baseline" else row.get("current")
            display = json.dumps(value, ensure_ascii=False, default=str) if value is not None else "—"
            item = QTreeWidgetItem([path, display])
            color = _STATUS_COLORS.get(status, "#334155")
            if status == "match":
                color = _STATUS_COLORS["match"]
            item.setBackground(0, _qt_color(color, 0.25))
            item.setBackground(1, _qt_color(color, 0.18))
            tree.addTopLevelItem(item)
        tree.expandAll()

    def set_prompt(self, text: str, fp: dict[str, Any], html: str = "") -> None:
        self._prompt_meta.setText(
            f"length={fp.get('length', len(text))}  sha256={fp.get('sha256', '')[:16]}…  "
            f"short={fp.get('short', '')}"
        )
        if html:
            self._prompt_browser.setHtml(_DIFF_STYLESHEET + html)
        else:
            self._prompt_browser.setPlainText(text)

    def set_output(self, text: str, html: str = "") -> None:
        if html:
            self._output_browser.setHtml(_DIFF_STYLESHEET + html)
        else:
            self._output_browser.setPlainText(text)


def _qt_color(hex_color: str, alpha: float):
    from PyQt6.QtGui import QColor

    c = QColor(hex_color)
    c.setAlphaF(alpha)
    return c


class CanonicalTraceDiffView(QWidget):
    """Split-panel debugger comparing baseline vs current CanonicalTrace objects."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._baseline: Optional[CanonicalTrace] = None
        self._current: Optional[CanonicalTrace] = None
        self._report: dict[str, Any] = {}
        self._mode: ViewMode = "diff"
        self._run_pair: dict[str, Any] | None = None
        self._run_pair_turn: int = 0
        self._pool = QThreadPool.globalInstance()
        self._diff_signals = _DiffSignals()
        self._diff_signals.finished.connect(self._on_async_diff_ready)
        self._pending_diff: dict[str, tuple[str, str]] = {}
        self._scenario_runner: Any | None = None
        self._session_comparer: Any | None = None
        self._workflow_starter: Any | None = None
        self.setObjectName("CanonicalTraceDiffWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        surface = QFrame()
        surface.setObjectName("CanonicalTraceDiffSurface")
        surface_l = QVBoxLayout(surface)
        surface_l.setSpacing(10)
        root.addWidget(surface)

        title = QLabel("Canonical Trace Diff")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        surface_l.addWidget(title)

        self._summary = QFrame()
        self._summary.setObjectName("CanonicalTraceDiffSummary")
        sum_l = QHBoxLayout(self._summary)
        self._lbl_request = QLabel("request: —")
        self._lbl_prompt = QLabel("prompt: —")
        self._lbl_output = QLabel("output: —")
        self._lbl_first = QLabel("first divergence: —")
        self._lbl_summary = QLabel("")
        self._lbl_summary.setWordWrap(True)
        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setObjectName("ViewSubtitle")
        for w in (self._lbl_request, self._lbl_prompt, self._lbl_output, self._lbl_first):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            sum_l.addWidget(w)
        sum_l.addStretch()
        surface_l.addWidget(self._summary)
        surface_l.addWidget(self._lbl_summary)
        surface_l.addWidget(self._lbl_status)

        self._collapse_base = CollapseTimelineWidget()
        self._collapse_base.turn_selected.connect(self._select_run_pair_turn)
        self._collapse_base.hide()
        surface_l.addWidget(self._collapse_base)

        self._collapse_current = CollapseTimelineWidget()
        self._collapse_current.turn_selected.connect(self._select_run_pair_turn)
        self._collapse_current.hide()
        surface_l.addWidget(self._collapse_current)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("View:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Diff View (default)", "diff")
        self._mode_combo.addItem("Normalized Canonical View", "normalized")
        self._mode_combo.addItem("Raw JSON", "raw")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_combo)

        self._btn_expand = QPushButton("Expand all")
        self._btn_expand.clicked.connect(lambda: self._set_all_sections(True))
        toolbar.addWidget(self._btn_expand)
        self._btn_collapse = QPushButton("Collapse all")
        self._btn_collapse.clicked.connect(lambda: self._set_all_sections(False))
        toolbar.addWidget(self._btn_collapse)

        self._btn_load_base = QPushButton("Load baseline…")
        self._btn_load_base.clicked.connect(self._load_baseline_file)
        toolbar.addWidget(self._btn_load_base)
        self._btn_load_cur = QPushButton("Load current…")
        self._btn_load_cur.clicked.connect(self._load_current_file)
        toolbar.addWidget(self._btn_load_cur)

        self._btn_load_pair = QPushButton("Load diff…")
        self._btn_load_pair.clicked.connect(self._load_scenario_run_file)
        toolbar.addWidget(self._btn_load_pair)

        self._btn_run_scenario = QPushButton("Run comparison workflow…")
        self._btn_run_scenario.clicked.connect(self._run_comparison_workflow)
        toolbar.addWidget(self._btn_run_scenario)

        self._btn_run_single = QPushButton("Run single backend…")
        self._btn_run_single.clicked.connect(self._run_scenario_serial)
        toolbar.addWidget(self._btn_run_single)

        self._btn_compare_sessions = QPushButton("Compare sessions…")
        self._btn_compare_sessions.clicked.connect(self._compare_sessions_offline)
        toolbar.addWidget(self._btn_compare_sessions)

        self._backend_combo = QComboBox()
        self._backend_combo.addItem("Qube pipeline", "qube")
        self._backend_combo.addItem("External (LM Studio)", "external")
        toolbar.addWidget(self._backend_combo)

        toolbar.addWidget(QLabel("Turn:"))
        self._turn_combo = QComboBox()
        self._turn_combo.setMinimumWidth(220)
        self._turn_combo.currentIndexChanged.connect(self._on_turn_changed)
        toolbar.addWidget(self._turn_combo)

        self._btn_first_div = QPushButton("First divergence")
        self._btn_first_div.clicked.connect(self._jump_to_first_divergence)
        toolbar.addWidget(self._btn_first_div)

        self._btn_copy_base = QPushButton("Copy baseline JSON")
        self._btn_copy_base.clicked.connect(self._copy_baseline_json)
        toolbar.addWidget(self._btn_copy_base)
        self._btn_copy_cur = QPushButton("Copy current JSON")
        self._btn_copy_cur.clicked.connect(self._copy_current_json)
        toolbar.addWidget(self._btn_copy_cur)
        self._btn_copy_report = QPushButton("Copy diff report")
        self._btn_copy_report.clicked.connect(self._copy_diff_report)
        toolbar.addWidget(self._btn_copy_report)
        toolbar.addStretch()
        surface_l.addLayout(toolbar)

        body = QHBoxLayout()
        self._rail = _DivergenceRail()
        body.addWidget(self._rail)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._left = _TraceSidePanel("Baseline Trace (Golden)")
        self._right = _TraceSidePanel("Current Trace")
        splitter.addWidget(self._left)
        splitter.addWidget(self._right)
        splitter.setSizes([600, 600])
        body.addWidget(splitter, 1)
        surface_l.addLayout(body, 1)

        legend = QLabel(
            "Legend: green=match · yellow=modified · red=missing · blue=extra"
        )
        legend.setObjectName("ViewSubtitle")
        surface_l.addWidget(legend)

    def set_scenario_hooks(
        self,
        *,
        scenario_runner: Any | None = None,
        session_comparer: Any | None = None,
        workflow_starter: Any | None = None,
    ) -> None:
        """Wire app-provided serial replay, workflow, and offline compare callbacks."""
        self._scenario_runner = scenario_runner
        self._session_comparer = session_comparer
        self._workflow_starter = workflow_starter

    def set_status_message(self, text: str) -> None:
        self._lbl_status.setText(str(text or ""))

    def load_trace_pair(
        self,
        *,
        baseline: CanonicalTrace | dict[str, Any],
        current: CanonicalTrace | dict[str, Any],
        preserve_run_pair: bool = False,
    ) -> None:
        """Load baseline (A) and current (B) traces and render comparison."""
        if not preserve_run_pair:
            self._run_pair = None
            self._turn_combo.blockSignals(True)
            self._turn_combo.clear()
            self._turn_combo.blockSignals(False)
            self._collapse_base.hide()
            self._collapse_current.hide()
        self._baseline = coerce_canonical_trace(baseline)
        self._current = coerce_canonical_trace(current)
        self._report = find_first_divergence(self._baseline, self._current)
        self._render()

    def load_scenario_run_pair(
        self,
        pair: Any,
        *,
        turn_index: int | None = None,
    ) -> None:
        """Load a ScenarioRunPair and show baseline vs compare backend at ``turn_index``."""
        from core.collapse_diagnostics import build_collapse_timeline
        from core.scenario_loader import first_diverging_turn_index

        self._run_pair = pair
        backends = list(getattr(pair, "backends", []) or [])
        if len(backends) < 2:
            raise ValueError("ScenarioRunPair requires at least two backends for diff view")

        runs = getattr(pair, "runs", None) or {}
        base_runs = runs.get(backends[0]) or []
        cur_runs = runs.get(backends[1]) or []
        self._collapse_base.set_timeline(
            build_collapse_timeline(base_runs, backend_label=str(backends[0])),
            backend_label=str(backends[0]),
        )
        self._collapse_current.set_timeline(
            build_collapse_timeline(cur_runs, backend_label=str(backends[1])),
            backend_label=str(backends[1]),
        )

        self._turn_combo.blockSignals(True)
        self._turn_combo.clear()
        diffs_by_turn = {
            int(d.turn_index): d for d in (getattr(pair, "diffs", None) or [])
        }
        baseline_runs = (getattr(pair, "runs", None) or {}).get(backends[0]) or []
        for trace in baseline_runs:
            idx = int(trace.turn_index)
            diff = diffs_by_turn.get(idx)
            div = getattr(diff, "first_divergence", None) if diff else None
            label = f"Turn {idx}: {trace.user_message[:48]!r}"
            risk = str((trace.trace.metadata or {}).get("collapse_risk") or "")
            if risk:
                label += f" [{risk}]"
            if div:
                label += f" [{div}]"
            self._turn_combo.addItem(label, idx)
        self._turn_combo.blockSignals(False)

        if turn_index is None:
            turn_index = first_diverging_turn_index(pair)
        if turn_index is None and baseline_runs:
            turn_index = int(baseline_runs[0].turn_index)

        if turn_index is not None:
            for i in range(self._turn_combo.count()):
                if self._turn_combo.itemData(i) == turn_index:
                    self._turn_combo.setCurrentIndex(i)
                    break
            self._show_run_pair_turn(int(turn_index))
        else:
            self._show_run_pair_turn(0)

    def _show_run_pair_turn(self, turn_index: int) -> None:
        if self._run_pair is None:
            return
        backends = list(getattr(self._run_pair, "backends", []) or [])
        if len(backends) < 2:
            return
        runs = getattr(self._run_pair, "runs", None) or {}
        base_traces = runs.get(backends[0]) or []
        cur_traces = runs.get(backends[1]) or []
        baseline_trace = next((t for t in base_traces if t.turn_index == turn_index), None)
        current_trace = next((t for t in cur_traces if t.turn_index == turn_index), None)
        if baseline_trace is None or current_trace is None:
            return
        self._run_pair_turn = turn_index
        self._collapse_base.set_selected_turn(turn_index)
        self._collapse_current.set_selected_turn(turn_index)
        self._left._sections["request"].setTitle(f"Request ({backends[0]})")
        self._right._sections["request"].setTitle(f"Request ({backends[1]})")
        self.load_trace_pair(
            baseline=baseline_trace.trace,
            current=current_trace.trace,
            preserve_run_pair=True,
        )

    def _select_run_pair_turn(self, turn_index: int) -> None:
        for i in range(self._turn_combo.count()):
            if self._turn_combo.itemData(i) == turn_index:
                self._turn_combo.setCurrentIndex(i)
                return

    def _on_turn_changed(self) -> None:
        if self._run_pair is None:
            return
        turn_index = self._turn_combo.currentData()
        if turn_index is None:
            return
        self._show_run_pair_turn(int(turn_index))

    def _jump_to_first_divergence(self) -> None:
        if self._run_pair is None:
            return
        from core.scenario_loader import first_diverging_turn_index

        idx = first_diverging_turn_index(self._run_pair)
        if idx is None:
            return
        for i in range(self._turn_combo.count()):
            if self._turn_combo.itemData(i) == idx:
                self._turn_combo.setCurrentIndex(i)
                break

    def _render(self) -> None:
        if self._baseline is None or self._current is None:
            return
        self._update_summary()
        self._rail.set_level(self._report.get("first_divergence_level"))
        mode = self._mode
        self._left.set_mode(mode)
        self._right.set_mode(mode)
        if mode == "raw":
            self._left.set_raw_json(self._baseline)
            self._right.set_raw_json(self._current)
            return

        req_rows = diff_json_trees(
            self._baseline.request.to_dict(),
            self._current.request.to_dict(),
        )
        self._left.populate_request_tree(req_rows, side="baseline")
        self._right.populate_request_tree(req_rows, side="current")

        meta_rows = diff_json_trees(
            self._baseline.metadata or {},
            self._current.metadata or {},
        )
        self._left.populate_metadata_tree(
            self._baseline.metadata,
            meta_rows,
            side="baseline",
        )
        self._right.populate_metadata_tree(
            self._current.metadata,
            meta_rows,
            side="current",
        )

        b_fp = (self._baseline.fingerprints or {}).get("prompt") or fingerprint_text(
            self._baseline.prompt
        )
        c_fp = (self._current.fingerprints or {}).get("prompt") or fingerprint_text(
            self._current.prompt
        )
        fp_match = b_fp.get("sha256") == c_fp.get("sha256")
        fp_note = "fingerprint match" if fp_match else "fingerprint MISMATCH"

        if mode == "normalized":
            self._left.set_prompt(self._baseline.prompt, b_fp)
            self._right.set_prompt(self._current.prompt, c_fp)
            self._left.set_output(self._baseline.output)
            self._right.set_output(self._current.output)
            return

        self._left.set_prompt(
            self._baseline.prompt,
            b_fp,
            html=f"<pre>{escape(self._baseline.prompt[:50000])}</pre>",
        )
        self._right.set_prompt(
            self._current.prompt,
            c_fp,
            html=f"<pre>{escape(self._current.prompt[:50000])}</pre>",
        )
        self._left._prompt_meta.setText(self._left._prompt_meta.text() + f"  · {fp_note}")
        self._schedule_async_diff("prompt", self._baseline.prompt, self._current.prompt)
        self._schedule_async_diff("output", self._baseline.output, self._current.output)

    def _schedule_async_diff(self, key: str, left: str, right: str) -> None:
        task = _PromptDiffTask(key, left, right, self._diff_signals)
        self._pool.start(task)

    def _on_async_diff_ready(self, section_key: str, left_html: str, right_html: str) -> None:
        if section_key == "prompt":
            b_fp = fingerprint_text(self._baseline.prompt if self._baseline else "")
            c_fp = fingerprint_text(self._current.prompt if self._current else "")
            self._left.set_prompt(self._baseline.prompt if self._baseline else "", b_fp, left_html)
            self._right.set_prompt(self._current.prompt if self._current else "", c_fp, right_html)
        elif section_key == "output":
            self._left.set_output(self._baseline.output if self._baseline else "", left_html)
            self._right.set_output(self._current.output if self._current else "", right_html)

    def _update_summary(self) -> None:
        r = self._report
        self._lbl_request.setText(f"request_match: {r.get('request_match')}")
        self._lbl_prompt.setText(f"prompt_match: {r.get('prompt_match')}")
        self._lbl_output.setText(f"output_match: {r.get('output_match')}")
        lvl = r.get("first_divergence_level") or "none"
        self._lbl_first.setText(f"first_divergence_level: {lvl}")

        collapse_bits: list[str] = []
        if self._baseline and (self._baseline.metadata or {}).get("collapse_risk"):
            b_meta = self._baseline.metadata or {}
            collapse_bits.append(
                "baseline collapse_risk="
                f"{b_meta.get('collapse_risk')} "
                f"(score={b_meta.get('collapse_score', '—')})"
            )
        if self._current and (self._current.metadata or {}).get("collapse_risk"):
            c_meta = self._current.metadata or {}
            collapse_bits.append(
                "current collapse_risk="
                f"{c_meta.get('collapse_risk')} "
                f"(score={c_meta.get('collapse_score', '—')})"
            )
        if collapse_bits:
            self._lbl_summary.setText(
                str(r.get("diff_summary") or "")
                + (" · " if r.get("diff_summary") else "")
                + " · ".join(collapse_bits)
            )
        else:
            self._lbl_summary.setText(str(r.get("diff_summary") or ""))

    def _on_mode_changed(self) -> None:
        self._mode = self._mode_combo.currentData() or "diff"
        self._render()

    def _set_all_sections(self, expanded: bool) -> None:
        for panel in (self._left, self._right):
            for box in panel.section_boxes():
                box.setChecked(expanded)

    def _load_baseline_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load baseline golden trace",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            trace = load_golden_trace(path)
            if self._current is not None:
                self.load_trace_pair(baseline=trace, current=self._current)
            else:
                self._baseline = trace
        except Exception as e:
            logger.warning("Failed to load baseline trace: %s", e)

    def _load_current_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load current trace",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            trace = load_golden_trace(path)
            if self._baseline is not None:
                self.load_trace_pair(baseline=self._baseline, current=trace)
            else:
                self._current = trace
        except Exception as e:
            logger.warning("Failed to load current trace: %s", e)

    def _load_scenario_run_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load scenario diff or session",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            from core.scenario_loader import load_backend_session, load_scenario_run_pair

            data = json.loads(Path(path).read_text(encoding="utf-8"))
            schema = str((data or {}).get("schema") or "")
            if schema == "qube.scenario_session.v1":
                session = load_backend_session(path)
                self.set_status_message(
                    f"Loaded session backend={session.backend} ({len(session.traces)} turns). "
                    "Use Compare sessions to diff against another backend."
                )
                return
            pair = load_scenario_run_pair(path)
            self.load_scenario_run_pair(pair)
            self.setWindowTitle(
                f"Qube — Scenario Diff: {getattr(pair, 'scenario_name', '') or path}"
            )
        except Exception as e:
            logger.warning("Failed to load scenario artifact: %s", e)

    def _run_comparison_workflow(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select scenario JSON",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        if self._workflow_starter is None:
            self.set_status_message(
                "Workflow not wired. Launch with --run-scenario or use the main window."
            )
            return
        try:
            self._workflow_starter(path, single_phase=False)
            self.set_status_message("Scenario comparison workflow opened.")
        except Exception as e:
            logger.warning("Scenario workflow failed: %s", e)
            self.set_status_message(f"Workflow failed: {e}")

    def _run_scenario_serial(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select scenario JSON",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        backend = str(self._backend_combo.currentData() or "qube")
        if self._scenario_runner is None:
            self.set_status_message(
                "Scenario runner not wired. Use CLI or launch with --run-scenario."
            )
            return
        try:
            out_path = self._scenario_runner(path, backend)
            self.set_status_message(
                f"Session saved ({backend}): {out_path}. "
                "Run the other backend, then Compare sessions."
            )
        except Exception as e:
            logger.warning("Scenario replay failed: %s", e)
            self.set_status_message(f"Replay failed: {e}")

    def _compare_sessions_offline(self) -> None:
        path_a, _ = QFileDialog.getOpenFileName(
            self,
            "Baseline session (e.g. qube)",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path_a:
            return
        path_b, _ = QFileDialog.getOpenFileName(
            self,
            "Compare session (e.g. external)",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path_b:
            return
        try:
            if self._session_comparer is not None:
                pair = self._session_comparer(path_a, path_b)
            else:
                from core.scenario_loader import compare_sessions

                pair = compare_sessions(path_a, path_b, save=True)
            self.load_scenario_run_pair(pair)
            self.setWindowTitle(f"Qube — Scenario Diff: {pair.scenario_name}")
            self.set_status_message("Offline comparison loaded.")
        except Exception as e:
            logger.warning("Session compare failed: %s", e)
            self.set_status_message(f"Compare failed: {e}")

    def _copy_baseline_json(self) -> None:
        if self._baseline:
            QApplication.clipboard().setText(json_pretty(self._baseline.to_dict()))

    def _copy_current_json(self) -> None:
        if self._current:
            QApplication.clipboard().setText(json_pretty(self._current.to_dict()))

    def _copy_diff_report(self) -> None:
        QApplication.clipboard().setText(json_pretty(self._report))


def load_trace_pair(
    *,
    baseline: CanonicalTrace | dict[str, Any],
    current: CanonicalTrace | dict[str, Any],
    view: Optional[CanonicalTraceDiffView] = None,
    parent: Optional[QWidget] = None,
    show: bool = True,
) -> CanonicalTraceDiffView:
    """
    Public entry: load traces into the debugger and optionally show the window.

    Python naming for the requested loadTracePair({ baseline, current }) API.
    """
    if view is None:
        view = CanonicalTraceDiffView(parent=parent)
        view.setWindowFlag(Qt.WindowType.Window, True)
        view.setWindowTitle("Qube — Canonical Trace Diff")
        view.resize(1280, 860)
    view.load_trace_pair(baseline=baseline, current=current)
    if show:
        view.show()
        view.raise_()
    return view


def load_scenario_run_pair_view(
    pair: Any,
    *,
    view: Optional[CanonicalTraceDiffView] = None,
    parent: Optional[QWidget] = None,
    turn_index: int | None = None,
    show: bool = True,
) -> CanonicalTraceDiffView:
    """Load a ScenarioRunPair into the diff debugger and jump to first divergence."""
    if view is None:
        view = CanonicalTraceDiffView(parent=parent)
        view.setWindowFlag(Qt.WindowType.Window, True)
        view.setWindowTitle("Qube — Canonical Trace Diff")
        view.resize(1280, 860)
    view.load_scenario_run_pair(pair, turn_index=turn_index)
    if show:
        view.show()
        view.raise_()
    return view


def open_canonical_trace_diff_window(parent: Optional[QWidget] = None) -> CanonicalTraceDiffView:
    """Open an empty debugger window (load traces via UI or load_trace_pair)."""
    view = CanonicalTraceDiffView(parent=parent)
    view.setWindowFlag(Qt.WindowType.Window, True)
    view.setWindowTitle("Qube — Canonical Trace Diff")
    view.resize(1280, 860)
    view.show()
    return view
