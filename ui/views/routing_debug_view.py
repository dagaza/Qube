"""Per-turn cognitive routing explainability panel."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("Qube.UI.RoutingDebug")


def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


class RoutingDebugView(QWidget):
    def __init__(self, workers: dict, gpu_monitor=None, native_engine=None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.workers = workers
        self._records_newest_first: list[dict] = []
        self._record_by_row: list[dict] = []
        self.setObjectName("RoutingDebugWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        surface = QFrame()
        surface.setObjectName("RoutingDebugSurface")
        surface.setFrameShape(QFrame.Shape.NoFrame)
        surface_l = QVBoxLayout(surface)
        surface_l.setContentsMargins(16, 16, 16, 16)
        surface_l.setSpacing(10)
        root.addWidget(surface)

        title = QLabel("Routing Debug")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        surface_l.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("RoutingDebugSplitter")
        left = QWidget()
        left.setObjectName("RoutingDebugLeftPane")
        left.setMinimumWidth(280)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(16, 8, 16, 8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Route:"))
        self._route_filter = QComboBox()
        for label in ("All", "WEB", "RAG", "MEMORY", "HYBRID", "NONE"):
            self._route_filter.addItem(label)
        self._route_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._route_filter, 1)
        left_l.addLayout(filter_row)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_row_changed)
        left_l.addWidget(self._list, 1)

        right_scroll = QScrollArea()
        right_scroll.setObjectName("RoutingDebugRightScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.viewport().setObjectName("RoutingDebugViewport")

        detail = QWidget()
        detail.setObjectName("RoutingDebugDetailPane")
        self._detail_layout = QVBoxLayout(detail)
        self._detail_layout.setContentsMargins(16, 8, 16, 24)
        self._detail_layout.setSpacing(12)

        self._hdr_route = QLabel("")
        self._hdr_route.setProperty("class", "PageTitle")
        self._hdr_summary = QLabel("")
        self._hdr_summary.setWordWrap(True)
        self._hdr_query = QLabel("")
        self._hdr_query.setWordWrap(True)
        self._detail_layout.addWidget(self._hdr_route)
        self._detail_layout.addWidget(self._hdr_summary)
        self._detail_layout.addWidget(self._hdr_query)

        self._gb_summary = QGroupBox("Summary")
        self._gb_summary.setCheckable(False)
        s_l = QVBoxLayout(self._gb_summary)
        self._lbl_winning_reason = QLabel("")
        self._lbl_winning_reason.setWordWrap(True)
        self._lbl_trace_level = QLabel("")
        s_l.addWidget(self._lbl_winning_reason)
        s_l.addWidget(self._lbl_trace_level)
        self._detail_layout.addWidget(self._gb_summary)

        self._gb_signals = QGroupBox("Key signals")
        sig_l = QVBoxLayout(self._gb_signals)
        self._lbl_winning_signal = QLabel("")
        self._lbl_winning_signal.setWordWrap(True)
        sig_l.addWidget(self._lbl_winning_signal)
        self._losing_list = QListWidget()
        self._losing_list.setMaximumHeight(160)
        sig_l.addWidget(self._losing_list)
        self._detail_layout.addWidget(self._gb_signals)

        self._gb_mod = QGroupBox("System modifiers (Tier 3 / 5 / 6)")
        mod_l = QVBoxLayout(self._gb_mod)
        self._lbl_tier3 = QLabel("")
        self._lbl_tier3.setWordWrap(True)
        self._lbl_tier5 = QLabel("")
        self._lbl_tier5.setWordWrap(True)
        self._lbl_tier6 = QLabel("")
        self._lbl_tier6.setWordWrap(True)
        mod_l.addWidget(self._lbl_tier3)
        mod_l.addWidget(self._lbl_tier5)
        mod_l.addWidget(self._lbl_tier6)
        self._detail_layout.addWidget(self._gb_mod)

        self._gb_trace = QGroupBox("Full trace JSON")
        self._gb_trace.setCheckable(True)
        self._gb_trace.setChecked(False)
        t_l = QVBoxLayout(self._gb_trace)
        self._trace_text = QPlainTextEdit()
        self._trace_text.setReadOnly(True)
        self._trace_text.setMaximumHeight(280)
        monof = self._trace_text.font()
        monof.setFamily("monospace")
        self._trace_text.setFont(monof)
        t_l.addWidget(self._trace_text)
        self._detail_layout.addWidget(self._gb_trace)

        self._raw_toggle = QCheckBox("Show raw decision data")
        self._raw_toggle.setChecked(False)
        self._raw_toggle.toggled.connect(self._on_raw_toggled)
        self._detail_layout.addWidget(self._raw_toggle)
        self._decision_text = QPlainTextEdit()
        self._decision_text.setReadOnly(True)
        self._decision_text.setMaximumHeight(320)
        self._decision_text.setFont(monof)
        self._decision_text.setVisible(False)
        self._detail_layout.addWidget(self._decision_text)

        self._detail_layout.addStretch(1)
        right_scroll.setWidget(detail)

        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        surface_l.addWidget(splitter, 1)

        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        self._load_snapshot_from_worker()

    def _load_snapshot_from_worker(self) -> None:
        w = self.workers.get("llm")
        if w is None or not hasattr(w, "routing_debug_buffer"):
            return
        try:
            snap = w.routing_debug_buffer.snapshot()
        except Exception as e:
            logger.debug("routing snapshot failed: %s", e)
            return
        self._records_newest_first = []
        for rec in reversed(snap):
            try:
                self._records_newest_first.append(
                    {
                        "timestamp": rec.timestamp,
                        "session_id": rec.session_id,
                        "turn_id": rec.turn_id,
                        "query": rec.query,
                        "route": rec.route,
                        "route_pre_policy": rec.route_pre_policy,
                        "strategy": rec.strategy,
                        "trace_level": rec.trace_level,
                        "top_intent": rec.top_intent,
                        "top_score": rec.top_score,
                        "summary": rec.summary,
                        "trace": rec.trace,
                        "decision": rec.decision,
                    }
                )
            except Exception:
                continue
        self._rebuild_list()

    def add_record(self, payload: dict) -> None:
        if not payload:
            return
        self._records_newest_first.insert(0, payload)
        cap = 100
        if len(self._records_newest_first) > cap:
            self._records_newest_first = self._records_newest_first[:cap]
        self._rebuild_list()

    def _route_filter_pred(self, rec: dict) -> bool:
        filt = self._route_filter.currentText()
        if filt == "All":
            return True
        return str(rec.get("route", "")).upper() == filt

    def _row_label(self, rec: dict) -> str:
        r = str(rec.get("route", "")).upper()
        ts = rec.get("top_score")
        q = (rec.get("query") or "").replace("\n", " ").strip()
        if len(q) > 56:
            q = q[:53] + "..."
        if ts is None:
            score_part = "—"
        else:
            try:
                score_part = f"{float(ts):.2f}"
            except (TypeError, ValueError):
                score_part = "—"
        return f'[{r} {score_part}] "{q}"'

    def _rebuild_list(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        self._record_by_row = []
        for rec in self._records_newest_first:
            if not self._route_filter_pred(rec):
                continue
            item = QListWidgetItem(self._row_label(rec))
            item.setToolTip(rec.get("summary") or "")
            self._list.addItem(item)
            self._record_by_row.append(rec)
        self._list.blockSignals(False)
        if self._list.count() > 0 and self._list.currentRow() < 0:
            self._list.setCurrentRow(0)
        else:
            self._clear_detail()

    def _on_filter_changed(self, _idx: int) -> None:
        self._rebuild_list()

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._record_by_row):
            self._clear_detail()
            return
        self._show_record(self._record_by_row[row])

    def _clear_detail(self) -> None:
        self._hdr_route.setText("")
        self._hdr_summary.setText("")
        self._hdr_query.setText("")
        self._lbl_winning_reason.setText("")
        self._lbl_trace_level.setText("")
        self._lbl_winning_signal.setText("")
        self._losing_list.clear()
        self._lbl_tier3.setText("")
        self._lbl_tier5.setText("")
        self._lbl_tier6.setText("")
        self._trace_text.clear()
        self._decision_text.clear()

    def _show_record(self, rec: dict) -> None:
        trace = rec.get("trace") or {}
        route = str(rec.get("route", "")).upper()
        pre = rec.get("route_pre_policy")
        eff_line = f"Effective route: {route}"
        if pre and str(pre).lower() != str(rec.get("route", "")).lower():
            eff_line += f"  (router pre-policy: {str(pre).upper()})"
        self._hdr_route.setText(eff_line)
        self._hdr_summary.setText(rec.get("summary") or "")
        self._hdr_query.setText(f'Query: {rec.get("query") or ""}')

        wr = trace.get("winning_reason", "—")
        self._lbl_winning_reason.setText(f"Winning reason: {wr}")
        self._lbl_trace_level.setText(
            f"Trace: {rec.get('trace_level', '—')} · strategy: {rec.get('strategy', '—')}"
        )

        ws = trace.get("winning_signal") or {}
        self._lbl_winning_signal.setText(
            "Winning lane: {lane} · score {score:.3f} · threshold {thr:.3f} · source {src}".format(
                lane=ws.get("lane", "—"),
                score=float(ws.get("score") or 0),
                thr=float(ws.get("threshold") or 0),
                src=ws.get("source", "—"),
            )
        )
        self._losing_list.clear()
        for c in trace.get("losing_candidates") or []:
            self._losing_list.addItem(
                "{lane} | score {sc:.3f} vs thr {th:.3f} | {reason}".format(
                    lane=c.get("lane"),
                    sc=float(c.get("score") or 0),
                    th=float(c.get("threshold") or 0),
                    reason=c.get("reason", ""),
                )
            )

        tier2 = bool((trace.get("confidence") or {}).get("tier2_active"))
        t3 = trace.get("tier3") or {}
        t3_on = bool(t3.get("band_active"))
        lb = t3.get("lane_bias") or {}
        self._lbl_tier3.setText(
            "Tier 3: band_active={} · lane_bias memory={:.3f} rag={:.3f} web={:.3f} · tier2_active={}".format(
                t3_on,
                float(lb.get("memory") or 0),
                float(lb.get("rag") or 0),
                float(lb.get("web") or 0),
                tier2,
            )
        )

        t56 = trace.get("tier5_6") or {}
        self._lbl_tier5.setText(
            "Tier 5: active={} · policy={} · reason={}".format(
                t56.get("tier5_active"),
                t56.get("policy"),
                t56.get("policy_reason"),
            )
        )
        self._lbl_tier6.setText(
            "Tier 6: active={} · conflicts={} · interpretation={}".format(
                t56.get("tier6_active"),
                t56.get("conflicts"),
                t56.get("interpretation"),
            )
        )

        self._trace_text.setPlainText(_json_pretty(trace))
        self._decision_text.setPlainText(_json_pretty(rec.get("decision") or {}))
        self._decision_text.setVisible(self._raw_toggle.isChecked())

    def _on_raw_toggled(self, on: bool) -> None:
        self._decision_text.setVisible(on)
        row = self._list.currentRow()
        if 0 <= row < len(self._record_by_row):
            self._show_record(self._record_by_row[row])
