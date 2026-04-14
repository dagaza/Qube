import logging
import os
import psutil
from collections import deque
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from core.app_settings import get_engine_mode

logger = logging.getLogger("Qube.UI.Telemetry")


class TelemetryView(QWidget):
    def __init__(self, workers: dict, gpu_monitor, native_engine=None):
        super().__init__()
        self.workers = workers
        self.gpu_monitor = gpu_monitor
        self._native_engine = native_engine

        # --- GRAPH DATA BUFFERS ---
        self.history_size = 60
        self.cpu_data = deque([0] * self.history_size, maxlen=self.history_size)
        self.ram_data = deque([0] * self.history_size, maxlen=self.history_size)
        self.gpu_data = deque([0] * self.history_size, maxlen=self.history_size)

        # --- ROUTER TELEMETRY PLACEHOLDERS ---
        self.route_distribution = {}
        self.router_avg_latency = 0
        self.memory_hit_rate = 0
        self.rag_hit_rate = 0
        self.tuner_state = {
            "hybrid_sensitivity": 0,
            "memory_sensitivity": 0,
            "rag_sensitivity": 0
        }
        self.router_health = "🟢 System Stable"

        self._setup_ui()
        self._start_hardware_monitor()
        eng = self._resolve_native_engine()
        if eng is not None and hasattr(eng, "load_finished"):
            try:
                eng.load_finished.connect(self._on_native_load_finished_telemetry)
            except Exception as e:
                logger.debug("Telemetry native load_finished connect skipped: %s", e)
        self._refresh_router_from_worker_snapshot()

    # ============================================================
    # UI SETUP
    # ============================================================
    def _setup_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Header
        title = QLabel("ADVANCED TELEMETRY")
        title.setObjectName("ViewTitle")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Main Dashboard Layout:
        # Row 1 -> Graph (left) + Latency/Capability (right)
        # Row 2 -> Router card full-width
        dashboard_layout = QVBoxLayout()
        dashboard_layout.setSpacing(20)

        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(20)

        # Left Column: Hardware Graph
        self.hardware_card = self._build_hardware_card()
        left_column = QVBoxLayout()
        left_column.setSpacing(0)
        left_column.addWidget(self.hardware_card)
        left_column.addStretch(1)
        top_row_layout.addLayout(left_column, stretch=2)

        # Right Column: Latency + Model capability + Router Metrics
        right_column = QVBoxLayout()
        self.latency_card = self._build_latency_card()
        self.model_capability_card = self._build_model_capability_card()
        self.router_card = self._build_router_card()
        right_column.addWidget(self.latency_card)
        right_column.addWidget(self.model_capability_card)
        right_column.addStretch()
        top_row_layout.addLayout(right_column, stretch=1)

        dashboard_layout.addLayout(top_row_layout)
        dashboard_layout.addWidget(self.router_card)
        layout.addLayout(dashboard_layout)
        if os.environ.get("QUBE_LLM_LOG_UI", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            try:
                from ui.components.llm_debug_log_panel import LLMDebugLogPanel

                self.llm_debug_log_panel = LLMDebugLogPanel()
                layout.addWidget(self.llm_debug_log_panel)
            except Exception:
                self.llm_debug_log_panel = None  # type: ignore[assignment]
        else:
            self.llm_debug_log_panel = None  # type: ignore[assignment]
        layout.addStretch()

    # ============================================================
    # HARDWARE CARD
    # ============================================================
    def _build_hardware_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("HardwareCard")
        frame.setSizePolicy(frame.sizePolicy().horizontalPolicy(), frame.sizePolicy().verticalPolicy())
        # Align hardware-card bottom with "Native LLM — Model capability" card bottom at compact sizes.
        frame.setMinimumHeight(355)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)

        header_layout = QHBoxLayout()
        header = QLabel("System Load Timeline (%)")
        header.setProperty("class", "SectionHeaderLabel")

        cpu_item, self.live_cpu_lbl = self._create_legend_item("CPU: 0%", "#10b981")
        ram_item, self.live_ram_lbl = self._create_legend_item("RAM: 0%", "#3b82f6")
        gpu_item, self.live_gpu_lbl = self._create_legend_item("GPU: 0%", "#8b5cf6")

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(self.live_cpu_lbl)
        header_layout.addWidget(self.live_ram_lbl)
        header_layout.addWidget(self.live_gpu_lbl)
        layout.addLayout(header_layout)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setYRange(-5, 105)
        self.plot_widget.setLimits(yMin=-5, yMax=105, minYRange=110, maxYRange=110)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(220)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getViewBox().setMouseEnabled(x=False, y=False)
        self.plot_widget.getViewBox().setMouseMode(self.plot_widget.getViewBox().PanMode)
        self.plot_widget.getPlotItem().hideButtons()
        # Fully disable wheel interaction so page scrolling always wins.
        self.plot_widget.wheelEvent = self._ignore_plot_wheel_event

        self.plot_widget.getAxis('bottom').setStyle(showValues=False)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#94a3b8', width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#94a3b8'))

        self.cpu_line = self.plot_widget.plot(pen=pg.mkPen('#10b981', width=2))
        self.ram_line = self.plot_widget.plot(pen=pg.mkPen('#3b82f6', width=2))
        self.gpu_line = self.plot_widget.plot(pen=pg.mkPen('#8b5cf6', width=2))

        layout.addWidget(self.plot_widget)
        return frame

    # ============================================================
    # LATENCY CARD
    # ============================================================
    def _build_latency_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LatencyCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Pipeline Latency")
        header.setProperty("class", "SectionHeaderLabel")
        layout.addWidget(header)

        stt_layout, self.stt_val = self._make_metric_row("Whisper STT", "Voice-to-Text inference time", "-- ms")
        ttft_layout, self.ttft_val = self._make_metric_row("LLM TTFT", "Time To First Token", "-- ms")
        tts_layout, self.tts_val = self._make_metric_row("TTS Generation", "Text-to-Speech synthesis time", "-- ms")

        layout.addLayout(stt_layout)
        layout.addLayout(ttft_layout)
        layout.addLayout(tts_layout)
        layout.addStretch()
        return frame

    # ============================================================
    # NATIVE MODEL CAPABILITY (telemetry only)
    # ============================================================
    def _build_model_capability_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LatencyCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Native LLM — Model capability")
        header.setProperty("class", "SectionHeaderLabel")
        layout.addWidget(header)

        model_row, self._cap_model_val = self._make_metric_row(
            "Model",
            "Loaded native model identity",
            "—",
        )
        reasoning_row, self._cap_reasoning_val = self._make_metric_row(
            "Reasoning-capable",
            "Thinking token capability",
            "—",
        )
        mode_row, self._cap_mode_val = self._make_metric_row(
            "Execution mode",
            "Resolved policy execution mode",
            "—",
        )
        conf_row, self._cap_conf_val = self._make_metric_row(
            "Confidence",
            "Model capability classification confidence",
            "—",
        )
        layout.addLayout(model_row)
        layout.addLayout(reasoning_row)
        layout.addLayout(mode_row)
        layout.addLayout(conf_row)
        layout.addStretch()

        return frame

    # ============================================================
    # ROUTER CARD
    # ============================================================
    def _build_router_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LatencyCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Router Intelligence")
        header.setProperty("class", "SectionHeaderLabel")
        layout.addWidget(header)

        routes_row, self.route_val = self._make_metric_row(
            "Routes",
            "Current route distribution",
            "—",
        )
        latency_row, self.latency_router_val = self._make_metric_row(
            "Avg retrieval phase",
            "Mean retrieval latency across turns",
            "—",
        )
        memory_row, self.memory_val = self._make_metric_row(
            "MEMORY route share",
            "Portion of turns routed to memory",
            "—",
        )
        rag_row, self.rag_val = self._make_metric_row(
            "RAG route share",
            "Portion of turns routed to RAG",
            "—",
        )
        tuner_row, self.tuner_val = self._make_metric_row(
            "Tuner weights",
            "Adaptive router weight state",
            "—",
        )
        health_row, self.health_val = self._make_metric_row(
            "System health",
            "Router health summary",
            "—",
        )

        layout.addLayout(routes_row)
        layout.addLayout(latency_row)
        layout.addLayout(memory_row)
        layout.addLayout(rag_row)
        layout.addLayout(tuner_row)
        layout.addLayout(health_row)
        layout.addStretch()

        return frame

    def _make_metric_row(self, title: str, description: str, value_text: str) -> tuple[QHBoxLayout, QLabel]:
        row = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setProperty("class", "MetricTitle")
        desc_lbl = QLabel(description)
        desc_lbl.setProperty("class", "MetricSubtext")
        desc_lbl.setWordWrap(True)
        vbox.addWidget(title_lbl)
        vbox.addWidget(desc_lbl)
        val_lbl = QLabel(value_text)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Match value size to field label scale (non-oversized).
        val_lbl.setProperty("class", "MetricTitle")
        row.addLayout(vbox)
        row.addStretch()
        row.addWidget(val_lbl)
        return row, val_lbl

    # ============================================================
    # LEGEND CREATOR
    # ============================================================
    def _create_legend_item(self, initial_text, color):
        from PyQt6.QtWidgets import QWidget
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        pill = QFrame()
        pill.setFixedSize(4, 14)
        pill.setStyleSheet(f"background-color: {color}; border-radius: 2px;")

        lbl = QLabel(initial_text)
        lbl.setProperty("class", "LiveMetricText")
        lbl.setStyleSheet(f"""
            color: {color};
            font-weight: bold;
            font-size: 13px;
            opacity: 1.0;
        """)

        layout.addWidget(pill)
        layout.addWidget(lbl)
        return container, lbl

    # ============================================================
    # HARDWARE MONITOR
    # ============================================================
    def _start_hardware_monitor(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_hardware)
        self.timer.start(1000)

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_card_surfaces()
        self._sync_hardware_card_min_height()
        self._refresh_model_capability_labels()
        self._refresh_router_from_worker_snapshot()
    
    def refresh_after_theme_toggle(self) -> None:
        """Keep telemetry card shells aligned with global light/dark theme."""
        self._apply_card_surfaces()
        self._sync_hardware_card_min_height()

    def _on_native_load_finished_telemetry(self, _ok: bool, _msg: str) -> None:
        self._refresh_model_capability_labels()

    def _resolve_native_engine(self):
        """Prefer ctor ref; fall back to workers dict (same object in normal app startup)."""
        return self._native_engine or (
            self.workers.get("native_engine") if getattr(self, "workers", None) else None
        )

    def _refresh_router_from_worker_snapshot(self) -> None:
        """Live read of in-memory router stats (no LLM turn required for tuner weights / idle copy)."""
        llm = self.workers.get("llm") if getattr(self, "workers", None) else None
        if not llm or not hasattr(llm, "telemetry") or not hasattr(llm, "router_tuner"):
            return
        try:
            summary = llm.telemetry.summarize()
            tuner_state = llm.router_tuner.get_weights()
            self.update_router_telemetry(summary or {}, tuner_state or {})
        except Exception as e:
            logger.debug("Router telemetry snapshot failed: %s", e)

    def _refresh_hardware(self):
        self._refresh_model_capability_labels()
        self._refresh_router_from_worker_snapshot()

        try:
            cpu = int(psutil.cpu_percent())
            ram = int(psutil.virtual_memory().percent)
        except Exception as e:
            logger.debug("CPU/RAM read failed: %s", e)
            cpu, ram = 0, 0
        try:
            gpu = int(self.gpu_monitor.get_load()) if self.gpu_monitor else 0
        except Exception as e:
            logger.debug("GPU load read failed: %s", e)
            gpu = 0

        self.live_cpu_lbl.setText(f"CPU: {cpu}%")
        self.live_ram_lbl.setText(f"RAM: {ram}%")
        self.live_gpu_lbl.setText(f"GPU: {gpu}%")

        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        self.cpu_line.setData(list(self.cpu_data))
        self.ram_line.setData(list(self.ram_data))
        self.gpu_line.setData(list(self.gpu_data))

    def _apply_card_surfaces(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        bg = "#232337" if is_dark else "#E9EFF5"
        border = "rgba(255, 255, 255, 0.08)" if is_dark else "#dbe4ee"
        for card in (
            getattr(self, "hardware_card", None),
            getattr(self, "latency_card", None),
            getattr(self, "model_capability_card", None),
            getattr(self, "router_card", None),
        ):
            if card is not None:
                name = card.objectName() or "TelemetryCard"
                card.setStyleSheet(
                    f"QFrame#{name} {{ background-color: {bg}; border: 1px solid {border}; border-radius: 12px; }}"
                )

    def _sync_hardware_card_min_height(self) -> None:
        """Match hardware-card bottom to end of Native LLM card (latency + spacing + model capability)."""
        if not all(
            hasattr(self, n)
            for n in ("hardware_card", "latency_card", "model_capability_card")
        ):
            return
        latency_h = int(self.latency_card.sizeHint().height())
        model_h = int(self.model_capability_card.sizeHint().height())
        # right_column spacing between cards is 6? currently default QVBoxLayout; use explicit conservative bridge.
        inter_card_gap = 20
        target = max(300, latency_h + inter_card_gap + model_h)
        self.hardware_card.setMinimumHeight(target)

    @staticmethod
    def _ignore_plot_wheel_event(_event: QWheelEvent) -> None:
        _event.ignore()

    def _refresh_model_capability_labels(self) -> None:
        eng = self._resolve_native_engine()
        mode = get_engine_mode()
        if eng is None:
            self._cap_model_val.setText("—")
            self._cap_reasoning_val.setText("—")
            self._cap_mode_val.setText(mode)
            self._cap_conf_val.setText("—")
            return
        try:
            snap = eng.get_model_reasoning_telemetry()
        except Exception as e:
            logger.debug("Model capability telemetry failed: %s", e)
            self._cap_model_val.setText("—")
            self._cap_reasoning_val.setText("—")
            self._cap_mode_val.setText(mode)
            self._cap_conf_val.setText("—")
            return
        if not snap.get("loaded"):
            self._cap_model_val.setText("—")
            self._cap_reasoning_val.setText("—")
            self._cap_mode_val.setText(str(snap.get("policy_execution_mode", mode)))
            self._cap_conf_val.setText("—")
            return
        base = snap.get("model_basename") or ""
        name = snap.get("model_name") or ""
        if name and base and name != base:
            self._cap_model_val.setText(f"{base} ({name})")
        elif base:
            self._cap_model_val.setText(str(base))
        elif name:
            self._cap_model_val.setText(str(name))
        else:
            self._cap_model_val.setText("—")
        sup = snap.get("supports_thinking_tokens")
        self._cap_reasoning_val.setText("yes" if sup else "no")
        pol_mode = snap.get("policy_execution_mode")
        det_mode = snap.get("execution_mode", "unknown")
        mode_txt = pol_mode if pol_mode else det_mode
        self._cap_mode_val.setText(str(mode_txt))
        conf = snap.get("confidence")
        if conf is None:
            self._cap_conf_val.setText("—")
        else:
            try:
                self._cap_conf_val.setText(f"{float(conf):.2f}")
            except (TypeError, ValueError):
                self._cap_conf_val.setText("—")

    # ============================================================
    # LATENCY UPDATE SLOTS
    # ============================================================
    def update_stt_latency(self, ms: float):
        self.stt_val.setText(f"{int(ms)} ms")

    def update_ttft_latency(self, ms: float):
        self.ttft_val.setText(f"{int(ms)} ms")

    def update_tts_latency(self, ms: float):
        self.tts_val.setText(f"{int(ms)} ms")

    # ============================================================
    # ROUTER TELEMETRY UPDATE SLOT
    # ============================================================
    def update_router_telemetry(self, summary: dict | None, tuner_state: dict | None):
        summary = summary or {}
        tuner_state = tuner_state or {}
        routes = summary.get("route_distribution") or {}
        total = int(summary.get("total_requests") or 0)

        if routes:
            route_txt = ", ".join(f"{k}: {v}" for k, v in sorted(routes.items()))
        else:
            route_txt = "—"
        self.route_val.setText(route_txt)

        # AdaptiveRouterSelfTunerV2.get_weights() uses keys hybrid / memory / rag (not *_sensitivity).
        try:
            hy = float(
                tuner_state.get("hybrid_sensitivity", tuner_state.get("hybrid", 1.0))
            )
            mem_w = float(
                tuner_state.get("memory_sensitivity", tuner_state.get("memory", 1.0))
            )
            rag_w = float(
                tuner_state.get("rag_sensitivity", tuner_state.get("rag", 1.0))
            )
        except (TypeError, ValueError):
            hy, mem_w, rag_w = 1.0, 1.0, 1.0

        self.tuner_val.setText(f"h:{hy:.2f} m:{mem_w:.2f} r:{rag_w:.2f}")

        if total <= 0:
            self.latency_router_val.setText("—")
            self.memory_val.setText("—")
            self.rag_val.setText("—")
            self.health_val.setText("⚪ Idle")
            self.health_val.setToolTip("Chat to record routing + retrieval latency.")
            return

        avg_lat = float(summary.get("avg_latency_ms") or 0)
        self.latency_router_val.setText(f"{avg_lat:.1f} ms")

        memory_count = routes.get("MEMORY", 0)
        rag_count = routes.get("RAG", 0)
        self.memory_val.setText(f"{memory_count}/{total} ({memory_count / total:.1%})")
        self.rag_val.setText(f"{rag_count}/{total} ({rag_count / total:.1%})")

        hybrid_ratio = routes.get("HYBRID", 0) / total
        if hybrid_ratio > 0.6:
            health = "⚠️ Over-reliance on HYBRID"
        elif mem_w < 0.6:
            health = "⚠️ Memory recall degraded"
        elif avg_lat > 1200:
            health = "⚠️ High latency"
        else:
            health = "🟢 System stable"

        self.health_val.setText(health)
        self.health_val.setToolTip(f"System health: {health}")