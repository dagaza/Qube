import os
import psutil
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg


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

    # ============================================================
    # UI SETUP
    # ============================================================
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Header
        title = QLabel("ADVANCED TELEMETRY")
        title.setObjectName("ViewTitle")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Main Dashboard Layout (2 Columns)
        dashboard_layout = QHBoxLayout()
        dashboard_layout.setSpacing(20)

        # Left Column: Hardware Graph
        self.hardware_card = self._build_hardware_card()
        dashboard_layout.addWidget(self.hardware_card, stretch=2)

        # Right Column: Latency + Model capability + Router Metrics
        right_column = QVBoxLayout()
        self.latency_card = self._build_latency_card()
        self.model_capability_card = self._build_model_capability_card()
        self.router_card = self._build_router_card()
        right_column.addWidget(self.latency_card)
        right_column.addWidget(self.model_capability_card)
        right_column.addWidget(self.router_card)
        right_column.addStretch()
        dashboard_layout.addLayout(right_column, stretch=1)

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
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)

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

        def make_metric(title, description):
            row = QHBoxLayout()
            vbox = QVBoxLayout()
            vbox.setSpacing(2)
            title_lbl = QLabel(title)
            title_lbl.setProperty("class", "MetricTitle")
            desc_lbl = QLabel(description)
            desc_lbl.setProperty("class", "MetricSubtext")
            vbox.addWidget(title_lbl)
            vbox.addWidget(desc_lbl)
            val_lbl = QLabel("-- ms")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setProperty("class", "MetricValueLarge")
            row.addLayout(vbox)
            row.addStretch()
            row.addWidget(val_lbl)
            return row, val_lbl

        stt_layout, self.stt_val = make_metric("Whisper STT", "Voice-to-Text inference time")
        ttft_layout, self.ttft_val = make_metric("LLM TTFT", "Time To First Token")
        tts_layout, self.tts_val = make_metric("TTS Generation", "Text-to-Speech synthesis time")

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
        frame.setObjectName("ModelCapabilityCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(10)

        header = QLabel("Native LLM — Model capability")
        header.setProperty("class", "SectionHeaderLabel")
        layout.addWidget(header)

        self._cap_model_lbl = QLabel("Model: —")
        self._cap_reasoning_lbl = QLabel("Reasoning-capable: —")
        self._cap_mode_lbl = QLabel("Execution mode: —")
        self._cap_conf_lbl = QLabel("Confidence: —")
        for lbl in (
            self._cap_model_lbl,
            self._cap_reasoning_lbl,
            self._cap_mode_lbl,
            self._cap_conf_lbl,
        ):
            lbl.setProperty("class", "MetricSubtext")
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        return frame

    # ============================================================
    # ROUTER CARD
    # ============================================================
    def _build_router_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("RouterCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        header = QLabel("Router Intelligence")
        header.setProperty("class", "SectionHeaderLabel")
        layout.addWidget(header)

        self.route_label = QLabel("Routes: ...")
        self.latency_router_label = QLabel("Avg Latency: ...")
        self.memory_label = QLabel("Memory Usage: ...")
        self.rag_label = QLabel("RAG Usage: ...")
        self.tuner_label = QLabel("Tuner State: ...")
        self.health_label = QLabel("System Health: 🟢")

        for lbl in [
            self.route_label,
            self.latency_router_label,
            self.memory_label,
            self.rag_label,
            self.tuner_label,
            self.health_label
        ]:
            lbl.setProperty("class", "MetricSubtext")
            layout.addWidget(lbl)

        return frame

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

    def _refresh_hardware(self):
        cpu = int(psutil.cpu_percent())
        ram = int(psutil.virtual_memory().percent)
        gpu = int(self.gpu_monitor.get_load()) if self.gpu_monitor else 0

        self._refresh_model_capability_labels()

        self.live_cpu_lbl.setText(f"CPU: {cpu}%")
        self.live_ram_lbl.setText(f"RAM: {ram}%")
        self.live_gpu_lbl.setText(f"GPU: {gpu}%")

        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        self.cpu_line.setData(list(self.cpu_data))
        self.ram_line.setData(list(self.ram_data))
        self.gpu_line.setData(list(self.gpu_data))

    def _refresh_model_capability_labels(self) -> None:
        eng = self._native_engine
        if eng is None:
            self._cap_model_lbl.setText("Model: — (no native engine)")
            self._cap_reasoning_lbl.setText("Reasoning-capable: —")
            self._cap_mode_lbl.setText("Execution mode: —")
            self._cap_conf_lbl.setText("Confidence: —")
            return
        try:
            snap = eng.get_model_reasoning_telemetry()
        except Exception:
            self._cap_model_lbl.setText("Model: —")
            self._cap_reasoning_lbl.setText("Reasoning-capable: —")
            self._cap_mode_lbl.setText("Execution mode: —")
            self._cap_conf_lbl.setText("Confidence: —")
            return
        if not snap.get("loaded"):
            self._cap_model_lbl.setText("Model: (no native model loaded)")
            self._cap_reasoning_lbl.setText("Reasoning-capable: —")
            self._cap_mode_lbl.setText("Execution mode: —")
            self._cap_conf_lbl.setText("Confidence: —")
            return
        base = snap.get("model_basename") or ""
        name = snap.get("model_name") or ""
        if name and base and name != base:
            self._cap_model_lbl.setText(f"Model: {base} ({name})")
        elif base:
            self._cap_model_lbl.setText(f"Model: {base}")
        elif name:
            self._cap_model_lbl.setText(f"Model: {name}")
        else:
            self._cap_model_lbl.setText("Model: —")
        sup = snap.get("supports_thinking_tokens")
        self._cap_reasoning_lbl.setText(
            f"Reasoning-capable: {'yes' if sup else 'no'}"
        )
        pol_mode = snap.get("policy_execution_mode")
        det_mode = snap.get("execution_mode", "unknown")
        mode_txt = pol_mode if pol_mode else det_mode
        self._cap_mode_lbl.setText(f"Execution mode: {mode_txt}")
        conf = snap.get("confidence")
        if conf is None:
            self._cap_conf_lbl.setText("Confidence: —")
        else:
            self._cap_conf_lbl.setText(f"Confidence: {float(conf):.2f}")

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
    def update_router_telemetry(self, summary: dict, tuner_state: dict):
        routes = summary.get("route_distribution") or {}
        total = max(summary.get("total_requests", 1), 1)

        self.route_label.setText(f"Routes: {routes}")
        avg_lat = float(summary.get("avg_latency_ms") or 0)
        self.latency_router_label.setText(f"Avg Latency: {avg_lat:.1f} ms")

        memory_count = routes.get("MEMORY", 0)
        rag_count = routes.get("RAG", 0)
        self.memory_label.setText(f"Memory Usage: {memory_count}/{total} ({memory_count/total:.1%})")
        self.rag_label.setText(f"RAG Usage: {rag_count}/{total} ({rag_count/total:.1%})")

        # AdaptiveRouterSelfTunerV2.get_weights() uses keys hybrid / memory / rag (not *_sensitivity).
        hy = float(
            tuner_state.get("hybrid_sensitivity", tuner_state.get("hybrid", 1.0))
        )
        mem_w = float(
            tuner_state.get("memory_sensitivity", tuner_state.get("memory", 1.0))
        )
        rag_w = float(tuner_state.get("rag_sensitivity", tuner_state.get("rag", 1.0)))

        self.tuner_label.setText(
            f"Hybrid:{hy:.2f} | Memory:{mem_w:.2f} | RAG:{rag_w:.2f}"
        )

        hybrid_ratio = routes.get("HYBRID", 0) / total
        if hybrid_ratio > 0.6:
            health = "⚠️ Over-reliance on HYBRID"
        elif mem_w < 0.6:
            health = "⚠️ Memory recall degraded"
        elif avg_lat > 1200:
            health = "⚠️ High latency"
        else:
            health = "🟢 System Stable"

        self.health_label.setText(f"System Health: {health}")