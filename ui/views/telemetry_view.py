import psutil
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg


class TelemetryView(QWidget):
    def __init__(self, workers: dict, gpu_monitor):
        super().__init__()
        self.workers = workers
        self.gpu_monitor = gpu_monitor

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

        # Right Column: Latency + Router Metrics
        right_column = QVBoxLayout()
        self.latency_card = self._build_latency_card()
        self.router_card = self._build_router_card()
        right_column.addWidget(self.latency_card)
        right_column.addWidget(self.router_card)
        right_column.addStretch()
        dashboard_layout.addLayout(right_column, stretch=1)

        layout.addLayout(dashboard_layout)
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

        self.live_cpu_lbl.setText(f"CPU: {cpu}%")
        self.live_ram_lbl.setText(f"RAM: {ram}%")
        self.live_gpu_lbl.setText(f"GPU: {gpu}%")

        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        self.cpu_line.setData(list(self.cpu_data))
        self.ram_line.setData(list(self.ram_data))
        self.gpu_line.setData(list(self.gpu_data))

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
        routes = summary.get("route_distribution", {})
        total = max(summary.get("total_requests", 1), 1)

        self.route_label.setText(f"Routes: {routes}")
        self.latency_router_label.setText(f"Avg Latency: {summary.get('avg_latency_ms', 0):.1f} ms")

        memory_count = routes.get("MEMORY", 0)
        rag_count = routes.get("RAG", 0)
        self.memory_label.setText(f"Memory Usage: {memory_count}/{total} ({memory_count/total:.1%})")
        self.rag_label.setText(f"RAG Usage: {rag_count}/{total} ({rag_count/total:.1%})")

        self.tuner_label.setText(
            f"Hybrid:{tuner_state['hybrid_sensitivity']:.2f} | "
            f"Memory:{tuner_state['memory_sensitivity']:.2f} | "
            f"RAG:{tuner_state['rag_sensitivity']:.2f}"
        )

        hybrid_ratio = routes.get("HYBRID", 0)/total
        if hybrid_ratio > 0.6:
            health = "⚠️ Over-reliance on HYBRID"
        elif tuner_state["memory_sensitivity"] < 0.6:
            health = "⚠️ Memory recall degraded"
        elif summary.get("avg_latency_ms", 0) > 1200:
            health = "⚠️ High latency"
        else:
            health = "🟢 System Stable"

        self.health_label.setText(f"System Health: {health}")