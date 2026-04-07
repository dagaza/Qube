import psutil
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, 
)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

class TelemetryView(QWidget):
    def __init__(self, workers: dict, gpu_monitor):
        super().__init__()
        self.workers = workers
        self.gpu_monitor = gpu_monitor
        
        # --- GRAPH DATA BUFFERS ---
        # We will store the last 60 seconds of data for a smooth rolling chart
        self.history_size = 60
        self.cpu_data = deque([0] * self.history_size, maxlen=self.history_size)
        self.ram_data = deque([0] * self.history_size, maxlen=self.history_size)
        self.gpu_data = deque([0] * self.history_size, maxlen=self.history_size)
        
        self._setup_ui()
        self._start_hardware_monitor()

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

        # Left Column: The Scientific Graph
        self.hardware_card = self._build_hardware_card()
        dashboard_layout.addWidget(self.hardware_card, stretch=2) # Give the graph more horizontal space

        # Right Column: Latency Metrics
        self.latency_card = self._build_latency_card()
        dashboard_layout.addWidget(self.latency_card, stretch=1)

        layout.addLayout(dashboard_layout)
        layout.addStretch()

    def _build_hardware_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("HardwareCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        
        header_layout = QHBoxLayout()
        header = QLabel("System Load Timeline (%)")
        header.setProperty("class", "SectionHeaderLabel")
        
        self.live_cpu_lbl = QLabel("CPU: 0%")
        self.live_ram_lbl = QLabel("RAM: 0%")
        self.live_gpu_lbl = QLabel("GPU: 0%")

        for lbl in [self.live_cpu_lbl, self.live_ram_lbl, self.live_gpu_lbl]:
            lbl.setProperty("class", "LiveMetricText")

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(self.live_cpu_lbl)
        header_layout.addWidget(self.live_ram_lbl)
        header_layout.addWidget(self.live_gpu_lbl)
        
        layout.addLayout(header_layout)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        
        # CRITICAL: Let the CSS background shine through the graph canvas
        self.plot_widget.setBackground('transparent') 
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        
        self.plot_widget.getAxis('bottom').setStyle(showValues=False)
        # In a perfect world we would dynamically theme the axis, but grey is safe for both modes
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#94a3b8', width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#94a3b8'))
        
        self.cpu_line = self.plot_widget.plot(pen=pg.mkPen('#10b981', width=2)) # Emerald green
        self.ram_line = self.plot_widget.plot(pen=pg.mkPen('#3b82f6', width=2)) # Blue
        self.gpu_line = self.plot_widget.plot(pen=pg.mkPen('#8b5cf6', width=2)) # Purple
        
        layout.addWidget(self.plot_widget)
        return frame

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

    # --------------------------------------------------------- #
    #  UPDATER LOGIC                                            #
    # --------------------------------------------------------- #
    
    def _start_hardware_monitor(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._refresh_hardware)
        self.timer.start(1000) # Update the graph every 1 second

    def _refresh_hardware(self):
        # 1. Grab fresh metrics
        cpu = int(psutil.cpu_percent())
        ram = int(psutil.virtual_memory().percent)
        gpu = int(self.gpu_monitor.get_load()) if self.gpu_monitor else 0

        # 2. Update the text labels
        self.live_cpu_lbl.setText(f"CPU: {cpu}%")
        self.live_ram_lbl.setText(f"RAM: {ram}%")
        self.live_gpu_lbl.setText(f"GPU: {gpu}%")

        # 3. Push new data into the rolling buffers
        self.cpu_data.append(cpu)
        self.ram_data.append(ram)
        self.gpu_data.append(gpu)

        # 4. Redraw the scientific graph lines
        self.cpu_line.setData(list(self.cpu_data))
        self.ram_line.setData(list(self.ram_data))
        self.gpu_line.setData(list(self.gpu_data))

    # Slots for Workers to hit
    def update_stt_latency(self, ms: float):
        self.stt_val.setText(f"{int(ms)} ms")

    def update_ttft_latency(self, ms: float):
        self.ttft_val.setText(f"{int(ms)} ms")

    def update_tts_latency(self, ms: float):
        self.tts_val.setText(f"{int(ms)} ms")