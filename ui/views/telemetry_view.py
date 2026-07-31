import logging
import os
import psutil
from collections import deque
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QScrollArea, QToolButton
)
from PyQt6.QtCore import Qt, QTimer, QSize
import pyqtgraph as pg
import qtawesome as qta

from ui.components.page_tour_help_button import PageTourHelpButton
from core.app_settings import get_engine_mode
from core.inference_transparency import aggregate_app_transparency
from core.theme.accessors import theme_for
from core.theme.view_theme import view_resolved_theme
from core.theme.color_utils import rgba_tuple
from ui.shell_theme import muted_icon_color, telemetry_metric_colors

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
        self._refresh_web_discovery_snapshot()

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

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        title = QLabel("Advanced Telemetry")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        header_row.addWidget(title)
        self.page_tour_help_btn = PageTourHelpButton(
            "telemetry",
            area_display_name="Advanced Telemetry",
            parent=content,
        )
        header_row.addWidget(self.page_tour_help_btn)
        header_row.addStretch(1)
        layout.addLayout(header_row)

        # Main Dashboard Layout:
        # Row 1 -> Graph (left) + Latency/Capability (right)
        # Row 2 -> Router Intelligence + Sidecar Cognition (side by side)
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

        # Right Column: Latency + Model capability
        right_column = QVBoxLayout()
        self.latency_card = self._build_latency_card()
        self.model_capability_card = self._build_model_capability_card()
        right_column.addWidget(self.latency_card)
        right_column.addWidget(self.model_capability_card)
        right_column.addStretch()
        top_row_layout.addLayout(right_column, stretch=1)

        bottom_row_layout = QHBoxLayout()
        bottom_row_layout.setSpacing(20)
        self.router_card = self._build_router_card()
        self.sidecar_card = self._build_sidecar_card()
        bottom_row_layout.addWidget(self.router_card, stretch=1)
        bottom_row_layout.addWidget(self.sidecar_card, stretch=1)

        discovery_row_layout = QHBoxLayout()
        discovery_row_layout.setSpacing(20)
        self.discovery_card = self._build_web_discovery_card()
        discovery_row_layout.addWidget(self.discovery_card, stretch=1)

        self.inference_transparency_card = self._build_inference_transparency_card()

        from ui.components.session_egress_panel import SessionEgressPanel

        self.session_egress_panel = SessionEgressPanel()

        dashboard_layout.addLayout(top_row_layout)
        dashboard_layout.addLayout(bottom_row_layout)
        dashboard_layout.addLayout(discovery_row_layout)
        dashboard_layout.addWidget(self.inference_transparency_card)
        dashboard_layout.addWidget(self.session_egress_panel)
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
        header.setToolTip(
            "Rolling 60-second chart of processor, memory, and GPU utilization."
        )

        cpu_item, self.live_cpu_lbl, self._cpu_legend_pill = self._create_legend_item(
            "CPU: 0%", theme_for(is_dark=True).success, "Processor utilization across all cores."
        )
        ram_item, self.live_ram_lbl, self._ram_legend_pill = self._create_legend_item(
            "RAM: 0%", theme_for(is_dark=True).info, "System memory currently in use."
        )
        gpu_item, self.live_gpu_lbl, self._gpu_legend_pill = self._create_legend_item(
            "GPU: 0%", theme_for(is_dark=True).accent, "Graphics processor compute utilization."
        )

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(cpu_item)
        header_layout.addWidget(ram_item)
        header_layout.addWidget(gpu_item)
        layout.addLayout(header_layout)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setYRange(-5, 105)
        self.plot_widget.setLimits(yMin=-5, yMax=105, minYRange=110, maxYRange=110)
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(220)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setToolTip("Live system load over the last 60 seconds.")
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.getViewBox().setMouseEnabled(x=False, y=False)
        self.plot_widget.getViewBox().setMouseMode(self.plot_widget.getViewBox().PanMode)
        self.plot_widget.getPlotItem().hideButtons()
        # Fully disable wheel interaction so page scrolling always wins.
        self.plot_widget.wheelEvent = self._ignore_plot_wheel_event

        self.plot_widget.getAxis('bottom').setStyle(showValues=False)

        self.cpu_line = self.plot_widget.plot(pen=pg.mkPen(width=2))
        self.ram_line = self.plot_widget.plot(pen=pg.mkPen(width=2))
        self.gpu_line = self.plot_widget.plot(pen=pg.mkPen(width=2))
        self._apply_plot_theme()

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
        header.setToolTip(
            "End-to-end timing for speech-to-text, first LLM token, and text-to-speech."
        )
        layout.addWidget(header)

        stt_layout, self.stt_val = self._make_metric_row(
            "Whisper STT",
            "Voice-to-Text inference time",
            "-- ms",
            "Measured in STTWorker as wall-clock time from entering transcription to final text assembly, then emitted via stt_latency (ms).",
        )
        ttft_layout, self.ttft_val = self._make_metric_row(
            "LLM TTFT",
            "Time To First Token",
            "-- ms",
            "Measured in LLMWorker as wall-clock time from stream request start to first emitted token, via ttft_latency (ms).",
        )
        tts_layout, self.tts_val = self._make_metric_row(
            "TTS Generation",
            "Text-to-Speech synthesis time",
            "-- ms",
            "Measured in TTSWorker as wall-clock time from sentence synth start to first playable PCM chunk, then emitted via tts_latency (ms).",
        )

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
        header.setToolTip(
            "Capability profile of the currently loaded native model, including reasoning support."
        )
        layout.addWidget(header)

        model_row, self._cap_model_val = self._make_metric_row(
            "Model",
            "Loaded native model identity",
            "—",
            "From NativeLlamaEngine.get_model_reasoning_telemetry(): prefers model file basename and appends profile model_name when both differ.",
        )
        reasoning_row, self._cap_reasoning_val = self._make_metric_row(
            "Reasoning-capable",
            "Thinking token capability",
            "—",
            "Boolean from supports_thinking_tokens in native engine reasoning profile detection (tokenizer/template/name signals).",
        )
        mode_row, self._cap_mode_val = self._make_metric_row(
            "Execution mode",
            "Resolved policy execution mode",
            "—",
            "Shows policy_execution_mode when available (resolved by execution_policy), otherwise profile execution_mode fallback.",
        )
        conf_row, self._cap_conf_val = self._make_metric_row(
            "Confidence",
            "Model capability classification confidence",
            "—",
            "Numeric reasoning-profile confidence from native detection pipeline, formatted to 2 decimals when present.",
        )
        pg_row, self._cap_publisher_guidance_val = self._make_metric_row(
            "Publisher guidance",
            "README/curated publisher contract",
            "—",
            "When set: publisher_default_reasoning and thinking tags from deterministic README extraction (not pasted presets).",
        )
        layout.addLayout(model_row)
        layout.addLayout(reasoning_row)
        layout.addLayout(mode_row)
        layout.addLayout(conf_row)
        layout.addLayout(pg_row)
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
        header.setToolTip(
            "Live cognitive routing stats: which tools fire, retrieval latency, and tuner weights."
        )
        layout.addWidget(header)

        routes_row, self.route_val = self._make_metric_row(
            "Routes",
            "Current route distribution",
            "—",
            "Counts by route from RouterTelemetryBrain.summarize() over its rolling in-memory event deque (max 200 samples).",
        )
        latency_row, self.latency_router_val = self._make_metric_row(
            "Avg retrieval phase",
            "Mean retrieval latency across turns",
            "—",
            "Average of per-turn retrieval-phase latency_ms recorded in LLMWorker (route/tool phase timing only, before token streaming).",
        )
        memory_row, self.memory_val = self._make_metric_row(
            "MEMORY route share",
            "Portion of turns routed to memory",
            "—",
            "Computed as MEMORY route count / total routed requests from telemetry summary, shown as count and percentage.",
        )
        rag_row, self.rag_val = self._make_metric_row(
            "RAG route share",
            "Portion of turns routed to RAG",
            "—",
            "Computed as RAG route count / total routed requests from telemetry summary, shown as count and percentage.",
        )
        tuner_row, self.tuner_val = self._make_metric_row(
            "Tuner weights",
            "Adaptive router weight state",
            "—",
            "Live adaptive weights from AdaptiveRouterSelfTunerV2.get_weights(): hybrid/memory/rag sensitivities (clamped 0.4 to 2.0).",
        )
        health_row, self.health_val = self._make_metric_row(
            "System health",
            "Router health summary",
            "—",
            "Rule-based status from telemetry snapshot: flags HYBRID overuse (>60%), weak memory weight (<0.6), or high latency (>1200 ms).",
        )

        layout.addLayout(routes_row)
        layout.addLayout(latency_row)
        layout.addLayout(memory_row)
        layout.addLayout(rag_row)
        layout.addLayout(tuner_row)
        layout.addLayout(health_row)
        layout.addStretch()

        return frame

    # ============================================================
    # SIDECAR CARD
    # ============================================================
    def _build_sidecar_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("LatencyCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Sidecar Cognition")
        header.setProperty("class", "SectionHeaderLabel")
        header.setToolTip(
            "CPU Qwen3 1.7B assist layer: health, queue depth, foreground latency, and rewrite/digest effectiveness."
        )
        layout.addWidget(header)

        status_row, self.sidecar_status_val = self._make_metric_row(
            "Status",
            "Runtime availability",
            "—",
            "Online / degraded / disabled from sidecar telemetry runtime snapshot.",
        )
        queue_row, self.sidecar_queue_val = self._make_metric_row(
            "Queue depth",
            "Pending sidecar jobs",
            "—",
            "Command queue size on SidecarLlmWorker (background titling, judge, digest, etc.).",
        )
        success_row, self.sidecar_success_val = self._make_metric_row(
            "Success rate",
            "Completed sidecar calls",
            "—",
            "ok / inference attempts over the rolling telemetry window (queue deferrals and ingest coalescing excluded).",
        )
        fg_row, self.sidecar_fg_p95_val = self._make_metric_row(
            "Foreground p95",
            "Rewrite + digest latency",
            "—",
            "95th percentile end-to-end latency for foreground tasks (query rewrite, source digest).",
        )
        rewrite_row, self.sidecar_rewrite_val = self._make_metric_row(
            "Query rewrite",
            "Assistive follow-up expansion",
            "—",
            "Applied / attempted on discourse follow-up turns (confidence-gated; never changes route).",
        )
        health_row, self.sidecar_health_val = self._make_metric_row(
            "System health",
            "Sidecar health summary",
            "—",
            "Rule-based status from queue depth, failure rate, and foreground latency.",
        )

        layout.addLayout(status_row)
        layout.addLayout(queue_row)
        layout.addLayout(success_row)
        layout.addLayout(fg_row)
        layout.addLayout(rewrite_row)
        layout.addLayout(health_row)
        layout.addStretch()

        return frame

    # ============================================================
    # WEB DISCOVERY CARD (R10 / Theme B)
    # ============================================================
    def _build_web_discovery_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("WebDiscoveryCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Web discovery")
        header.setProperty("class", "SectionHeaderLabel")
        header.setToolTip(
            "Live web search discovery policy: privacy tier, DDG budgets, pacing, "
            "and backoff — read-only; mirrors Settings → Knowledge → Web search discovery."
        )
        layout.addWidget(header)

        tier_row, self.discovery_tier_val = self._make_metric_row(
            "Privacy tier",
            "Active SERP discovery tier",
            "—",
            "Same setting as Settings → Knowledge → Web search discovery → Privacy tier.",
        )
        primary_row, self.discovery_primary_val = self._make_metric_row(
            "Primary provider",
            "Current discovery route",
            "—",
            "Primary SERP provider for @internet / Hybrid Internet / web routing.",
        )
        burst_row, self.discovery_burst_val = self._make_metric_row(
            "DDG burst budget",
            "Live DuckDuckGo calls in burst window",
            "—",
            "Rolling burst cap for live DDG HTTP requests (cache hits excluded).",
        )
        session_row, self.discovery_session_val = self._make_metric_row(
            "DDG session budget",
            "Live DuckDuckGo calls in session window",
            "—",
            "Rolling session cap for live DDG HTTP requests (cache hits excluded).",
        )
        pacing_row, self.discovery_pacing_val = self._make_metric_row(
            "Pacing",
            "Minimum gap between live DDG queries",
            "—",
            "Doubles automatically in conservative mode after repeated bot challenges.",
        )
        health_row, self.discovery_health_val = self._make_metric_row(
            "System health",
            "Discovery health summary",
            "—",
            "Rule-based status from backoff, budgets, and conservative pacing.",
        )

        layout.addLayout(tier_row)
        layout.addLayout(primary_row)
        layout.addLayout(burst_row)
        layout.addLayout(session_row)
        layout.addLayout(pacing_row)
        layout.addLayout(health_row)
        layout.addStretch()

        return frame

    # ============================================================
    # INFERENCE TRANSPARENCY
    # ============================================================
    def _build_inference_transparency_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("InferenceTransparencyCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        header = QLabel("Inference stack")
        header.setProperty("class", "SectionHeaderLabel")
        header.setToolTip(
            "Compile-time llama.cpp backend, hardware profile, layer configuration, "
            "and which compute path each model instance uses. Does not measure VRAM or timing."
        )
        layout.addWidget(header)

        build_row, self._inf_build_val = self._make_metric_row(
            "llama.cpp build",
            "Wheel backend and GPU offload support",
            "—",
            "From llama_print_system_info() and llama_supports_gpu_offload() at load time.",
        )
        hardware_row, self._inf_hardware_val = self._make_metric_row(
            "Hardware profile",
            "GPU memory kind and layer cap heuristics",
            "—",
            "From Qube GPU layer cap detection (NVIDIA, AMD discrete, AMD APU unified, Apple unified).",
        )
        native_row, self._inf_native_val = self._make_metric_row(
            "Native chat",
            "Loaded model and requested GPU layers",
            "—",
            "Requested layer count vs model depth from llama_model_n_layer(); not measured offload.",
        )
        embedder_row, self._inf_embedder_val = self._make_metric_row(
            "Embeddings",
            "RAG embedder compute path",
            "—",
            "GPU probe at embedder init (-1 layers) with CPU fallback.",
        )
        sidecar_row, self._inf_sidecar_val = self._make_metric_row(
            "Sidecar",
            "Auxiliary cognition compute path",
            "—",
            "Sidecar always loads with n_gpu_layers=0 (CPU).",
        )

        layout.addLayout(build_row)
        layout.addLayout(hardware_row)
        layout.addLayout(native_row)
        layout.addLayout(embedder_row)
        layout.addLayout(sidecar_row)
        layout.addStretch()
        return frame

    def _make_metric_row(
        self,
        title: str,
        description: str,
        value_text: str,
        tooltip_text: str = "",
    ) -> tuple[QHBoxLayout, QLabel]:
        row = QHBoxLayout()
        vbox = QVBoxLayout()
        vbox.setSpacing(2)
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setProperty("class", "MetricTitle")
        if tooltip_text:
            title_lbl.setToolTip(tooltip_text)
        title_row.addWidget(title_lbl)
        if tooltip_text:
            title_row.addWidget(self._make_metric_info_button(tooltip_text))
        title_row.addStretch()
        desc_lbl = QLabel(description)
        desc_lbl.setProperty("class", "MetricSubtext")
        desc_lbl.setWordWrap(True)
        vbox.addLayout(title_row)
        vbox.addWidget(desc_lbl)
        val_lbl = QLabel(value_text)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Match value size to field label scale (non-oversized).
        val_lbl.setProperty("class", "MetricTitle")
        if tooltip_text:
            val_lbl.setToolTip(tooltip_text)
        row.addLayout(vbox)
        row.addStretch()
        row.addWidget(val_lbl)
        return row, val_lbl

    def _make_metric_info_button(self, tooltip_text: str) -> QToolButton:
        btn = QToolButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip_text)
        btn.setIcon(qta.icon("fa5s.info-circle", color=muted_icon_color(self._theme())))
        btn.setIconSize(QSize(12, 12))
        btn.setAutoRaise(True)
        btn.setStyleSheet("QToolButton { border: none; padding: 0px; background: transparent; }")
        return btn

    def _theme(self, is_dark: bool | None = None):
        return view_resolved_theme(self, is_dark=is_dark)

    def _apply_plot_theme(self, is_dark: bool | None = None) -> None:
        theme = self._theme(is_dark)
        cpu_c, ram_c, gpu_c = telemetry_metric_colors(theme)
        axis_css = theme.text_muted
        axis_pen = rgba_tuple(axis_css)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=axis_pen, width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=axis_pen))
        self.cpu_line.setPen(pg.mkPen(rgba_tuple(cpu_c), width=2))
        self.ram_line.setPen(pg.mkPen(rgba_tuple(ram_c), width=2))
        self.gpu_line.setPen(pg.mkPen(rgba_tuple(gpu_c), width=2))
        for pill, lbl, color in (
            (getattr(self, "_cpu_legend_pill", None), getattr(self, "live_cpu_lbl", None), cpu_c),
            (getattr(self, "_ram_legend_pill", None), getattr(self, "live_ram_lbl", None), ram_c),
            (getattr(self, "_gpu_legend_pill", None), getattr(self, "live_gpu_lbl", None), gpu_c),
        ):
            if pill is not None:
                pill.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            if lbl is not None:
                lbl.setStyleSheet(
                    f"color: {color}; font-weight: bold; font-size: 13px; opacity: 1.0;"
                )

    # ============================================================
    # LEGEND CREATOR
    # ============================================================
    def _create_legend_item(self, initial_text, color, tooltip_text=""):
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
        if tooltip_text:
            lbl.setToolTip(tooltip_text)
            container.setToolTip(tooltip_text)

        layout.addWidget(pill)
        layout.addWidget(lbl)
        return container, lbl, pill

    def set_active_session_id(self, session_id: str | None) -> None:
        if hasattr(self, "session_egress_panel"):
            self.session_egress_panel.set_session_id(session_id)

    def _refresh_session_egress_panel(self) -> None:
        if hasattr(self, "session_egress_panel"):
            self.session_egress_panel.refresh()

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
        self._refresh_sidecar_from_worker_snapshot()
        self._refresh_session_egress_panel()
        self._refresh_web_discovery_snapshot()

    def refresh_after_theme_toggle(self) -> None:
        """Keep telemetry card shells aligned with global light/dark theme."""
        self._apply_card_surfaces()
        self._apply_plot_theme()
        self._sync_hardware_card_min_height()

    def _on_native_load_finished_telemetry(self, _ok: bool, _msg: str) -> None:
        self._refresh_model_capability_labels()
        self._refresh_inference_transparency_labels()

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

    def _refresh_sidecar_from_worker_snapshot(self) -> None:
        try:
            from core.sidecar_telemetry import get_sidecar_telemetry

            sw = self.workers.get("sidecar_worker") if getattr(self, "workers", None) else None
            if sw is not None and getattr(sw, "model_loaded", False):
                get_sidecar_telemetry().set_runtime_state(model_loaded=True)
            summary = get_sidecar_telemetry().summarize()
            self.update_sidecar_telemetry(summary or {})
        except Exception as e:
            logger.debug("Sidecar telemetry snapshot failed: %s", e)

    def _refresh_web_discovery_snapshot(self) -> None:
        try:
            from core.knowledge.discovery_telemetry import discovery_telemetry_snapshot

            self.update_web_discovery_telemetry(discovery_telemetry_snapshot())
        except Exception as e:
            logger.debug("Web discovery telemetry snapshot failed: %s", e)

    def _refresh_hardware(self):
        self._refresh_model_capability_labels()
        self._refresh_inference_transparency_labels()
        self._refresh_router_from_worker_snapshot()
        self._refresh_sidecar_from_worker_snapshot()
        self._refresh_session_egress_panel()
        self._refresh_web_discovery_snapshot()

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
        theme = self._theme()
        bg = theme.surface_elevated if theme.is_dark else theme.surface
        border = theme.border_subtle if theme.is_dark else theme.border
        for card in (
            getattr(self, "hardware_card", None),
            getattr(self, "latency_card", None),
            getattr(self, "model_capability_card", None),
            getattr(self, "router_card", None),
            getattr(self, "sidecar_card", None),
            getattr(self, "discovery_card", None),
            getattr(self, "inference_transparency_card", None),
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
            self._cap_publisher_guidance_val.setText("—")
            return
        if not snap.get("loaded"):
            self._cap_model_val.setText("—")
            self._cap_reasoning_val.setText("—")
            self._cap_mode_val.setText(str(snap.get("policy_execution_mode", mode)))
            self._cap_conf_val.setText("—")
            self._cap_publisher_guidance_val.setText("—")
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
        pg_src = snap.get("publisher_guidance_source")
        if pg_src:
            default = snap.get("publisher_default_reasoning") or "unknown"
            tags = snap.get("publisher_thinking_tags") or []
            tag_txt = ", ".join(str(t) for t in tags[:2]) if tags else "none"
            self._cap_publisher_guidance_val.setText(f"{pg_src}; default={default}; tags={tag_txt}")
        else:
            self._cap_publisher_guidance_val.setText("—")

    def _refresh_inference_transparency_labels(self) -> None:
        if not hasattr(self, "_inf_build_val"):
            return
        try:
            snap = aggregate_app_transparency(
                native_engine=self._resolve_native_engine(),
                embedder=self.workers.get("embedder") if getattr(self, "workers", None) else None,
                sidecar_worker=self.workers.get("sidecar_worker") if getattr(self, "workers", None) else None,
            )
        except Exception as e:
            logger.debug("Inference transparency refresh failed: %s", e)
            for lbl in (
                self._inf_build_val,
                self._inf_hardware_val,
                self._inf_native_val,
                self._inf_embedder_val,
                self._inf_sidecar_val,
            ):
                lbl.setText("—")
            return

        build = snap.get("build") or {}
        hardware = snap.get("hardware") or {}
        native = snap.get("native") or {}
        embedder = snap.get("embedder") or {}
        sidecar = snap.get("sidecar") or {}

        backend = str(build.get("backend_hint") or "unknown")
        ver = build.get("llama_cpp_python_version") or "?"
        offload = build.get("supports_gpu_offload")
        offload_txt = "yes" if offload else "no"
        self._inf_build_val.setText(f"{backend} (offload={offload_txt}, v{ver})")

        hw_label = hardware.get("gpu_memory_kind_label") or hardware.get("gpu_memory_kind") or "unknown"
        vram_gb = hardware.get("vram_budget_gb") or 0
        cap = hardware.get("max_safe_n_gpu_layers")
        unified = hardware.get("is_unified_gpu_memory")
        hw_txt = f"{hw_label}"
        if vram_gb:
            hw_txt += f", budget≈{vram_gb} GB"
        if cap is not None:
            hw_txt += f", cap={cap}"
        if unified:
            hw_txt += ", APU/unified"
        self._inf_hardware_val.setText(hw_txt)

        if native.get("loaded"):
            name = native.get("model_basename") or "model"
            params = native.get("model_n_params_label") or "?"
            layers = native.get("model_n_layers")
            layer_cfg = native.get("layer_configuration") or "?"
            self._inf_native_val.setText(f"{name} ({params}, {layers}L) — {layer_cfg}")
        else:
            mode = (snap.get("settings") or {}).get("engine_mode") or get_engine_mode()
            if mode != "internal":
                self._inf_native_val.setText("External engine (no native model)")
            else:
                self._inf_native_val.setText("Not loaded")

        emb_backend = embedder.get("backend")
        if emb_backend and emb_backend != "unknown":
            emb_name = embedder.get("model_basename") or "embedder"
            self._inf_embedder_val.setText(f"{emb_name} on {str(emb_backend).upper()}")
        else:
            self._inf_embedder_val.setText("—")

        if sidecar.get("loaded"):
            sc_name = sidecar.get("model_basename") or "sidecar"
            self._inf_sidecar_val.setText(f"{sc_name} on CPU (n_gpu_layers=0)")
        elif sidecar.get("degraded_reason"):
            self._inf_sidecar_val.setText(f"Unavailable ({sidecar.get('degraded_reason')})")
        else:
            self._inf_sidecar_val.setText("Not loaded")

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

    # ============================================================
    # WEB DISCOVERY TELEMETRY UPDATE SLOT
    # ============================================================
    def update_web_discovery_telemetry(self, snapshot: dict | None) -> None:
        snapshot = snapshot or {}
        tier_label = str(snapshot.get("privacy_tier_label") or "—")
        self.discovery_tier_val.setText(tier_label)

        primary = str(snapshot.get("primary_provider_label") or "—")
        backoff_summary = snapshot.get("backoff_summary")
        if backoff_summary:
            primary = f"{primary} · {backoff_summary}"
        self.discovery_primary_val.setText(primary)

        burst_limit = int(snapshot.get("burst_limit") or 0)
        if burst_limit <= 0:
            burst_txt = "Off"
        else:
            burst_used = int(snapshot.get("burst_used") or 0)
            burst_remaining = int(snapshot.get("burst_remaining") or 0)
            burst_txt = f"{burst_used}/{burst_limit} ({burst_remaining} left)"
        self.discovery_burst_val.setText(burst_txt)

        session_limit = int(snapshot.get("session_limit") or 0)
        if session_limit <= 0:
            session_txt = "Off"
        else:
            session_used = int(snapshot.get("session_used") or 0)
            session_remaining = int(snapshot.get("session_remaining") or 0)
            session_txt = f"{session_used}/{session_limit} ({session_remaining} left)"
        self.discovery_session_val.setText(session_txt)

        if not snapshot.get("pacing_enabled"):
            pacing_txt = "Off"
        else:
            effective = float(snapshot.get("pacing_effective_seconds") or 0)
            base = float(snapshot.get("pacing_base_seconds") or 0)
            if snapshot.get("conservative_mode") and effective > base + 0.01:
                pacing_txt = (
                    f"~{effective:.0f}s (conservative; base ~{base:.0f}s)"
                )
            else:
                pacing_txt = f"~{effective:.0f}s between live DDG queries"
        self.discovery_pacing_val.setText(pacing_txt)

        from core.knowledge.discovery_telemetry import format_discovery_health_status

        health = format_discovery_health_status(snapshot)
        self.discovery_health_val.setText(health)
        self.discovery_health_val.setToolTip(f"Discovery health: {health}")

    # ============================================================
    # SIDECAR TELEMETRY UPDATE SLOT
    # ============================================================
    def update_sidecar_telemetry(self, summary: dict | None) -> None:
        summary = summary or {}
        runtime = summary.get("runtime") or {}
        status = str(runtime.get("status") or "—")
        basename = str(runtime.get("active_model_basename") or "")
        if basename and not runtime.get("is_bundled_default", True):
            status = f"{status} · {basename}"
        self.sidecar_status_val.setText(status)

        depth = int(summary.get("queue_depth") or 0)
        self.sidecar_queue_val.setText(str(depth))

        total = int(summary.get("total_invocations") or 0)
        inference_attempts = int(summary.get("inference_attempts") or total)
        if inference_attempts <= 0 and total <= 0:
            self.sidecar_success_val.setText("—")
            self.sidecar_fg_p95_val.setText("—")
            self.sidecar_rewrite_val.setText("—")
            self.sidecar_health_val.setText(str(summary.get("health") or "⚪ Idle"))
            tip = summary.get("health_tip") or "Chat or background jobs to populate sidecar stats."
            self.sidecar_health_val.setToolTip(tip)
            return

        rate = float(summary.get("success_rate") or 0.0)
        if inference_attempts != total and total > inference_attempts:
            success_label = (
                f"{rate:.0%} ({inference_attempts} tasks, {total - inference_attempts} skipped)"
            )
            success_tip = (
                "Success rate counts completed inference only; queue deferrals and "
                "ingest coalescing are excluded from the denominator."
            )
        else:
            success_label = f"{rate:.0%} ({inference_attempts} tasks)"
            success_tip = "ok / inference attempts over the rolling telemetry window."
        self.sidecar_success_val.setText(success_label)
        self.sidecar_success_val.setToolTip(success_tip)

        fg = summary.get("foreground") or {}
        p95 = float(fg.get("p95_latency_ms") or 0.0)
        p95_wait = float(fg.get("p95_wait_ms") or 0.0)
        timeout_rate = float(fg.get("timeout_rate") or 0.0)
        fg_label = f"{p95:.0f} ms"
        if p95_wait > 0 and fg.get("attempts"):
            fg_label += f" · wait {p95_wait:.0f}ms"
        if fg.get("attempts"):
            fg_label += f" · timeout {timeout_rate:.0%}"
        self.sidecar_fg_p95_val.setText(fg_label if fg.get("attempts") else "—")

        rewrite = summary.get("rewrite") or {}
        attempted = int(rewrite.get("attempted") or 0)
        applied = int(rewrite.get("applied") or 0)
        if attempted:
            self.sidecar_rewrite_val.setText(
                f"{applied}/{attempted} ({float(rewrite.get('apply_rate') or 0):.0%})"
            )
        else:
            self.sidecar_rewrite_val.setText("—")

        health = str(summary.get("health") or "—")
        self.sidecar_health_val.setText(health)
        self.sidecar_health_val.setToolTip(str(summary.get("health_tip") or health))