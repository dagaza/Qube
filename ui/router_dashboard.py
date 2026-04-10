from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGridLayout, QGroupBox
)
from PyQt6.QtCore import QTimer


class RouterDashboard(QWidget):
    """
    Live Router Intelligence Dashboard
    """

    def __init__(self, telemetry, tuner):
        super().__init__()

        self.telemetry = telemetry
        self.tuner = tuner

        self.setWindowTitle("🧠 Router Dashboard")

        layout = QVBoxLayout()

        # =========================
        # ROUTE DISTRIBUTION
        # =========================
        self.route_label = QLabel("Route Distribution: loading...")

        # =========================
        # LATENCY
        # =========================
        self.latency_label = QLabel("Avg Latency: ...")

        # =========================
        # MEMORY / RAG HEALTH
        # =========================
        self.memory_label = QLabel("Memory Hit Rate: ...")
        self.rag_label = QLabel("RAG Hit Rate: ...")

        # =========================
        # TUNER STATE
        # =========================
        self.tuner_label = QLabel("Tuner State: ...")

        # =========================
        # HEALTH SIGNALS
        # =========================
        self.health_label = QLabel("System Health: OK")

        # layout
        layout.addWidget(self.route_label)
        layout.addWidget(self.latency_label)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.rag_label)
        layout.addWidget(self.tuner_label)
        layout.addWidget(self.health_label)

        self.setLayout(layout)

        # refresh loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)

    # ============================================================
    # LIVE UPDATE LOOP
    # ============================================================
    def refresh(self):
        summary = self.telemetry.summarize()
        tuner_state = self.tuner.debug()

        # ----------------------------
        # ROUTE DISTRIBUTION
        # ----------------------------
        routes = summary.get("route_distribution", {})
        self.route_label.setText(f"Routes: {routes}")

        # ----------------------------
        # LATENCY
        # ----------------------------
        avg_latency = summary.get("avg_latency_ms", 0)
        self.latency_label.setText(f"Avg Latency: {avg_latency:.1f} ms")

        # ----------------------------
        # MEMORY / RAG SIGNALS
        # ----------------------------
        total = summary.get("total_requests", 1)

        memory_count = routes.get("MEMORY", 0)
        rag_count = routes.get("RAG", 0)

        self.memory_label.setText(
            f"Memory Usage: {memory_count}/{total} "
            f"({memory_count/total:.1%})"
        )

        self.rag_label.setText(
            f"RAG Usage: {rag_count}/{total} "
            f"({rag_count/total:.1%})"
        )

        # ----------------------------
        # TUNER STATE
        # ----------------------------
        self.tuner_label.setText(
            f"Hybrid:{tuner_state['hybrid_sensitivity']:.2f} | "
            f"Memory:{tuner_state['memory_sensitivity']:.2f} | "
            f"RAG:{tuner_state['rag_sensitivity']:.2f}"
        )

        # ----------------------------
        # HEALTH SIGNALS
        # ----------------------------
        health = self._compute_health(summary, tuner_state)
        self.health_label.setText(health)

    # ============================================================
    def _compute_health(self, summary, tuner_state):
        routes = summary.get("route_distribution", {})
        total = max(summary.get("total_requests", 1), 1)

        hybrid_ratio = routes.get("HYBRID", 0) / total

        if hybrid_ratio > 0.6:
            return "⚠️ Over-reliance on HYBRID (latency risk)"

        if tuner_state["memory_sensitivity"] < 0.6:
            return "⚠️ Memory recall degraded"

        if summary.get("avg_latency_ms", 0) > 1200:
            return "⚠️ High latency detected"

        return "🟢 System Stable"