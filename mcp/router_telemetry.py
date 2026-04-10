import time
import logging
from collections import deque

logger = logging.getLogger("Qube.RouterTelemetry")


class RouterTelemetryBrain:
    """
    Lightweight in-memory telemetry collector for routing behavior.
    """

    def __init__(self, max_samples=200):
        self.events = deque(maxlen=max_samples)

    def log(self, event: dict):
        event["ts"] = time.time()
        self.events.append(event)

        logger.debug(
            f"[Telemetry] route={event.get('route')} "
            f"mem={event.get('memory_hits')} rag={event.get('rag_hits')} "
            f"lat={event.get('latency_ms'):.1f}ms"
        )

    def summarize(self):
        if not self.events:
            return {}

        routes = {}
        total_latency = 0

        for e in self.events:
            r = e.get("route", "unknown")
            routes[r] = routes.get(r, 0) + 1
            total_latency += e.get("latency_ms", 0)

        return {
            "total_requests": len(self.events),
            "route_distribution": routes,
            "avg_latency_ms": total_latency / len(self.events),
        }