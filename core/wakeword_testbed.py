from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque


@dataclass
class WakewordScoreSample:
    raw: float
    smoothed: float
    ts: float


@dataclass
class AttemptResult:
    peak_confidence: float
    detected: bool
    timed_out: bool = False


@dataclass
class TestbedSummary:
    successes: int
    attempts: int
    completed_attempts: int
    missed_attempts: int
    average_confidence: float
    consistency_label: str
    false_triggers: int
    reliability_label: str
    explanation: str
    recommendation: str


@dataclass
class WakewordTestbedState:
    attempt_target: int = 5
    attempt_prep_seconds: int = 3
    detection_results: list[AttemptResult] = field(default_factory=list)
    false_positive_seconds: int = 30
    false_triggers: int = 0

    def record_detection_attempt(
        self,
        peak_confidence: float,
        detected: bool,
        timed_out: bool = False,
    ) -> None:
        self.detection_results.append(
            AttemptResult(
                peak_confidence=float(peak_confidence),
                detected=bool(detected),
                timed_out=bool(timed_out),
            )
        )

    def record_false_trigger(self) -> None:
        self.false_triggers += 1

    def build_summary(self) -> TestbedSummary:
        completed_attempts = len(self.detection_results)
        attempts = max(completed_attempts, self.attempt_target)
        successes = sum(1 for a in self.detection_results if a.detected)
        missed_attempts = sum(1 for a in self.detection_results if not a.detected)
        peaks = [a.peak_confidence for a in self.detection_results]
        avg_conf = float(mean(peaks)) if peaks else 0.0
        variance = float(pstdev(peaks)) if len(peaks) > 1 else 0.0
        if variance < 0.08:
            consistency = "High"
        elif variance < 0.18:
            consistency = "Medium"
        else:
            consistency = "Low"

        detection_rate = successes / float(max(1, self.attempt_target))
        penalty = min(self.false_triggers, 5) * 0.12
        reliability = (detection_rate * 0.6) + (avg_conf * 0.4) - penalty
        if reliability >= 0.72:
            reliability_label = "Works great"
        elif reliability >= 0.45:
            reliability_label = "Usable with minor issues"
        else:
            reliability_label = "Not recommended"

        if self.false_triggers > 0:
            explanation = "Triggered during normal speech."
            recommendation = "Retest with lower sensitivity."
        elif detection_rate < 0.8:
            explanation = "Missed detections in your environment."
            recommendation = "Retest with higher sensitivity."
        else:
            explanation = "Stable detections with low false-positive risk."
            recommendation = "Apply this wakeword."

        return TestbedSummary(
            successes=successes,
            attempts=attempts,
            completed_attempts=completed_attempts,
            missed_attempts=missed_attempts,
            average_confidence=avg_conf,
            consistency_label=consistency,
            false_triggers=self.false_triggers,
            reliability_label=reliability_label,
            explanation=explanation,
            recommendation=recommendation,
        )


class ConfidenceSmoother:
    def __init__(self, maxlen: int = 12):
        self._window: Deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> float:
        self._window.append(float(value))
        return float(mean(self._window))

    def reset(self) -> None:
        self._window.clear()
