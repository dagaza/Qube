"""Wake-word evaluation metrics (milestone M5) — pure, stdlib-only, fully testable.

Wake-word quality is an *operating point*, not an accuracy number. This module computes
the operating-point metrics the review feedback asked for — recall/FRR, false-accepts per
hour, precision, adversarial false-accept rate, DET/ROC points, and latency percentiles —
across a threshold sweep, plus the ship-criteria threshold selection.

Inputs are plain dicts/lists so the math is trivially unit-testable without any audio or
model. ``evaluate.py`` turns a real corpus + model into these inputs.

Conventions:
- ``positives``: list of ``{"score": float, "environment": str, "latency_ms": float?}``
  (one true wake-phrase utterance each; label 1).
- ``adversarial``: list of ``{"score": float}`` (discrete sound-alikes; label 0) — drives
  precision + adversarial false-accept rate.
- ``longform``: list of ``{"fire_scores": [float], "duration_seconds": float}`` — candidate
  activation peaks over long negative audio; drives false-positives-per-hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    recall: float
    frr: float
    precision: float
    fp_per_hour: float
    adversarial_far: float
    true_positives: int
    adversarial_false_positives: int
    longform_false_positives: int
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    recall_quiet: float = 0.0
    recall_noisy: float = 0.0

    def as_dict(self) -> dict:
        return dict(sorted(self.__dict__.items()))


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in [0, 100]); 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def _recall(positives: list[dict], threshold: float, *, environment: str | None = None) -> tuple[int, int]:
    subset = positives if environment is None else [p for p in positives if p.get("environment") == environment]
    if not subset:
        return 0, 0
    detected = sum(1 for p in subset if float(p["score"]) >= threshold)
    return detected, len(subset)


def compute_threshold_metrics(
    positives: list[dict],
    adversarial: list[dict],
    longform: list[dict],
    threshold: float,
) -> ThresholdMetrics:
    tp, n_pos = _recall(positives, threshold)
    recall = tp / n_pos if n_pos else 0.0

    adv_fp = sum(1 for a in adversarial if float(a["score"]) >= threshold)
    adv_far = adv_fp / len(adversarial) if adversarial else 0.0

    longform_fp = sum(
        sum(1 for s in lf.get("fire_scores", []) if float(s) >= threshold) for lf in longform
    )
    total_hours = sum(float(lf.get("duration_seconds", 0.0)) for lf in longform) / 3600.0
    fp_per_hour = longform_fp / total_hours if total_hours > 0 else 0.0

    fp_total = adv_fp + longform_fp
    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 1.0

    detected_latencies = [
        float(p["latency_ms"]) for p in positives
        if "latency_ms" in p and float(p["score"]) >= threshold
    ]
    q_tp, q_n = _recall(positives, threshold, environment="quiet")
    n_tp, n_n = _recall(positives, threshold, environment="noisy")

    return ThresholdMetrics(
        threshold=round(float(threshold), 4),
        recall=recall,
        frr=1.0 - recall,
        precision=precision,
        fp_per_hour=fp_per_hour,
        adversarial_far=adv_far,
        true_positives=tp,
        adversarial_false_positives=adv_fp,
        longform_false_positives=longform_fp,
        latency_ms_p50=percentile(detected_latencies, 50),
        latency_ms_p95=percentile(detected_latencies, 95),
        recall_quiet=(q_tp / q_n if q_n else 0.0),
        recall_noisy=(n_tp / n_n if n_n else 0.0),
    )


def sweep(
    positives: list[dict],
    adversarial: list[dict],
    longform: list[dict],
    thresholds: list[float],
) -> list[ThresholdMetrics]:
    return [compute_threshold_metrics(positives, adversarial, longform, t) for t in thresholds]


def default_thresholds(start: float = 0.3, stop: float = 0.7, step: float = 0.05) -> list[float]:
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 4) for i in range(n)]


def select_threshold(
    metrics: list[ThresholdMetrics],
    *,
    max_fp_per_hour: float,
    min_recall: float,
) -> float | None:
    """Ship rule: max recall subject to FP/hr <= target and recall >= floor.

    Ties on recall break toward the *higher* threshold (fewer false accepts). Returns
    ``None`` if no threshold meets both constraints.
    """
    eligible = [m for m in metrics if m.fp_per_hour <= max_fp_per_hour and m.recall >= min_recall]
    if not eligible:
        return None
    best = max(eligible, key=lambda m: (m.recall, m.threshold))
    return best.threshold


def roc_points(metrics: list[ThresholdMetrics]) -> list[tuple[float, float]]:
    """ROC as (adversarial FAR, TPR/recall) points, ordered by ascending FAR."""
    pts = sorted({(round(m.adversarial_far, 6), round(m.recall, 6)) for m in metrics})
    return pts


def det_points(metrics: list[ThresholdMetrics]) -> list[tuple[float, float]]:
    """DET as (adversarial FAR, FRR) points, ordered by ascending FAR."""
    pts = sorted({(round(m.adversarial_far, 6), round(m.frr, 6)) for m in metrics})
    return pts


@dataclass
class Verdict:
    passed: bool
    recommended_threshold: float | None
    reasons: list[str] = field(default_factory=list)


def verdict(
    metrics: list[ThresholdMetrics],
    *,
    max_fp_per_hour: float,
    min_recall: float,
) -> Verdict:
    """Pass iff a threshold exists that meets both the FP/hr and recall constraints."""
    threshold = select_threshold(metrics, max_fp_per_hour=max_fp_per_hour, min_recall=min_recall)
    if threshold is None:
        return Verdict(False, None, [
            f"No threshold met recall >= {min_recall} with FP/hr <= {max_fp_per_hour}."
        ])
    return Verdict(True, threshold, [])
