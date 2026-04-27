from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Optional


_DEFAULT_ALPHA_QUALITY = 0.10
_DEFAULT_ALPHA_LATENCY = 0.10
_RETRY_PENALTY = 0.02


@dataclass
class ModelPerformanceRecord:
    model_name: str
    total_requests: int = 0
    successful_outputs: int = 0
    avg_response_quality: float = 0.5
    avg_latency: float = 0.0
    structural_failure_rate: float = 0.0


@dataclass
class ModelPerformanceSnapshot:
    model_name: str
    confidence: float
    notes: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _looks_structurally_failed(validation_result: Any) -> bool:
    if validation_result is None:
        return False
    sev = str(getattr(validation_result, "severity", "") or "").lower()
    if sev == "high":
        return True
    is_valid = getattr(validation_result, "is_valid", None)
    if is_valid is False:
        return True
    issues = list(getattr(validation_result, "issues", []) or [])
    return bool(issues)


def _is_success(validation_result: Any, retry_used: bool) -> bool:
    if _looks_structurally_failed(validation_result):
        return False
    if retry_used:
        # Retry implies structure was not right on first pass; treat as partial miss.
        return False
    return True


class ModelPerformanceStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or os.path.join(
            os.path.expanduser("~"),
            ".qube",
            "model_performance_store.json",
        )
        self._records: dict[str, ModelPerformanceRecord] = {}
        self._lock = threading.Lock()
        self.load()

    @property
    def path(self) -> str:
        return self._path

    def load(self) -> None:
        with self._lock:
            try:
                if not os.path.isfile(self._path):
                    self._records = {}
                    return
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                self._records = {}
                return

            out: dict[str, ModelPerformanceRecord] = {}
            rows = raw.get("models", []) if isinstance(raw, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("model_name") or "").strip()
                if not name:
                    continue
                try:
                    rec = ModelPerformanceRecord(
                        model_name=name,
                        total_requests=max(0, int(row.get("total_requests", 0))),
                        successful_outputs=max(0, int(row.get("successful_outputs", 0))),
                        avg_response_quality=_clamp01(float(row.get("avg_response_quality", 0.5))),
                        avg_latency=max(0.0, float(row.get("avg_latency", 0.0))),
                        structural_failure_rate=_clamp01(float(row.get("structural_failure_rate", 0.0))),
                    )
                except Exception:
                    continue
                out[name] = rec
            self._records = out

    def save(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            payload = {
                "models": [asdict(rec) for rec in self._records.values()],
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, sort_keys=True, indent=2)

    def get(self, model_name: str) -> Optional[ModelPerformanceRecord]:
        key = str(model_name or "").strip()
        if not key:
            return None
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return None
            return ModelPerformanceRecord(**asdict(rec))

    def list_all(self) -> list[ModelPerformanceRecord]:
        with self._lock:
            return [ModelPerformanceRecord(**asdict(v)) for v in self._records.values()]

    def update(self, model_name: str, metrics: dict[str, Any]) -> Optional[ModelPerformanceRecord]:
        key = str(model_name or "").strip()
        if not key:
            return None
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                rec = ModelPerformanceRecord(model_name=key)
                self._records[key] = rec

            rec.total_requests += 1
            validation_result = metrics.get("validation_result")
            retry_used = bool(metrics.get("retry_used", False))
            quality_score = metrics.get("quality_score")
            latency = float(metrics.get("latency", 0.0) or 0.0)

            success = _is_success(validation_result, retry_used)
            if success:
                rec.successful_outputs += 1

            failures = max(0, rec.total_requests - rec.successful_outputs)
            rec.structural_failure_rate = _clamp01(failures / max(1, rec.total_requests))

            if quality_score is not None:
                try:
                    q = _clamp01(float(quality_score))
                    rec.avg_response_quality = (
                        (1.0 - _DEFAULT_ALPHA_QUALITY) * rec.avg_response_quality
                        + _DEFAULT_ALPHA_QUALITY * q
                    )
                except (TypeError, ValueError):
                    pass

            if retry_used:
                rec.avg_response_quality = _clamp01(rec.avg_response_quality - _RETRY_PENALTY)

            # Rolling average by request count for stability and explainability.
            if rec.total_requests <= 1:
                rec.avg_latency = max(0.0, latency)
            else:
                prev_n = rec.total_requests - 1
                rec.avg_latency = ((rec.avg_latency * prev_n) + max(0.0, latency)) / rec.total_requests

            saved = ModelPerformanceRecord(**asdict(rec))
        self.save()
        return saved

    def update_model_metrics(
        self,
        model_name: str,
        validation_result: Any,
        quality_score: Optional[float],
        latency: float,
        retry_used: bool,
    ) -> Optional[ModelPerformanceRecord]:
        return self.update(
            model_name=model_name,
            metrics={
                "validation_result": validation_result,
                "quality_score": quality_score,
                "latency": latency,
                "retry_used": retry_used,
            },
        )

    def snapshot(self, model_name: str) -> Optional[ModelPerformanceSnapshot]:
        rec = self.get(model_name)
        if rec is None:
            return None
        notes: list[str] = []
        if rec.structural_failure_rate > 0.6:
            notes.append("unreliable: structural_failure_rate>0.6")
        if rec.avg_response_quality < 0.40:
            notes.append("low_quality_trend")
        if rec.total_requests < 3:
            notes.append("low_sample_count")
        confidence = _clamp01(min(1.0, rec.total_requests / 20.0))
        return ModelPerformanceSnapshot(
            model_name=rec.model_name,
            confidence=confidence,
            notes=notes,
        )
