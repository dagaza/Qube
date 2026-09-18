"""model_card.json builder — the auditable provenance record for a trained model.

Every released model ships one of these so the MIT-compatibility claim is verifiable years
later: it records the exact config, dataset versions + checksums, training params + seed,
the pinned openWakeWord commit, hardware, duration, eval metrics, and the license-gate
result. Pure/stdlib-only so it is fully unit-testable and runs in the license-gate env.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"

# Runtime contract the exported model must satisfy (openWakeWord 0.4.0 / Qube runtime).
INPUT_SHAPE = [16, 96]
OUTPUT_SHAPE = [1]


def canonical_config_hash(config: dict) -> str:
    """Stable sha256 of a config dict (order-independent), for reproducibility."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_model_card(
    config: dict,
    *,
    version: str,
    git_commit: str,
    training_datasets: list[dict],
    metrics: dict[str, Any] | None = None,
    hardware: str = "",
    duration_seconds: float | None = None,
    license_audit: str = "passed",
) -> dict:
    """Assemble a model_card dict from a config + run metadata.

    ``training_datasets`` is a list of ``{name, version, license, sha256?}`` entries
    (typically derived from ``manifest.lock.json`` + generated-feature manifests).
    ``license_audit`` is the gate outcome: ``passed`` / ``failed`` / ``skipped``.
    """
    wake = config.get("wakeword", {})
    training = config.get("training", {})
    provenance = config.get("provenance", {})
    tier = str(provenance.get("tier", "personal"))

    return {
        "schema_version": SCHEMA_VERSION,
        "id": str(wake.get("id", "")),
        "display_name": str(wake.get("display_name", "")),
        "phrase": str(wake.get("phrase", "")),
        "version": version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_date": date.today().isoformat(),
        "runtime": {
            "framework": "openWakeWord",
            "openwakeword_version": "0.4.0",
            "oww_commit": str(provenance.get("oww_commit", "")),
            "sample_rate": int(config.get("data", {}).get("sample_rate", 16000)),
            "input_shape": INPUT_SHAPE,
            "output_shape": OUTPUT_SHAPE,
        },
        "training": {
            "examples": int(training.get("examples", 0)),
            "steps": int(training.get("steps", 0)),
            "false_penalty": int(training.get("false_penalty", 0)),
            "layer_dim": int(training.get("layer_dim", 32)),
            "augmentation_rounds": int(training.get("augmentation_rounds", 0)),
            "seed": int(training.get("seed", 0)),
            "hardware": hardware,
            "duration_seconds": duration_seconds,
        },
        "provenance": {
            "git_commit": git_commit,
            "config_hash": canonical_config_hash(config),
            "license_tier": tier,
            "license_audit": license_audit,
        },
        "training_datasets": training_datasets,
        "metrics": metrics or {},
        "license": {
            "tier": tier,
            "commercial_use": tier == "commercial" and license_audit == "passed",
            "note": (
                "All training inputs on the commercial allowlist; distributable under "
                "terms compatible with Qube's MIT license."
                if tier == "commercial" and license_audit == "passed"
                else "Personal-use model — trained data is not fully commercial-cleared; "
                "must not be promoted to 'recommended' in Qube."
            ),
        },
    }


def write_model_card(card: dict, path: "Any") -> "Any":
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    return out
