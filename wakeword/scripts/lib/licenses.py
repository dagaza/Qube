"""License policy + manifest loading for the wake word pipeline.

Standard-library only on purpose: the fail-closed license gate must run anywhere
(CI, a bare checkout) without the pinned training environment.

See ``docs/licensing.md`` for the policy rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_SUFFIX = ".license.json"

# SPDX-style ids that permit commercial use AND redistribution.
COMMERCIAL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-3.0",
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Public-Domain",
    }
)

# Allowed for configs/docs only — NOT for data that could make the model a
# share-alike derivative. Flagged (warned) rather than silently accepted.
COMMERCIAL_REVIEW: frozenset[str] = frozenset({"CC-BY-SA-4.0", "CC-BY-SA-3.0"})

REQUIRED_FIELDS: tuple[str, ...] = (
    "asset",
    "dataset",
    "source_url",
    "license",
    "commercial_use",
    "retrieved",
)


@dataclass
class Manifest:
    """A single ``<asset>.license.json`` sidecar."""

    path: Path
    data: dict

    @property
    def license_id(self) -> str:
        return str(self.data.get("license", "")).strip()

    @property
    def commercial_use(self) -> bool:
        return bool(self.data.get("commercial_use", False))

    @property
    def dataset(self) -> str:
        return str(self.data.get("dataset", "")).strip()


@dataclass
class GateResult:
    """Outcome of a license-gate run."""

    checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def is_example_manifest(path: Path) -> bool:
    """``*.example.license.json`` files are templates, not real provenance."""
    return path.name.endswith(".example" + MANIFEST_SUFFIX)


def iter_manifest_paths(datasets_root: Path) -> list[Path]:
    """All real (non-example) license manifests under ``datasets_root``."""
    return sorted(
        p
        for p in datasets_root.rglob("*" + MANIFEST_SUFFIX)
        if not is_example_manifest(p)
    )


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(path=path, data=data)


def _validate_fields(manifest: Manifest) -> list[str]:
    missing = [f for f in REQUIRED_FIELDS if f not in manifest.data]
    if missing:
        rel = manifest.path.name
        return [f"{rel}: missing required field(s): {', '.join(missing)}"]
    return []


def evaluate_manifest(manifest: Manifest, *, require_commercial: bool) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for one manifest."""
    errors = _validate_fields(manifest)
    warnings: list[str] = []
    if errors:
        return errors, warnings

    lic = manifest.license_id
    rel = manifest.path.name

    if require_commercial:
        if lic in COMMERCIAL_REVIEW:
            warnings.append(
                f"{rel}: license '{lic}' is share-alike — allowed for configs/docs "
                f"only, review before using for training data."
            )
        elif lic not in COMMERCIAL_ALLOWLIST:
            errors.append(
                f"{rel}: license '{lic}' is NOT on the commercial allowlist "
                f"(dataset '{manifest.dataset}')."
            )
        if not manifest.commercial_use:
            errors.append(
                f"{rel}: commercial_use=false (dataset '{manifest.dataset}')."
            )

    return errors, warnings


def run_gate(datasets_root: Path, *, require_commercial: bool) -> GateResult:
    """Walk every manifest and apply the policy. Fail-closed by design."""
    result = GateResult()
    paths = iter_manifest_paths(datasets_root)

    for path in paths:
        result.checked += 1
        try:
            manifest = load_manifest(path)
        except (json.JSONDecodeError, OSError) as exc:
            result.errors.append(f"{path.name}: unreadable manifest ({exc}).")
            continue
        errors, warnings = evaluate_manifest(manifest, require_commercial=require_commercial)
        result.errors.extend(errors)
        result.warnings.extend(warnings)

    return result
