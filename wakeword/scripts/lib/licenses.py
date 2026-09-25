"""License policy + manifest loading for the wake word pipeline.

Standard-library only on purpose: the fail-closed license gate must run anywhere
(CI, a bare checkout) without the pinned training environment.

See ``docs/licensing.md`` for the policy rationale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
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


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 so we never load large feature files fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    asset_path: Path,
    *,
    datasets_root: Path,
    dataset: str,
    source_url: str,
    license_id: str,
    commercial_use: bool,
    attribution: str = "",
    dataset_version: str = "",
    notes: str = "",
    compute_sha256: bool = True,
) -> Path:
    """Write a ``<asset>.license.json`` provenance sidecar next to ``asset_path``.

    Used by generation stages (e.g. precompute_features) so derived artifacts carry
    the same machine-checkable provenance as downloaded data and stay green under the
    commercial license gate.
    """
    try:
        rel_asset = str(asset_path.relative_to(datasets_root))
    except ValueError:
        rel_asset = asset_path.name

    payload = {
        "asset": rel_asset,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "source_url": source_url,
        "license": license_id,
        "commercial_use": commercial_use,
        "attribution": attribution,
        "retrieved": date.today().isoformat(),
        "notes": notes,
    }
    if compute_sha256 and asset_path.is_file():
        payload["sha256"] = sha256_file(asset_path)

    manifest_path = asset_path.with_name(asset_path.name + MANIFEST_SUFFIX)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_dataset_manifest(
    *,
    datasets_root: Path,
    key: str,
    category: str,
    dataset: str,
    source_url: str,
    license_id: str,
    commercial_use: bool,
    attribution: str = "",
    dataset_version: str = "",
    notes: str = "",
) -> Path:
    """Write a dataset-level provenance manifest under ``datasets/licenses/``.

    A single manifest covers a whole extracted dataset tree (datasets can contain
    tens of thousands of files), placed in the git-tracked ``licenses/`` dir so
    provenance survives even though the audio itself is gitignored.
    """
    licenses_dir = datasets_root / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset": f"{category}/{key}",
        "dataset": dataset,
        "dataset_version": dataset_version,
        "source_url": source_url,
        "license": license_id,
        "commercial_use": commercial_use,
        "attribution": attribution,
        "retrieved": date.today().isoformat(),
        "notes": notes,
    }
    manifest_path = licenses_dir / f"{key}{MANIFEST_SUFFIX}"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def update_lock(
    datasets_root: Path,
    *,
    key: str,
    version: str,
    archives: dict[str, str],
) -> Path:
    """Record dataset version + per-archive sha256 into ``manifest.lock.json``.

    This is the reproducibility anchor: a future clone either reproduces the same
    inputs (checksums match) or fails loudly. Uses trust-on-first-use — the first
    download records the checksum, later runs verify against it.
    """
    lock_path = datasets_root / "licenses" / "manifest.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    else:
        lock = {"datasets": {}}
    lock.setdefault("datasets", {})[key] = {
        "version": version,
        "archives": archives,
        "retrieved": date.today().isoformat(),
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock_path


def read_lock(datasets_root: Path) -> dict:
    lock_path = datasets_root / "licenses" / "manifest.lock.json"
    if not lock_path.is_file():
        return {"datasets": {}}
    return json.loads(lock_path.read_text(encoding="utf-8"))


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
