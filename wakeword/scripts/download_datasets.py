#!/usr/bin/env python3
"""Stage 1 (milestone M1) — download commercially-licensed source datasets.

Materializes FOSS-compliant audio into ``datasets/<category>/<key>/`` so the M2
feature precompute has real LibriSpeech / MUSAN inputs to consume. Every dataset is
declared in ``lib/datasets.py`` (all on the commercial allowlist), so this script can
only ever fetch commercially-usable data.

For every dataset fetched it:
  - downloads + extracts the audio (16 kHz mono, no resampling needed),
  - writes a dataset-level ``datasets/licenses/<key>.license.json`` provenance sidecar,
  - records dataset version + archive sha256 into ``datasets/licenses/manifest.lock.json``
    (trust-on-first-use: first run records, later runs verify),
  - runs the commercial license gate at the end and fails closed on any violation.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py                       # default profile: m2-min
    python scripts/download_datasets.py --profile m2-full
    python scripts/download_datasets.py --dataset musan librispeech-dev-clean
    python scripts/download_datasets.py --only speech --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import datasets, licenses  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
DOWNLOAD_CACHE = DATASETS_ROOT / ".downloads"

log = logging.getLogger("download_datasets")

_CHUNK = 1 << 20  # 1 MiB


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_http(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` with HTTP-range resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.is_file() else 0

    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    try:
        response = urllib.request.urlopen(req)  # noqa: S310 (trusted FOSS mirrors)
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        if exc.code == 416 and existing:  # already complete
            log.info("    already downloaded: %s", dest.name)
            return
        raise

    mode = "ab" if (existing and response.status == 206) else "wb"
    if mode == "wb":
        existing = 0
    total = response.length or 0
    done = existing
    with response, dest.open(mode) as out:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / (existing + total)
                print(f"    {dest.name}: {done >> 20} MiB ({pct:4.1f}%)", end="\r", file=sys.stderr)
    print(file=sys.stderr)


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract a tar(.gz/.bz2) or zip archive into ``dest`` (idempotent)."""
    dest.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            _safe_extract_tar(tar, dest)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest)
    else:
        raise ValueError(f"Unsupported archive type: {archive.name}")


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    for member in tar.getmembers():
        if not _is_within(dest, dest / member.name):
            raise ValueError(f"Refusing path traversal in archive member: {member.name}")
    # ``filter="data"`` (Py 3.12+, backported) rejects unsafe members; fall back for older.
    try:
        tar.extractall(dest, filter="data")
    except TypeError:  # pragma: no cover - Python < 3.12 without backport
        tar.extractall(dest)  # noqa: S202 (members validated above)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    for name in zf.namelist():
        if not _is_within(dest, dest / name):
            raise ValueError(f"Refusing path traversal in archive member: {name}")
    zf.extractall(dest)  # noqa: S202 (members validated above)


def _verify_or_record(
    *, key: str, url: str, sha: str, lock: dict, skip_verify: bool
) -> None:
    recorded = lock.get("datasets", {}).get(key, {}).get("archives", {}).get(url)
    if recorded and not skip_verify and recorded != sha:
        raise ValueError(
            f"Checksum mismatch for {key} <{url}>:\n"
            f"  locked:   {recorded}\n  download: {sha}\n"
            "Refusing to proceed (reproducibility guard). Delete the lock entry to re-pin."
        )


def _fetch_http(spec: datasets.DatasetSpec, dest: Path, *, skip_verify: bool, lock: dict) -> dict[str, str]:
    archives: dict[str, str] = {}
    for url in spec.archive_urls:
        archive_path = DOWNLOAD_CACHE / spec.key / url.rsplit("/", 1)[-1]
        log.info("  downloading %s", url)
        _download_http(url, archive_path)
        sha = _sha256_stream(archive_path)
        _verify_or_record(key=spec.key, url=url, sha=sha, lock=lock, skip_verify=skip_verify)
        archives[url] = sha
        log.info("  extracting %s -> %s", archive_path.name, dest)
        _extract_archive(archive_path, dest)
    return archives


def _fetch_hf(spec: datasets.DatasetSpec, dest: Path) -> dict[str, str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"'{spec.key}' needs huggingface_hub. Install the training env "
            "(environment/requirements-training.txt) or run with --only for HTTP datasets."
        ) from exc
    log.info("  snapshot_download %s -> %s", spec.hf_repo, dest)
    snapshot_download(
        repo_id=spec.hf_repo,
        repo_type=spec.hf_repo_type,
        local_dir=str(dest),
    )
    # HF snapshots are content-addressed by the hub; record the repo id as the pin.
    return {f"hf://{spec.hf_repo}": spec.hf_repo}


def _fetch_one(spec: datasets.DatasetSpec, *, skip_verify: bool, lock: dict) -> None:
    dest = DATASETS_ROOT / spec.dest_subpath()
    log.info("[%s] %s (%s)", spec.key, spec.category, spec.license_id)

    if spec.source_kind == "http":
        archives = _fetch_http(spec, dest, skip_verify=skip_verify, lock=lock)
    elif spec.source_kind == "hf":
        archives = _fetch_hf(spec, dest)
    else:  # pragma: no cover - registry is validated in tests
        raise ValueError(f"Unknown source_kind '{spec.source_kind}' for {spec.key}")

    licenses.write_dataset_manifest(
        datasets_root=DATASETS_ROOT,
        key=spec.key,
        category=spec.category,
        dataset=spec.key,
        source_url=spec.source_url,
        license_id=spec.license_id,
        commercial_use=spec.commercial_use,
        attribution=spec.attribution,
        dataset_version=spec.dataset_version,
        notes=spec.notes,
    )
    licenses.update_lock(DATASETS_ROOT, key=spec.key, version=spec.dataset_version, archives=archives)
    log.info("  done: %s", dest)


def _print_listing() -> None:
    print("Available datasets (all commercial-allowlisted):\n")
    for spec in datasets.REGISTRY.values():
        print(f"  {spec.key:32s} {spec.category:16s} {spec.license_id}")
        print(f"  {'':32s} {spec.notes}")
    print("\nProfiles:")
    for name, keys in datasets.PROFILES.items():
        print(f"  {name:12s} {', '.join(keys)}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List registry + profiles and exit.")
    parser.add_argument("--profile", choices=sorted(datasets.PROFILES), help="Named dataset group (default: m2-min).")
    parser.add_argument("--dataset", nargs="*", help="Explicit dataset key(s) from the registry.")
    parser.add_argument(
        "--only",
        choices=list(datasets.CATEGORIES),
        help="Restrict to a single category.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched, download nothing.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip checksum verification against the lock.")
    args = parser.parse_args(argv)

    if args.list:
        _print_listing()
        return 0

    try:
        specs = datasets.resolve_selection(
            profile=args.profile, datasets=args.dataset, only_category=args.only
        )
    except KeyError as exc:
        parser.error(str(exc))

    if not specs:
        parser.error("Nothing selected. Use --list to see datasets/profiles.")

    log.info("Selected %d dataset(s): %s", len(specs), ", ".join(s.key for s in specs))
    if args.dry_run:
        for spec in specs:
            target = DATASETS_ROOT / spec.dest_subpath()
            src = spec.archive_urls or (f"hf://{spec.hf_repo}",)
            log.info("  would fetch %s -> %s", " ; ".join(src), target)
        return 0

    lock = licenses.read_lock(DATASETS_ROOT)
    for spec in specs:
        _fetch_one(spec, skip_verify=args.skip_verify, lock=lock)
        lock = licenses.read_lock(DATASETS_ROOT)  # refresh after write

    log.info("\nRunning commercial license gate ...")
    result = licenses.run_gate(DATASETS_ROOT, require_commercial=True)
    for warning in result.warnings:
        log.warning("WARN: %s", warning)
    if not result.ok:
        for err in result.errors:
            log.error("FAIL: %s", err)
        log.error("License gate FAILED — see errors above.")
        return 1
    log.info("License gate OK (%d manifest(s) checked).", result.checked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
