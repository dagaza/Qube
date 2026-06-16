"""Tests for the dataset registry + downloader plumbing.

Network downloads are intentionally NOT exercised here — only the pure logic
(registry integrity, selection resolution, archive extraction/safety, dataset
manifest provenance, and the reproducibility lock) which can run in CI without
fetching gigabytes of audio.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import download_datasets as dl  # noqa: E402
from lib import datasets, licenses  # noqa: E402


# --- Registry integrity ----------------------------------------------------------

def test_every_dataset_is_commercial_and_allowlisted() -> None:
    for spec in datasets.REGISTRY.values():
        assert spec.commercial_use, f"{spec.key} not flagged commercial_use"
        assert spec.license_id in licenses.COMMERCIAL_ALLOWLIST, (
            f"{spec.key} license {spec.license_id} not on commercial allowlist"
        )


def test_registry_specs_are_well_formed() -> None:
    for key, spec in datasets.REGISTRY.items():
        assert spec.key == key
        assert spec.category in datasets.CATEGORIES
        assert spec.source_kind in ("http", "hf")
        if spec.source_kind == "http":
            assert spec.archive_urls and not spec.hf_repo
        else:
            assert spec.hf_repo and not spec.archive_urls
        assert spec.dest_subpath() == f"{spec.category}/{spec.key}"


def test_profiles_reference_real_keys() -> None:
    for keys in datasets.PROFILES.values():
        for key in keys:
            assert key in datasets.REGISTRY


def test_m2_min_profile_covers_librispeech_and_musan() -> None:
    specs = datasets.resolve_selection(profile="m2-min", datasets=None, only_category=None)
    keys = {s.key for s in specs}
    assert "musan" in keys
    assert any(k.startswith("librispeech") for k in keys)


# --- Selection resolution --------------------------------------------------------

def test_default_selection_is_m2_min() -> None:
    specs = datasets.resolve_selection(profile=None, datasets=None, only_category=None)
    assert {s.key for s in specs} == set(datasets.PROFILES["m2-min"])


def test_explicit_dataset_selection_is_deduped_and_ordered() -> None:
    specs = datasets.resolve_selection(
        profile=None, datasets=["musan", "musan", "librispeech-dev-clean"], only_category=None
    )
    assert [s.key for s in specs] == ["musan", "librispeech-dev-clean"]


def test_only_category_without_keys_returns_all_in_category() -> None:
    specs = datasets.resolve_selection(profile=None, datasets=None, only_category="speech")
    assert specs
    assert all(s.category == "speech" for s in specs)


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        datasets.resolve_selection(profile=None, datasets=["nope"], only_category=None)


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        datasets.resolve_selection(profile="bogus", datasets=None, only_category=None)


# --- Archive extraction + safety -------------------------------------------------

def _make_tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def test_extract_tar_roundtrip(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    _make_tar(archive, {"corpus/1.flac": b"RIFFfake", "corpus/sub/2.flac": b"more"})
    dest = tmp_path / "out"
    dl._extract_archive(archive, dest)
    assert (dest / "corpus" / "1.flac").read_bytes() == b"RIFFfake"
    assert (dest / "corpus" / "sub" / "2.flac").read_bytes() == b"more"


def test_extract_zip_roundtrip(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("music/a.wav", b"wavdata")
    dest = tmp_path / "out"
    dl._extract_archive(archive, dest)
    assert (dest / "music" / "a.wav").read_bytes() == b"wavdata"


def test_tar_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    _make_tar(archive, {"../escape.flac": b"x"})
    with pytest.raises(ValueError, match="traversal"):
        dl._extract_archive(archive, tmp_path / "out")


# --- Provenance manifest + lock --------------------------------------------------

def test_dataset_manifest_passes_commercial_gate(tmp_path: Path) -> None:
    spec = datasets.REGISTRY["musan"]
    licenses.write_dataset_manifest(
        datasets_root=tmp_path,
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
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok
    assert result.checked == 1


def test_lock_records_and_reads_back(tmp_path: Path) -> None:
    licenses.update_lock(
        tmp_path, key="musan", version="v1.0", archives={"http://x/musan.tar.gz": "abc123"}
    )
    lock = licenses.read_lock(tmp_path)
    assert lock["datasets"]["musan"]["version"] == "v1.0"
    assert lock["datasets"]["musan"]["archives"]["http://x/musan.tar.gz"] == "abc123"


def test_lock_update_is_incremental(tmp_path: Path) -> None:
    licenses.update_lock(tmp_path, key="musan", version="v1", archives={"u1": "a"})
    licenses.update_lock(tmp_path, key="librispeech-dev-clean", version="dc", archives={"u2": "b"})
    lock = licenses.read_lock(tmp_path)
    assert set(lock["datasets"]) == {"musan", "librispeech-dev-clean"}


def test_verify_or_record_detects_mismatch() -> None:
    lock = {"datasets": {"musan": {"archives": {"u1": "expected"}}}}
    with pytest.raises(ValueError, match="mismatch"):
        dl._verify_or_record(key="musan", url="u1", sha="different", lock=lock, skip_verify=False)


def test_verify_or_record_allows_match() -> None:
    lock = {"datasets": {"musan": {"archives": {"u1": "same"}}}}
    dl._verify_or_record(key="musan", url="u1", sha="same", lock=lock, skip_verify=False)


def test_verify_or_record_skips_when_no_prior_record() -> None:
    dl._verify_or_record(key="musan", url="u1", sha="new", lock={"datasets": {}}, skip_verify=False)


# --- CLI smoke -------------------------------------------------------------------

def test_dry_run_downloads_nothing(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    rc = dl.main(["--dataset", "musan", "--dry-run"])
    assert rc == 0


def test_list_exits_clean() -> None:
    assert dl.main(["--list"]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
