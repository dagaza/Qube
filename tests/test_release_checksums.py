"""Tests for scripts/release/write_windows_installer_checksums.py."""

from __future__ import annotations

from pathlib import Path

from scripts.release.write_windows_installer_checksums import installer_names, write_checksums


def test_write_checksums_for_windows_installers(tmp_path: Path) -> None:
    version = "9.9.9"
    blobs = {
        f"Qube-{version}-Setup.exe": b"cpu-bytes",
        f"Qube-{version}-vulkan-Setup.exe": b"vulkan-bytes",
        f"Qube-{version}-cuda-Setup.exe": b"cuda-bytes",
    }
    for name, payload in blobs.items():
        (tmp_path / name).write_bytes(payload)

    hashes = write_checksums(version, tmp_path)

    assert set(hashes) == {"cpu", "vulkan", "cuda"}
    sums_text = (tmp_path / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for filename in blobs:
        assert filename in sums_text
    assert sums_text.count("\n") == 3


def test_installer_names_follow_release_convention() -> None:
    names = dict(installer_names("1.3.51"))
    assert names["cpu"] == "Qube-1.3.51-Setup.exe"
    assert names["vulkan"] == "Qube-1.3.51-vulkan-Setup.exe"
    assert names["cuda"] == "Qube-1.3.51-cuda-Setup.exe"
