"""Tests for scripts/linux/recompress_deb_data.sh."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "linux" / "recompress_deb_data.sh"


@pytest.mark.skipif(shutil.which("ar") is None, reason="ar not available")
def test_recompress_deb_data_repacks_minimal_deb(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    payload = work / "payload"
    payload.mkdir()
    (payload / "opt" / "qube").mkdir(parents=True)
    (payload / "opt" / "qube" / "marker.txt").write_text("ok\n", encoding="utf-8")

    data_tar = work / "data.tar.xz"
    with tarfile.open(data_tar, "w:xz") as tar:
        tar.add(payload, arcname=".")

    control_tar = work / "control.tar.xz"
    control_dir = work / "control"
    control_dir.mkdir()
    (control_dir / "control").write_text("Package: qube-cuda\nVersion: 0.0.0\n", encoding="utf-8")
    with tarfile.open(control_tar, "w:xz") as tar:
        tar.add(control_dir / "control", arcname="control")

    (work / "debian-binary").write_text("2.0\n", encoding="utf-8")

    deb = tmp_path / "sample.deb"
    subprocess.run(
        ["ar", "cr", str(deb), "debian-binary", control_tar.name, data_tar.name],
        cwd=work,
        check=True,
    )

    before = deb.stat().st_size
    subprocess.run(["bash", str(_SCRIPT), str(deb)], check=True)
    after = deb.stat().st_size

    assert after > 0
    assert deb.exists()

    extract = tmp_path / "extract"
    extract.mkdir()
    subprocess.run(["ar", "x", str(deb)], cwd=extract, check=True)
    assert (extract / "debian-binary").is_file()
    assert (extract / "data.tar.xz").is_file()
    assert before >= 0  # exercise ran end-to-end without corrupting the archive
