"""Optional smoke test against a packaged Qube executable."""

from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.packaging
def test_frozen_exe_launches_when_configured():
    exe = os.environ.get("QUBE_FROZEN_EXE")
    if not exe:
        pytest.skip("Set QUBE_FROZEN_EXE to run packaging smoke tests")
    path = os.path.abspath(exe)
    assert os.path.isfile(path), path
    proc = subprocess.Popen(
        [path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)
            return
        pytest.fail(f"Packaged executable exited early with code {proc.returncode}")
    finally:
        if proc.poll() is None:
            proc.kill()
