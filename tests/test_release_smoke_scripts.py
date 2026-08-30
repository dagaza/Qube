"""Release smoke script consistency checks."""

from __future__ import annotations

from pathlib import Path


def test_installed_and_upgrade_smoke_use_shared_launch_env():
    root = Path(__file__).resolve().parent.parent / "scripts" / "release"
    helper = root / "smoke_launch_env.ps1"
    assert helper.is_file()

    for name in ("smoke_installed.ps1", "smoke_upgrade.ps1", "smoke_dist.ps1"):
        text = (root / name).read_text(encoding="utf-8")
        assert "smoke_launch_env.ps1" in text
        assert "--mock-bootstrap-download" in text or "Get-QubeSmokeLaunchArgumentList" in text

    installed = (root / "smoke_installed.ps1").read_text(encoding="utf-8")
    assert "Enter-QubeSmokeLaunchEnvironment" in installed
    assert "Exit-QubeSmokeLaunchEnvironment" in installed

    launch_env = (root / "smoke_launch_env.ps1").read_text(encoding="utf-8")
    assert '$env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD = "1"' not in launch_env
    assert "Remove-Item Env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD" not in launch_env


def test_smoke_dist_cuda_skips_runtime_launch():
    root = Path(__file__).resolve().parent.parent / "scripts" / "release"
    text = (root / "smoke_dist.ps1").read_text(encoding="utf-8")
    assert "verify_windows_cuda_bundle.ps1" in text
    assert '$variant -eq "cuda"' in text
    assert "--winget-validation" not in text
