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
    assert "Install-QubeSilentSetup" in installed
    uninstall_idx = installed.index("Invoke-QubeSilentUninstallWhileRunning")
    exit_idx = installed.index("Exit-QubeSmokeLaunchEnvironment")
    assert uninstall_idx < exit_idx

    launch_env = (root / "smoke_launch_env.ps1").read_text(encoding="utf-8")
    assert "Invoke-QubeSilentUninstallWhileRunning" in launch_env
    assert "Install-QubeSilentSetup" in launch_env
    assert "Wait-QubeProcessesExited" in launch_env
    assert '"/IM", "Qube.exe", "/T"' in launch_env
    assert "WaitForExit" in launch_env
    assert '$env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD = "1"' not in launch_env
    assert "Remove-Item Env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD" not in launch_env

    upgrade = (root / "smoke_upgrade.ps1").read_text(encoding="utf-8")
    assert "Install-QubeSilentSetup" in upgrade
    cuda = (root / "smoke_installed_cuda.ps1").read_text(encoding="utf-8")
    assert "Install-QubeSilentSetup" in cuda


def test_smoke_dist_cuda_skips_runtime_launch():
    root = Path(__file__).resolve().parent.parent / "scripts" / "release"
    text = (root / "smoke_dist.ps1").read_text(encoding="utf-8")
    assert "verify_windows_cuda_bundle.ps1" in text
    assert '$variant -eq "cuda"' in text
    assert "--winget-validation" not in text
