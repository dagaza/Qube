# Smoke-test the PyInstaller dist EXE (must stay alive for 10 seconds).
# CUDA builds: bundle layout only on CI (no NVIDIA driver) — mirrors smoke_linux_dist.sh.
$ErrorActionPreference = "Stop"

$exe = (Resolve-Path (Join-Path $PSScriptRoot "..\..\dist\Qube\Qube.exe")).Path
$distDir = Split-Path -Parent $exe
$variantMarker = Join-Path $distDir ".qube-windows-variant"
$variant = if (Test-Path $variantMarker) { (Get-Content $variantMarker -Raw).Trim() } else { "cpu" }

if ($variant -eq "cuda") {
    & "$PSScriptRoot/verify_windows_cuda_bundle.ps1" -Dist $distDir
    exit 0
}

. "$PSScriptRoot/smoke_launch_env.ps1"

$launchArgs = Get-QubeSmokeLaunchArgumentList
$state = Enter-QubeSmokeLaunchEnvironment
$proc = $null
try {
    $proc = Start-Process -FilePath $exe -ArgumentList $launchArgs -PassThru
    Start-Sleep -Seconds 10
    if ($proc.HasExited) {
        throw "App crashed on launch (exit code: $($proc.ExitCode))"
    }
    Write-Host "Smoke test passed — dist EXE alive after 10 s"
}
finally {
    Stop-QubeProcessIfRunning -Process $proc
    Exit-QubeSmokeLaunchEnvironment -State $state
}
