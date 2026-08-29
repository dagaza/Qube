# Smoke-test the PyInstaller dist EXE (must stay alive for 10 seconds).
$ErrorActionPreference = "Stop"
. "$PSScriptRoot/smoke_launch_env.ps1"

$exe = (Resolve-Path (Join-Path $PSScriptRoot "..\..\dist\Qube\Qube.exe")).Path
$distDir = Split-Path -Parent $exe
$variantMarker = Join-Path $distDir ".qube-windows-variant"
$variant = if (Test-Path $variantMarker) { (Get-Content $variantMarker -Raw).Trim() } else { "cpu" }

$launchArgs = Get-QubeSmokeLaunchArgumentList
if ($variant -eq "cuda") {
    $env:QUBE_WINGET_VALIDATION = "1"
    $launchArgs += "--winget-validation"
}

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
    Remove-Item Env:QUBE_WINGET_VALIDATION -ErrorAction SilentlyContinue
}
