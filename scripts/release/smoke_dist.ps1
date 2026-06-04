# Smoke-test the PyInstaller dist EXE (must stay alive for 10 seconds).
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "..\..\dist\Qube\Qube.exe" | Resolve-Path
$proc = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds 10
if ($proc.HasExited) {
    throw "App crashed on launch (exit code: $($proc.ExitCode))"
}
Write-Host "Smoke test passed — dist EXE alive after 10 s"
Stop-Process -Id $proc.Id -Force
