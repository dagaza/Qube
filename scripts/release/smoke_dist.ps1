# Smoke-test the PyInstaller dist EXE (must stay alive for 10 seconds).
$ErrorActionPreference = "Stop"
$exe = (Resolve-Path (Join-Path $PSScriptRoot "..\..\dist\Qube\Qube.exe")).Path
$distDir = Split-Path -Parent $exe

$fakeAppData = Join-Path $env:TEMP ("qube-smoke-" + [guid]::NewGuid().ToString())
$settingsDir = Join-Path $fakeAppData "Qube"
New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
@'
{
  "qube.bootstrap.completed": true
}
'@ | Set-Content -Path (Join-Path $settingsDir "settings.json") -Encoding utf8

$previousAppData = $env:LOCALAPPDATA
$env:LOCALAPPDATA = $fakeAppData
$variantMarker = Join-Path $distDir ".qube-windows-variant"
$variant = if (Test-Path $variantMarker) { (Get-Content $variantMarker -Raw).Trim() } else { "cpu" }
if ($variant -eq "cuda") {
    $env:QUBE_WINGET_VALIDATION = "1"
}
try {
    $launchArgs = @("--mock-bootstrap-download")
    if ($variant -eq "cuda") {
        $launchArgs += "--winget-validation"
    }
    $proc = Start-Process -FilePath $exe -ArgumentList $launchArgs -PassThru
    Start-Sleep -Seconds 10
    if ($proc.HasExited) {
        throw "App crashed on launch (exit code: $($proc.ExitCode))"
    }
    Write-Host "Smoke test passed — dist EXE alive after 10 s"
    Stop-Process -Id $proc.Id -Force
}
finally {
    $env:LOCALAPPDATA = $previousAppData
    Remove-Item -Recurse -Force $fakeAppData -ErrorAction SilentlyContinue
}
