# Smoke-test the PyInstaller dist EXE (must stay alive for 10 seconds).
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "..\..\dist\Qube\Qube.exe" | Resolve-Path

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
try {
    $proc = Start-Process -FilePath $exe -ArgumentList "--mock-bootstrap-download" -PassThru
    Start-Sleep -Seconds 10
    if ($proc.HasExited) {
        throw "App crashed on launch (exit code: $($proc.ExitCode))"
    }
    Write-Host "Smoke test passed — dist EXE alive after 10 s"
    Stop-Process -Id $proc.Id -Force
}
finally {
    $env:LOCALAPDATA = $previousAppData
    Remove-Item -Recurse -Force $fakeAppData -ErrorAction SilentlyContinue
}
