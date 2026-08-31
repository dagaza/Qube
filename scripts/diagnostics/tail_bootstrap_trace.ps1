# Tail Qube bootstrap trace JSONL on Windows (run in a second PowerShell window).
param(
    [string]$TracePath = ""
)

$ErrorActionPreference = "Stop"

if (-not $TracePath) {
    $TracePath = Join-Path $env:LOCALAPPDATA "Qube\logs\bootstrap-trace.jsonl"
}

if (-not (Test-Path $TracePath)) {
    Write-Host "Waiting for trace file: $TracePath"
    while (-not (Test-Path $TracePath)) {
        Start-Sleep -Seconds 1
    }
}

Write-Host "Tailing $TracePath"
Get-Content $TracePath -Wait -Tail 20
