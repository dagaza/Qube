# Silent install, launch CUDA build with WinGet validation guard, verify no llama_cpp import.
param(
    [string]$SetupPath = ""
)

$ErrorActionPreference = "Stop"

if ($SetupPath) {
    $setup = Get-Item $SetupPath
} else {
    $setup = Get-ChildItem (Join-Path $PSScriptRoot "..\..\installer\output\Qube-*-cuda-Setup.exe") |
        Sort-Object { [version]($_.BaseName -replace '^Qube-','' -replace '-cuda$','') } -Descending |
        Select-Object -First 1
}

if (-not $setup) {
    throw "No Qube-*-cuda-Setup.exe found under installer/output"
}

$installDir = Join-Path $env:LOCALAPPDATA "Programs\Qube"
$installedExe = Join-Path $installDir "Qube.exe"
$uninstaller = Join-Path $installDir "unins000.exe"
$userData = Join-Path $env:LOCALAPPDATA "Qube"
$settingsDir = Join-Path $env:USERPROFILE ".qube"
$settingsPath = Join-Path $settingsDir "settings.json"
$cognitionDir = Join-Path $userData "models\cognition"
$resultPath = Join-Path $userData ".winget-validation-smoke.json"
$dummyGguf = Join-Path $cognitionDir "Qwen3-1.7B-Q6_K.gguf"

Write-Host "Installing $($setup.Name) silently..."
Start-Process -Wait -FilePath $setup.FullName `
    -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART"

if (-not (Test-Path $installedExe)) {
    throw "Silent install failed — $installedExe not found"
}
Write-Host "Silent install verified at $installedExe"

$installMarker = Join-Path $installDir ".qube-install-ts"
if (-not (Test-Path $installMarker)) {
    throw "Missing CUDA install grace marker: $installMarker"
}

# Simulate a validation VM with completed bootstrap and a sidecar model on disk.
New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
New-Item -ItemType Directory -Path $cognitionDir -Force | Out-Null
Set-Content -Path $dummyGguf -Value "dummy" -Encoding ascii
@'
{
  "qube.bootstrap.completed": true,
  "qube.sidecar.enabled": true
}
'@ | Set-Content -Path $settingsPath -Encoding utf8

Remove-Item $resultPath -Force -ErrorAction SilentlyContinue

Write-Host "Launching installed CUDA EXE with WinGet validation guard..."
$env:QUBE_WINGET_VALIDATION = "1"
$proc = Start-Process -FilePath $installedExe `
    -ArgumentList "--mock-bootstrap-download" `
    -PassThru

$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        throw "Installed CUDA app exited early (exit code: $($proc.ExitCode))"
    }
    if (Test-Path $resultPath) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not (Test-Path $resultPath)) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for validation smoke result at $resultPath"
}

$result = Get-Content $resultPath -Raw | ConvertFrom-Json
if (-not $result.ok -or $result.llama_import_attempted) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "WinGet validation guard failed: llama_cpp import was attempted"
}

Write-Host "CUDA WinGet validation smoke passed (pid $($proc.Id), no llama_cpp import)"

if (-not (Test-Path $uninstaller)) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "Uninstaller not found at $uninstaller"
}

Write-Host "Uninstalling while Qube.exe is still running..."
Start-Process -Wait -FilePath $uninstaller `
    -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART"

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline) {
    Get-Process -Name Qube -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $installedExe)) {
        break
    }
    Start-Sleep -Seconds 1
}

if (Test-Path $installedExe) {
    throw "Uninstall failed — $installedExe still exists"
}
Write-Host "CUDA validation smoke + uninstall verified"
