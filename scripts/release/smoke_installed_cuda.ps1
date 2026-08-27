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
$bootStatePath = Join-Path $userData ".winget-validation-boot-state.json"
$dummyGguf = Join-Path $cognitionDir "Qwen3-1.7B-Q6_K.gguf"

function Format-SmokeFailureMessage {
    param(
        [string]$ResultPath,
        [string]$BootStatePath
    )
    $parts = @("WinGet validation smoke did not succeed.")
    if (Test-Path $ResultPath) {
        try {
            $result = Get-Content $ResultPath -Raw | ConvertFrom-Json
            if ($result.stage) { $parts += "stage=$($result.stage)" }
            if ($result.error) { $parts += "error=$($result.error)" }
            if ($null -ne $result.ok) { $parts += "ok=$($result.ok)" }
        } catch {
            $parts += "smoke result present but unreadable"
        }
    } else {
        $parts += "no smoke result at $ResultPath"
    }
    if (Test-Path $BootStatePath) {
        try {
            $state = Get-Content $BootStatePath -Raw | ConvertFrom-Json
            if ($state.state) { $parts += "last_boot_state=$($state.state)" }
            if ($null -ne $state.phase) { $parts += "phase=$($state.phase)" }
        } catch {
            $parts += "boot state present but unreadable"
        }
    }
    return ($parts -join "; ")
}

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
'@ | Set-Content -Path $settingsPath -Encoding utf8NoBOM

Remove-Item $resultPath -Force -ErrorAction SilentlyContinue
Remove-Item $bootStatePath -Force -ErrorAction SilentlyContinue

Write-Host "Launching installed CUDA EXE with WinGet validation guard..."
$env:QUBE_WINGET_VALIDATION = "1"
$proc = Start-Process -FilePath $installedExe `
    -ArgumentList "--mock-bootstrap-download", "--winget-validation" `
    -PassThru

$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        $msg = Format-SmokeFailureMessage -ResultPath $resultPath -BootStatePath $bootStatePath
        throw "Installed CUDA app exited early (exit code: $($proc.ExitCode)). $msg"
    }
    if (Test-Path $resultPath) {
        $result = Get-Content $resultPath -Raw | ConvertFrom-Json
        if (-not $result.ok) {
            $msg = Format-SmokeFailureMessage -ResultPath $resultPath -BootStatePath $bootStatePath
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            throw $msg
        }
        break
    }
    Start-Sleep -Seconds 1
}

if (-not (Test-Path $resultPath)) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    $msg = Format-SmokeFailureMessage -ResultPath $resultPath -BootStatePath $bootStatePath
    throw "Timed out waiting for validation smoke result at $resultPath. $msg"
}

$result = Get-Content $resultPath -Raw | ConvertFrom-Json
if (-not $result.ok -or $result.llama_import_attempted) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    $msg = Format-SmokeFailureMessage -ResultPath $resultPath -BootStatePath $bootStatePath
    throw "WinGet validation guard failed: llama_cpp import was attempted. $msg"
}

Write-Host "CUDA WinGet validation smoke passed (pid $($proc.Id), stage $($result.stage), no llama_cpp import)"

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
