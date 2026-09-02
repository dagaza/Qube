# Silent install, launch CUDA build with WinGet validation guard, verify no llama_cpp import.
param(
    [string]$SetupPath = ""
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot/smoke_launch_env.ps1"

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
$bootTracePath = Join-Path $userData ".winget-validation-boot-trace.jsonl"
$graceTracePath = Join-Path $userData ".winget-validation-grace.jsonl"
$dummyGguf = Join-Path $cognitionDir "Qwen3-1.7B-Q6_K.gguf"

function Clear-ValidationDiagnostics {
    Remove-Item $resultPath -Force -ErrorAction SilentlyContinue
    Remove-Item $bootStatePath -Force -ErrorAction SilentlyContinue
    Remove-Item $bootTracePath -Force -ErrorAction SilentlyContinue
    Remove-Item $graceTracePath -Force -ErrorAction SilentlyContinue
}

function Format-SmokeFailureMessage {
    param(
        [string]$ResultPath,
        [string]$BootStatePath,
        [string]$BootTracePath,
        [string]$GraceTracePath = ""
    )
    $parts = @("WinGet validation smoke did not succeed.")
    if (Test-Path $ResultPath) {
        try {
            $result = Get-Content $ResultPath -Raw | ConvertFrom-Json
            if ($result.mode) { $parts += "mode=$($result.mode)" }
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
            $parts += "boot_state=$(Get-Content $BootStatePath -Raw)"
        } catch {
            $parts += "boot state present but unreadable"
        }
    } else {
        $parts += "no boot state at $BootStatePath"
    }
    if (Test-Path $BootTracePath) {
        try {
            $parts += "boot_trace=$(Get-Content $BootTracePath -Raw)"
        } catch {
            $parts += "boot trace present but unreadable"
        }
    } else {
        $parts += "no boot trace at $BootTracePath"
    }
    if ($GraceTracePath -and (Test-Path $GraceTracePath)) {
        try {
            $parts += "grace_trace=$(Get-Content $GraceTracePath -Raw)"
        } catch {
            $parts += "grace trace present but unreadable"
        }
    }
    return ($parts -join "; ")
}

function Wait-ValidationSmokeResult {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$ExpectedMode,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($Process.HasExited) {
            $msg = Format-SmokeFailureMessage `
                -ResultPath $resultPath `
                -BootStatePath $bootStatePath `
                -BootTracePath $bootTracePath `
                -GraceTracePath $graceTracePath
            throw "Installed CUDA app exited early (exit code: $($Process.ExitCode)). $msg"
        }
        if (Test-Path $resultPath) {
            $result = Get-Content $resultPath -Raw | ConvertFrom-Json
            if ($ExpectedMode -and $result.mode -ne $ExpectedMode) {
                Start-Sleep -Seconds 1
                continue
            }
            if (-not $result.ok) {
                $msg = Format-SmokeFailureMessage `
                    -ResultPath $resultPath `
                    -BootStatePath $bootStatePath `
                    -BootTracePath $bootTracePath `
                    -GraceTracePath $graceTracePath
                throw $msg
            }
            return $result
        }
        Start-Sleep -Seconds 1
    }

    $msg = Format-SmokeFailureMessage `
        -ResultPath $resultPath `
        -BootStatePath $bootStatePath `
        -BootTracePath $bootTracePath `
        -GraceTracePath $graceTracePath
    throw "Timed out waiting for validation smoke result at $resultPath. $msg"
}

Install-QubeSilentSetup -SetupPath $setup.FullName

if (-not (Test-Path $installedExe)) {
    throw "Silent install failed — $installedExe not found"
}
Write-Host "Silent install verified at $installedExe"

$installMarker = Join-Path $installDir ".qube-install-ts"
if (-not (Test-Path $installMarker)) {
    throw "Missing CUDA install grace marker: $installMarker"
}

Clear-ValidationDiagnostics

Write-Host "Phase 1: explicit smoke validation (--winget-validation)..."

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

$env:QUBE_WINGET_VALIDATION = "1"
$smokeProc = Start-Process -FilePath $installedExe `
    -ArgumentList "--mock-bootstrap-download", "--winget-validation" `
    -PassThru

try {
    $smokeResult = Wait-ValidationSmokeResult -Process $smokeProc -ExpectedMode "smoke"
    if ($smokeResult.llama_import_attempted) {
        $msg = Format-SmokeFailureMessage `
            -ResultPath $resultPath `
            -BootStatePath $bootStatePath `
            -BootTracePath $bootTracePath `
            -GraceTracePath $graceTracePath
        throw "WinGet validation guard failed: llama_cpp import was attempted. $msg"
    }
    Write-Host "Phase 1 passed (pid $($smokeProc.Id), stage $($smokeResult.stage), no llama_cpp import)"
}
finally {
    Stop-QubeProcessIfRunning -Process $smokeProc
    Remove-Item Env:QUBE_WINGET_VALIDATION -ErrorAction SilentlyContinue
}

Write-Host "Phase 2: WinGet install-grace launch (no validation flags)..."

Clear-ValidationDiagnostics
Remove-Item $settingsPath -Force -ErrorAction SilentlyContinue
Remove-Item $dummyGguf -Force -ErrorAction SilentlyContinue

$graceProc = Start-Process -FilePath $installedExe -PassThru

try {
    $graceResult = Wait-ValidationSmokeResult -Process $graceProc -ExpectedMode "install_grace"
    if ($graceResult.llama_import_attempted) {
        $msg = Format-SmokeFailureMessage `
            -ResultPath $resultPath `
            -BootStatePath $bootStatePath `
            -BootTracePath $bootTracePath `
            -GraceTracePath $graceTracePath
        throw "Install-grace validation failed: llama_cpp import was attempted. $msg"
    }
    Write-Host "Phase 2 passed (pid $($graceProc.Id), stage $($graceResult.stage), no llama_cpp import)"

    Invoke-QubeSilentUninstallWhileRunning `
        -Uninstaller $uninstaller `
        -InstalledExe $installedExe
    Write-Host "CUDA validation smoke + install-grace + uninstall verified"
}
finally {
    Stop-QubeProcessIfRunning -Process $graceProc
}
