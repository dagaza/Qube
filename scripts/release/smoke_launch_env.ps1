# Shared isolated profile + mock bootstrap helpers for Windows release smoke scripts.
# Prevents first-run bootstrap/consent/download hangs on CI runners.

function Initialize-QubeSmokeSettings {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SettingsDir
    )

    New-Item -ItemType Directory -Path $SettingsDir -Force | Out-Null
    @'
{
  "qube.bootstrap.completed": true
}
'@ | Set-Content -Path (Join-Path $SettingsDir "settings.json") -Encoding utf8NoBOM
}

function Enter-QubeSmokeLaunchEnvironment {
    $state = [ordered]@{
        PreviousAppData = $env:LOCALAPPDATA
        PreviousProfile = $env:USERPROFILE
        FakeAppData       = Join-Path $env:TEMP ("qube-smoke-" + [guid]::NewGuid().ToString())
        FakeProfile       = Join-Path $env:TEMP ("qube-smoke-profile-" + [guid]::NewGuid().ToString())
    }
    Initialize-QubeSmokeSettings -SettingsDir (Join-Path $state.FakeProfile ".qube")
    $env:LOCALAPPDATA = $state.FakeAppData
    $env:USERPROFILE = $state.FakeProfile
    # Mock downloads are enabled via --mock-bootstrap-download on the child CLI only.
    # Do not set QUBE_BOOTSTRAP_MOCK_DOWNLOAD here: inheriting it broke CUDA dist smoke
    # (WinGet validation mode exited with code 2 on CI).
    return $state
}

function Exit-QubeSmokeLaunchEnvironment {
    param($State)

    if ($null -eq $State) {
        return
    }
    $env:LOCALAPPDATA = $State.PreviousAppData
    $env:USERPROFILE = $State.PreviousProfile
    Remove-Item -Recurse -Force $State.FakeAppData -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $State.FakeProfile -ErrorAction SilentlyContinue
}

function Get-QubeSmokeLaunchArgumentList {
    param(
        [string[]]$Additional = @()
    )

    return @("--mock-bootstrap-download") + $Additional
}

function Stop-AllQubeProcesses {
    # Match installer/qube.iss KillRunningQube: /T so PyInstaller children release AppMutex.
    # Do not invoke taskkill as a native command under $ErrorActionPreference=Stop — missing
    # Qube.exe returns 128 and would abort the smoke script on PowerShell 7.4+.
    Get-Process -Name Qube -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    $taskkill = Join-Path $env:SystemRoot "System32\taskkill.exe"
    if ((Test-Path $taskkill) -and (Get-Process -Name Qube -ErrorAction SilentlyContinue)) {
        Start-Process -FilePath $taskkill -ArgumentList "/F", "/IM", "Qube.exe", "/T" `
            -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue | Out-Null
    }
    Get-Process -Name Qube -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

function Wait-QubeProcessesExited {
    param([int]$TimeoutSec = 15)

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Name Qube -ErrorAction SilentlyContinue)) {
            return
        }
        Stop-AllQubeProcesses
        Start-Sleep -Milliseconds 500
    }
    $left = @(Get-Process -Name Qube -ErrorAction SilentlyContinue)
    if ($left.Count -gt 0) {
        throw "Qube.exe still running after stop (pids $($left.Id -join ', ')); silent Setup would hang on AppMutex"
    }
}

function Stop-QubeProcessIfRunning {
    param(
        [System.Diagnostics.Process]$Process
    )

    if ($null -ne $Process) {
        if (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Stop-AllQubeProcesses
    try {
        Wait-QubeProcessesExited
    } catch {
        Write-Host "WARNING: $($_.Exception.Message)"
    }
}

function Install-QubeSilentSetup {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SetupPath,
        [int]$TimeoutSec = 240
    )

    if (-not (Test-Path $SetupPath)) {
        throw "Installer not found: $SetupPath"
    }

    Stop-AllQubeProcesses
    Wait-QubeProcessesExited -TimeoutSec 15

    Write-Host "Installing $SetupPath silently..."
    $setup = Start-Process -FilePath $SetupPath `
        -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" `
        -PassThru
    if (-not $setup.WaitForExit($TimeoutSec * 1000)) {
        Stop-Process -Id $setup.Id -Force -ErrorAction SilentlyContinue
        Stop-AllQubeProcesses
        throw "Silent install timed out after ${TimeoutSec}s ($SetupPath). Leftover Qube.exe holding AppMutex?"
    }
    if ($setup.ExitCode -ne 0) {
        throw "Silent install exited with code $($setup.ExitCode) ($SetupPath)"
    }
}

function Wait-QubeInstallRemoved {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstalledExe,
        [string]$InternalDir = "",
        [int]$TimeoutSec = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        Stop-AllQubeProcesses
        $removed = -not (Test-Path $InstalledExe)
        if ($removed -and $InternalDir -and (Test-Path $InternalDir)) {
            $removed = $false
        }
        if ($removed) {
            return
        }
        Start-Sleep -Seconds 1
    }

    if (Test-Path $InstalledExe) {
        throw "Uninstall failed — $InstalledExe still exists"
    }
    if ($InternalDir -and (Test-Path $InternalDir)) {
        throw "Uninstall failed — $InternalDir still exists"
    }
}

function Invoke-QubeSilentUninstallWhileRunning {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uninstaller,
        [Parameter(Mandatory = $true)]
        [string]$InstalledExe,
        [string]$InternalDir = ""
    )

    if (-not (Test-Path $Uninstaller)) {
        throw "Uninstaller not found at $Uninstaller"
    }

    Write-Host "Uninstalling while Qube.exe is still running..."
    $uninstall = Start-Process -FilePath $Uninstaller `
        -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" `
        -PassThru
    if (-not $uninstall.WaitForExit(180000)) {
        Stop-Process -Id $uninstall.Id -Force -ErrorAction SilentlyContinue
        Stop-AllQubeProcesses
        throw "Silent uninstall timed out after 180s. AppMutex still held?"
    }
    if ($uninstall.ExitCode -ne 0) {
        Write-Host "Uninstaller exit code $($uninstall.ExitCode); force-stopping Qube and retrying..."
        Stop-AllQubeProcesses
        Wait-QubeProcessesExited
        Start-Process -Wait -FilePath $Uninstaller `
            -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
    }

    Wait-QubeInstallRemoved -InstalledExe $InstalledExe -InternalDir $InternalDir
    Write-Host "Uninstall verified (app was running during removal)"
}
