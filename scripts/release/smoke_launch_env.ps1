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

function Stop-QubeProcessIfRunning {
    param(
        [System.Diagnostics.Process]$Process
    )

    if ($null -eq $Process) {
        return
    }
    if (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

function Stop-AllQubeProcesses {
    Get-Process -Name Qube -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
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
        -PassThru -Wait
    if ($uninstall.ExitCode -ne 0) {
        Write-Host "Uninstaller exit code $($uninstall.ExitCode); force-stopping Qube and retrying..."
        Stop-AllQubeProcesses
        Start-Sleep -Seconds 2
        Start-Process -Wait -FilePath $Uninstaller `
            -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
    }

    Wait-QubeInstallRemoved -InstalledExe $InstalledExe -InternalDir $InternalDir
    Write-Host "Uninstall verified (app was running during removal)"
}
