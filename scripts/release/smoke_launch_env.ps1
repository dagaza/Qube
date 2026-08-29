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
    $env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD = "1"
    return $state
}

function Exit-QubeSmokeLaunchEnvironment {
    param($State)

    if ($null -eq $State) {
        return
    }
    $env:LOCALAPPDATA = $State.PreviousAppData
    $env:USERPROFILE = $State.PreviousProfile
    Remove-Item Env:QUBE_BOOTSTRAP_MOCK_DOWNLOAD -ErrorAction SilentlyContinue
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
