# Smoke-test upgrading an existing Qube install with a newer Inno Setup installer.
# Requires two Setup.exe files built from the same PyInstaller dist (different AppVersion).
param(
    [Parameter(Mandatory = $true)]
    [string]$OldSetup,

    [Parameter(Mandatory = $true)]
    [string]$NewSetup,

    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\Qube"),

    [string]$ExpectedOldVersion,

    [string]$ExpectedNewVersion
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/smoke_launch_env.ps1"

$AppId = "{B7E4A3F1-92C0-4D8B-A6E5-3F1C7D9B0E42}_is1"
$UninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
$InstalledExe = Join-Path $InstallDir "Qube.exe"

function Get-InstalledQubeVersion {
    if (-not (Test-Path $UninstallKey)) {
        return $null
    }
    return (Get-ItemProperty $UninstallKey -ErrorAction SilentlyContinue).DisplayVersion
}

foreach ($setup in @($OldSetup, $NewSetup)) {
    if (-not (Test-Path $setup)) {
        throw "Installer not found: $setup"
    }
}

if (Test-Path $InstalledExe) {
    Write-Host "Removing leftover install at $InstallDir before upgrade smoke test..."
    $existingUninstaller = Join-Path $InstallDir "unins000.exe"
    if (Test-Path $existingUninstaller) {
        Stop-AllQubeProcesses
        Wait-QubeProcessesExited
        Start-Process -Wait -FilePath $existingUninstaller `
            -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
    }
}

Install-QubeSilentSetup -SetupPath $OldSetup

if (-not (Test-Path $InstalledExe)) {
    throw "Initial install failed — $InstalledExe not found"
}

$oldVersion = Get-InstalledQubeVersion
if (-not $oldVersion) {
    throw "Could not read installed version from registry ($UninstallKey)"
}
if ($ExpectedOldVersion -and $oldVersion -ne $ExpectedOldVersion) {
    throw "Expected old version $ExpectedOldVersion but registry reports $oldVersion"
}
Write-Host "Initial install verified (version $oldVersion at $InstallDir)"

Install-QubeSilentSetup -SetupPath $NewSetup

if (-not (Test-Path $InstalledExe)) {
    throw "Upgrade failed — $InstalledExe not found"
}

$newVersion = Get-InstalledQubeVersion
if (-not $newVersion) {
    throw "Could not read upgraded version from registry ($UninstallKey)"
}
if ($ExpectedNewVersion -and $newVersion -ne $ExpectedNewVersion) {
    throw "Expected new version $ExpectedNewVersion but registry reports $newVersion"
}
if ($oldVersion -eq $newVersion) {
    throw "Upgrade did not change DisplayVersion ($oldVersion)"
}
Write-Host "Upgrade verified (version $oldVersion -> $newVersion at $InstallDir)"

Write-Host "Launching upgraded EXE..."
$state = Enter-QubeSmokeLaunchEnvironment
$proc = $null
try {
    $launchArgs = Get-QubeSmokeLaunchArgumentList
    $proc = Start-Process -FilePath $InstalledExe -ArgumentList $launchArgs -PassThru
    Start-Sleep -Seconds 10
    if ($proc.HasExited) {
        throw "Upgraded app crashed on launch (exit code: $($proc.ExitCode))"
    }
    Write-Host "Upgrade smoke test passed"
}
finally {
    Stop-QubeProcessIfRunning -Process $proc
    Exit-QubeSmokeLaunchEnvironment -State $state
}
