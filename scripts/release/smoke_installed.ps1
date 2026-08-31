# Silent install, launch installed EXE, uninstall while running, verify removal.
param(
    [string]$SetupPath = ""
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/smoke_launch_env.ps1"

if ($SetupPath) {
    $setup = Get-Item $SetupPath
} else {
    $setup = Get-ChildItem (Join-Path $PSScriptRoot "..\..\installer\output\Qube-*-Setup.exe") |
        Sort-Object { [version]($_.BaseName -replace '^Qube-','' -replace '-(vulkan|cuda)$','') } -Descending |
        Select-Object -First 1
}

if (-not $setup) {
    throw "No Qube-*-Setup.exe found under installer/output"
}

$installDir = Join-Path $env:LOCALAPPDATA "Programs\Qube"
$installedExe = Join-Path $installDir "Qube.exe"
$internalDir = Join-Path $installDir "_internal"
$uninstaller = Join-Path $installDir "unins000.exe"

Install-QubeSilentSetup -SetupPath $setup.FullName

if (-not (Test-Path $installedExe)) {
    throw "Silent install failed — $installedExe not found"
}
Write-Host "Silent install verified at $installedExe"

Write-Host "Launching installed EXE (simulates tray background before uninstall)..."
$state = Enter-QubeSmokeLaunchEnvironment
$proc = $null
try {
    $launchArgs = Get-QubeSmokeLaunchArgumentList
    $proc = Start-Process -FilePath $installedExe -ArgumentList $launchArgs -PassThru
    Start-Sleep -Seconds 8
    if ($proc.HasExited) {
        throw "Installed app crashed on launch (exit code: $($proc.ExitCode))"
    }
    if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
        throw "Installed app process exited before uninstall test"
    }
    Write-Host "Installed EXE running (pid $($proc.Id))"

    Invoke-QubeSilentUninstallWhileRunning `
        -Uninstaller $uninstaller `
        -InstalledExe $installedExe `
        -InternalDir $internalDir
}
finally {
    Stop-QubeProcessIfRunning -Process $proc
    Exit-QubeSmokeLaunchEnvironment -State $state
}
