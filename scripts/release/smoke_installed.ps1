# Silent install, launch installed EXE, uninstall while running, verify removal.
param(
    [string]$SetupPath = ""
)

$ErrorActionPreference = "Stop"

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

Write-Host "Installing $($setup.Name) silently..."
Start-Process -Wait -FilePath $setup.FullName `
    -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART"

if (-not (Test-Path $installedExe)) {
    throw "Silent install failed — $installedExe not found"
}
Write-Host "Silent install verified at $installedExe"

Write-Host "Launching installed EXE (simulates tray background before uninstall)..."
$proc = Start-Process -FilePath $installedExe -PassThru
Start-Sleep -Seconds 8
if ($proc.HasExited) {
    throw "Installed app crashed on launch (exit code: $($proc.ExitCode))"
}
if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
    throw "Installed app process exited before uninstall test"
}
Write-Host "Installed EXE running (pid $($proc.Id))"

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
    if (-not (Test-Path $installedExe) -and -not (Test-Path $internalDir)) {
        break
    }
    Start-Sleep -Seconds 1
}

if (Test-Path $installedExe) {
    throw "Uninstall failed — $installedExe still exists"
}
if (Test-Path $internalDir) {
    throw "Uninstall failed — $internalDir still exists"
}
Write-Host "Uninstall verified (app was running during removal)"
