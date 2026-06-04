# Silent install, launch installed EXE, then uninstall.
$ErrorActionPreference = "Stop"

$setup = Get-Item (Join-Path $PSScriptRoot "..\..\installer\output\Qube-*-Setup.exe") | Select-Object -First 1
$installDir = Join-Path $env:LOCALAPPDATA "Programs\Qube"
$installedExe = Join-Path $installDir "Qube.exe"
$uninstaller = Join-Path $installDir "unins000.exe"

Write-Host "Installing $($setup.Name) silently..."
Start-Process -Wait -FilePath $setup.FullName `
    -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART"

if (-not (Test-Path $installedExe)) {
    throw "Silent install failed — $installedExe not found"
}
Write-Host "Silent install verified at $installedExe"

Write-Host "Launching installed EXE..."
$proc = Start-Process -FilePath $installedExe -PassThru
Start-Sleep -Seconds 10
if ($proc.HasExited) {
    throw "Installed app crashed on launch (exit code: $($proc.ExitCode))"
}
Write-Host "Installed EXE smoke test passed"
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

if (-not (Test-Path $uninstaller)) {
    throw "Uninstaller not found at $uninstaller"
}
Write-Host "Uninstalling..."
Start-Process -Wait -FilePath $uninstaller `
    -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART"
Start-Sleep -Seconds 2

if (Test-Path $installedExe) {
    throw "Uninstall failed — $installedExe still exists"
}
Write-Host "Uninstall verified"
