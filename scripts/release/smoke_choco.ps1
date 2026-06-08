# Smoke-test a Chocolatey package: install from local nupkg, verify app, uninstall.
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$PackageDir
)

$ErrorActionPreference = "Stop"

$nupkg = Get-ChildItem -Path $PackageDir -Filter "qube.$Version.nupkg" -Recurse | Select-Object -First 1
if (-not $nupkg) {
    $nupkg = Get-ChildItem -Path $PackageDir -Filter "qube.*.nupkg" -Recurse | Select-Object -First 1
}
if (-not $nupkg) {
    throw "No qube nupkg found under $PackageDir"
}

$sourceDir = Split-Path -Parent $nupkg.FullName
$installDir = Join-Path $env:LOCALAPPDATA "Programs\Qube"
$installedExe = Join-Path $installDir "Qube.exe"

Write-Host "Installing qube $Version from $($nupkg.FullName)..."
& choco install qube -y --source="$sourceDir" --version=$Version --force

if (-not (Test-Path $installedExe)) {
    throw "Chocolatey install failed — $installedExe not found"
}
Write-Host "Chocolatey install verified at $installedExe"

Write-Host "Launching installed EXE..."
$proc = Start-Process -FilePath $installedExe -PassThru
Start-Sleep -Seconds 10
if ($proc.HasExited) {
    throw "Installed app crashed on launch (exit code: $($proc.ExitCode))"
}
Write-Host "Installed EXE smoke test passed"
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

Write-Host "Uninstalling qube via Chocolatey..."
& choco uninstall qube -y

Start-Sleep -Seconds 2
if (Test-Path $installedExe) {
    throw "Chocolatey uninstall failed — $installedExe still exists"
}
Write-Host "Chocolatey uninstall verified"
