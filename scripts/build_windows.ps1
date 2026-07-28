# Local Windows release build (parity with CI).
param(
    # Semver X.Y.Z; defaults to core/__version__.py when omitted.
    [string]$Version,

    [ValidateSet("cpu", "vulkan", "cuda")]
    [string]$Variant = "cpu",

    # Stop after PyInstaller + dist EXE smoke (skip Inno installer steps).
    [switch]$SkipInstaller,

    # Do not attempt `choco install innosetup` when ISCC.exe is missing.
    [switch]$SkipChoco
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $Version) {
    $Version = python -c "from core.__version__ import __version__; print(__version__, end='')"
    if (-not $Version) { throw "Could not read version from core/__version__.py" }
    Write-Host "Using version from core/__version__.py: $Version"
}

function Resolve-InnoSetupCompiler {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) { return $path }
    }
    $cmd = Get-Command iscc -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Install-InnoSetupViaChocolatey {
    if ($SkipChoco) { return $null }
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if (-not $choco) {
        Write-Warning "Chocolatey not found; skipping automatic Inno Setup install."
        return $null
    }
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
    if (-not $isAdmin) {
        Write-Warning @"
Not running as Administrator — Chocolatey cannot install Inno Setup into C:\ProgramData\chocolatey.
Install Inno Setup 6 manually from https://jrsoftware.org/isinfo.php (add ISCC to PATH), then re-run.
"@
        return $null
    }
    Write-Host "Installing Inno Setup via Chocolatey..."
    & choco install innosetup -y --no-progress
    return Resolve-InnoSetupCompiler
}

python scripts/set_version.py $Version
& "$Root\scripts\windows\build_windows_variant.ps1" $Version $Variant

if ($SkipInstaller) {
    Write-Host "SkipInstaller set — dist build OK at dist\Qube\Qube.exe"
    exit 0
}

$iscc = Resolve-InnoSetupCompiler
if (-not $iscc) {
    $iscc = Install-InnoSetupViaChocolatey
}
if (-not $iscc) {
    throw @"
Inno Setup 6 (ISCC.exe) is required to build the installer.

PyInstaller succeeded: dist\Qube\Qube.exe

Install Inno Setup, then compile the installer only:
  & "`$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" /DMyAppVersion=$Version installer\qube.iss
  (or "`${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" for a system-wide install)
  .\scripts\release\smoke_installed.ps1

Or install via elevated PowerShell: choco install innosetup -y
Or re-run this script with -SkipInstaller after validating dist\Qube\Qube.exe manually.
"@
}

Write-Host "Using Inno Setup compiler: $iscc"
& $iscc "/DMyAppVersion=$Version" "/DMyAppVariant=$Variant" "installer\qube.iss"
& "$Root\scripts\release\smoke_installed.ps1"
Get-Item installer\output\Qube-*-Setup.exe | Format-List Name, Length
