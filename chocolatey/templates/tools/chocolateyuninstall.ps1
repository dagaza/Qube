$ErrorActionPreference = 'Stop'

$installDir = Join-Path $env:LOCALAPPDATA 'Programs\Qube'
$uninstaller = Join-Path $installDir 'unins000.exe'

if (Test-Path $uninstaller) {
    $proc = Start-Process -FilePath $uninstaller `
        -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' `
        -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Qube uninstaller exited with code $($proc.ExitCode)"
    }
    Write-Host "Qube uninstalled via Inno uninstaller."
    return
}

# Fallback: registry uninstall string (Inno per-user entry)
$appId = '{B7E4A3F1-92C0-4D8B-A6E5-3F1C7D9B0E42}_is1'
$regPaths = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$appId",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$appId"
)
foreach ($regPath in $regPaths) {
    if (-not (Test-Path $regPath)) { continue }
    $quiet = (Get-ItemProperty $regPath).QuietUninstallString
    $normal = (Get-ItemProperty $regPath).UninstallString
    $cmd = if ($quiet) { $quiet } else { $normal }
    if ($cmd) {
        Write-Host "Uninstalling via registry: $cmd"
        cmd /c $cmd
        return
    }
}

Write-Warning 'Qube does not appear to be installed; nothing to uninstall.'
