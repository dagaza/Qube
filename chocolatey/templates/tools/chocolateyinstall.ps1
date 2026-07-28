$ErrorActionPreference = 'Stop'

# Qube is x64-only
if ($env:chocolateyforcex86 -eq 'true') {
    throw 'Qube does not support 32-bit Windows.'
}

$version    = '{{VERSION}}'
$url        = '{{INSTALLER_URL}}'
$checksum   = '{{SHA256}}'
$silentArgs = '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART'

$packageArgs = @{
    packageName    = $env:ChocolateyPackageName
    fileType       = 'exe'
    url            = $url
    checksum       = $checksum
    checksumType   = 'sha256'
    silentArgs     = $silentArgs
    validExitCodes = @(0)
    softwareName   = 'Qube*'
}

Install-ChocolateyPackage @packageArgs
