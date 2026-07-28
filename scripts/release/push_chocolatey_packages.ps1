# Push all rendered Chocolatey packages for a release version.
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$PackageRoot
)

$ErrorActionPreference = "Stop"

$nupkgs = Get-ChildItem -Path $PackageRoot -Recurse -Filter "*.nupkg" | Sort-Object Name
if (-not $nupkgs) {
    throw "No .nupkg files found under $PackageRoot"
}

foreach ($nupkg in $nupkgs) {
    Write-Host "Pushing $($nupkg.FullName)..."
    & choco push $nupkg.FullName --source https://push.chocolatey.org/ --api-key $env:CHOCOLATEY_API_KEY
}

Write-Host "Pushed $($nupkgs.Count) Chocolatey package(s) for version $Version."
