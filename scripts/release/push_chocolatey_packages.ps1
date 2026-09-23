# Push rendered Chocolatey packages for a release version.
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,

    # Optional subset, e.g. @('qube-cuda') — defaults to all .nupkg files found.
    [string[]]$PackageIds = @()
)

$ErrorActionPreference = "Stop"

$nupkgs = Get-ChildItem -Path $PackageRoot -Recurse -Filter "*.nupkg" | Sort-Object Name
if (-not $nupkgs) {
    throw "No .nupkg files found under $PackageRoot"
}

if ($PackageIds.Count -gt 0) {
    $allowed = @{}
    foreach ($id in $PackageIds) {
        $allowed[$id.ToLowerInvariant()] = $true
    }
    $nupkgs = @(
        $nupkgs | Where-Object {
            $name = $_.BaseName
            foreach ($id in $allowed.Keys) {
                if ($name -eq "$id.$Version" -or $name -like "$id.*") {
                    return $true
                }
            }
            return $false
        }
    )
    if (-not $nupkgs) {
        throw "No .nupkg files matched PackageIds: $($PackageIds -join ', ')"
    }
}

$pushed = 0
foreach ($nupkg in $nupkgs) {
    Write-Host "Pushing $($nupkg.FullName)..."
    & choco push $nupkg.FullName --source https://push.chocolatey.org/ --api-key $env:CHOCOLATEY_API_KEY
    if ($LASTEXITCODE -ne 0) {
        throw "choco push failed for $($nupkg.Name) (exit code $LASTEXITCODE)"
    }
    $pushed++
}

Write-Host "Pushed $pushed Chocolatey package(s) for version $Version."
