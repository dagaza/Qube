# Verify a Windows Vulkan PyInstaller bundle without requiring a GPU device.
#
# Usage:   scripts/release/verify_windows_vulkan_bundle.ps1 [path-to-dist\Qube]
param(
    [string]$Dist = (Join-Path $PSScriptRoot "..\..\dist\Qube" | Resolve-Path -ErrorAction SilentlyContinue)
)

$ErrorActionPreference = "Stop"

if (-not $Dist) {
    $Dist = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "dist\Qube"
}

$binary = Join-Path $Dist "Qube.exe"
$libDir = Join-Path $Dist "_internal\llama_cpp\lib"
$marker = Join-Path $Dist ".qube-windows-variant"

if (-not (Test-Path $binary)) {
    throw "Binary not found: $binary"
}
if (-not (Test-Path $marker) -or (Get-Content $marker -Raw).Trim() -ne "vulkan") {
    throw "Expected Vulkan variant marker at $marker"
}
if (-not (Test-Path $libDir)) {
    throw "Missing llama_cpp lib dir: $libDir"
}

$requiredLibs = @(
    "llama.dll",
    "vulkan-1.dll",
    "ggml-vulkan.dll"
)
foreach ($lib in $requiredLibs) {
    $path = Join-Path $libDir $lib
    if (-not (Test-Path $path)) {
        throw "Missing bundled Vulkan dependency: $path"
    }
}

Write-Host "Vulkan bundle verification passed"
