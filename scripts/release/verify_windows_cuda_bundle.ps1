# Verify a Windows CUDA PyInstaller bundle without launching the app.
#
# CUDA llama-cpp wheels load nvcuda.dll (NVIDIA driver) at import time.
# GitHub-hosted runners have no GPU/driver, so release CI validates layout
# and bundled runtime deps instead of running the full app smoke test.
#
# Usage:   scripts/release/verify_windows_cuda_bundle.ps1 [path-to-dist\Qube]
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
if (-not (Test-Path $marker) -or (Get-Content $marker -Raw).Trim() -ne "cuda") {
    throw "Expected CUDA variant marker at $marker"
}
if (-not (Test-Path $libDir)) {
    throw "Missing llama_cpp lib dir: $libDir"
}

$requiredLibs = @(
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "ggml-cuda.dll"
)
foreach ($lib in $requiredLibs) {
    $path = Join-Path $libDir $lib
    if (-not (Test-Path $path)) {
        throw "Missing bundled CUDA dependency: $path"
    }
}

$llamaCandidates = @("llama.dll")
$llamaFound = $false
foreach ($candidate in $llamaCandidates) {
    if (Test-Path (Join-Path $libDir $candidate)) {
        $llamaFound = $true
        break
    }
}
if (-not $llamaFound) {
    throw "Missing llama shared library under $libDir"
}

Write-Host "CUDA bundle verification passed (runtime smoke skipped: no NVIDIA driver on CI)"
