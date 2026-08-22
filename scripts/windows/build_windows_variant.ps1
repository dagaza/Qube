# Build the PyInstaller one-dir bundle and Inno Setup installer for Windows.
#
# Usage:   scripts/windows/build_windows_variant.ps1 [version] [cpu|vulkan|cuda]
param(
    [string]$Version = "",
    [ValidateSet("cpu", "vulkan", "cuda")]
    [string]$Variant = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

if ($Version) {
    python scripts/set_version.py $Version
}

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller pillow

& "$Root\scripts\windows\install_llama_cpp_variant.ps1" $Variant

if ($Variant -eq "cuda") {
    python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
}

python scripts/generate_ico.py

$env:QUBE_WINDOWS_VARIANT = $Variant
python -m PyInstaller qube.spec --noconfirm

$libDir = Join-Path $Root "dist\Qube\_internal\llama_cpp\lib"
if ($Variant -eq "cuda") {
    python scripts/stage_cuda_runtime_libs.py $libDir
    if ($LASTEXITCODE -ne 0) { throw "CUDA runtime staging failed" }
}
if ($Variant -eq "vulkan") {
    python scripts/stage_vulkan_runtime_libs.py $libDir
    if ($LASTEXITCODE -ne 0) { throw "Vulkan runtime staging failed" }
}

$distExe = Join-Path $Root "dist\Qube\Qube.exe"
if (-not (Test-Path $distExe)) {
    throw "PyInstaller output missing: $distExe"
}

Set-Content -Path (Join-Path $Root "dist\Qube\.qube-windows-variant") -Value $Variant -NoNewline
Write-Host "Built $distExe ($Variant)"
