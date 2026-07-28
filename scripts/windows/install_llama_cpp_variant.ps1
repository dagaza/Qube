# Install a specific llama-cpp-python backend into the active Python environment.
#
# Usage:
#   scripts/windows/install_llama_cpp_variant.ps1 cpu|vulkan|cuda
#
# Environment:
#   LLAMA_CPP_VERSION      default 0.3.29
#   LLAMA_BUILD_JOBS       default 1
#   CUDA_WHEEL_TAG         default cu124
#   LLAMA_CPP_FORCE_CUDA_SOURCE=1  skip CUDA prebuilt wheel
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("cpu", "vulkan", "cuda")]
    [string]$Variant
)

$ErrorActionPreference = "Stop"

$Version = if ($env:LLAMA_CPP_VERSION) { $env:LLAMA_CPP_VERSION } else { "0.3.29" }
$Jobs = if ($env:LLAMA_BUILD_JOBS) { $env:LLAMA_BUILD_JOBS } else { "1" }

function Invoke-PipInstall {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    python -m pip install @Args
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed (exit $LASTEXITCODE)"
    }
}

function Test-LlamaCppInstall {
    param([bool]$ExpectGpu)
    if ($env:GITHUB_ACTIONS -eq "true" -and $ExpectGpu) {
        Write-Host "==> CI: skipping GPU runtime import check (runner has no GPU device)."
        python -c "import importlib.metadata as md; print('installed:', md.version('llama_cpp_python'))"
        return
    }
    $expectFlag = if ($ExpectGpu) { "1" } else { "0" }
    python -c @"
import sys
expect_gpu = sys.argv[1] == '1'
import llama_cpp
print('version:', llama_cpp.__version__)
print('supports_gpu_offload:', llama_cpp.llama_supports_gpu_offload())
info = llama_cpp.llama_print_system_info()
print('system_info:', info.decode() if isinstance(info, (bytes, bytearray)) else info)
if expect_gpu and not llama_cpp.llama_supports_gpu_offload():
    raise SystemExit('Expected GPU offload to be available for this build variant.')
if not expect_gpu and llama_cpp.llama_supports_gpu_offload():
    print('warning: CPU build reports GPU offload available', file=sys.stderr)
"@ $expectFlag
    if ($LASTEXITCODE -ne 0) {
        throw "llama-cpp-python verification failed"
    }
}

Write-Host "==> Qube llama-cpp variant installer: $Variant"
Write-Host "    version: $Version"
Write-Host "    build jobs: $Jobs"

switch ($Variant) {
    "cpu" {
        Write-Host "==> Installing CPU llama-cpp-python ($Version)..."
        Invoke-PipInstall "llama-cpp-python==$Version" --force-reinstall --no-cache-dir
        Test-LlamaCppInstall $false
    }
    "vulkan" {
        if (-not $env:VULKAN_SDK) {
            $sdkRoot = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -First 1
            if ($sdkRoot) {
                $env:VULKAN_SDK = $sdkRoot.FullName
            }
        }
        if (-not $env:VULKAN_SDK) {
            throw "VULKAN_SDK is not set and no SDK was found under C:\VulkanSDK"
        }
        $env:Path = "$env:VULKAN_SDK\Bin;$env:Path"
        $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
        $env:CMAKE_BUILD_PARALLEL_LEVEL = $Jobs
        $env:MAX_JOBS = $Jobs
        Write-Host "==> Building llama-cpp-python with Vulkan ($Version, jobs=$Jobs)..."
        Invoke-PipInstall "llama-cpp-python==$Version" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
        Test-LlamaCppInstall $true
    }
    "cuda" {
        $cu = if ($env:CUDA_WHEEL_TAG) { $env:CUDA_WHEEL_TAG } else { "cu124" }
        Write-Host "==> Installing CUDA llama-cpp-python ($Version, wheel $cu)..."
        if ($env:LLAMA_CPP_FORCE_CUDA_SOURCE -eq "1") {
            $env:CMAKE_ARGS = "-DGGML_CUDA=on"
            Invoke-PipInstall "llama-cpp-python==$Version" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
        }
        else {
            try {
                Invoke-PipInstall "llama-cpp-python==$Version" --force-reinstall --no-cache-dir `
                    --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/$cu"
            }
            catch {
                Write-Host "==> CUDA wheel install failed; falling back to source build..."
                $env:CMAKE_ARGS = "-DGGML_CUDA=on"
                Invoke-PipInstall "llama-cpp-python==$Version" --force-reinstall --no-cache-dir --no-binary=llama-cpp-python
            }
        }
        Test-LlamaCppInstall $true
    }
}

Write-Host "==> llama-cpp-python ($Variant) install complete."
