# ROCm support exploration (AMD HIP on Linux)

**Status:** Exploration only — no decision to implement.  
**Date:** 2026-07-25  
**Context:** Qube ships CUDA (NVIDIA), Vulkan (AMD/Intel on Linux), and Metal (macOS). This document captures feasibility, trade-offs, and implications if we add a packaged **ROCm / HIP** variant for AMD discrete GPUs on Linux.

---

## Current GPU support

| Platform | Native inference backend | How it is shipped |
|----------|-------------------------|-------------------|
| **Linux** | CPU, **Vulkan** (AMD/Intel), **CUDA** (NVIDIA) | Three AppImages + three `.deb`s (`cpu`, `vulkan`, `cuda`) |
| **macOS** | **Metal** | Signed DMGs (arm64 + x86_64) |
| **Windows** | Generic `llama-cpp-python` from PyPI (effectively **CPU** in release CI) | Single installer |

For **AMD on Linux**, the official packaged path today is the **`vulkan`** variant. A ROCm variant would be a *fourth* Linux backend, not a replacement for Vulkan.

Relevant code paths:

- Variant registry: `core/linux_release_variants.py`
- llama.cpp install per variant: `scripts/linux/install_llama_cpp_variant.sh`
- Linux packaging: `scripts/linux/build_linux.sh`, `build_appimage.sh`, `build_deb.sh`
- Release matrix: `.github/workflows/release.yml` (`cpu`, `vulkan`, `cuda`)
- GPU memory / layer caps: `core/gpu_layers_cap.py` (already distinguishes AMD discrete vs APU unified memory)
- Backend detection: `core/inference_transparency.py` (already parses `HIP` from `llama_print_system_info()`)

User-facing summary: [system requirements — GPU acceleration](user/system-requirements.md).

---

## What ROCm would add

**ROCm (HIP)** is AMD’s compute stack — the functional analog of CUDA for AMD GPUs. In llama.cpp / llama-cpp-python it is enabled at build time with CMake flags such as `GGML_HIP=ON` and an `AMDGPU_TARGETS` (or `GPU_TARGETS`) architecture list.

**Potential benefit:** On supported AMD discrete GPUs, HIP offload can outperform Vulkan for llama.cpp inference, especially on recent RDNA cards where ROCm maturity has improved.

**Important scope limit:** Like CUDA today, ROCm would accelerate **native llama.cpp chat only**. STT (faster-whisper / ONNX Runtime), embeddings (fastembed / ONNX), and related pipelines would remain on **CPU** unless we separately invest in ROCm ports of those stacks ( uncommon in desktop apps).

---

## Would it be easy?

### Architecturally: moderately straightforward

Qube already has a clean **Linux variant pattern**. Adding `rocm` would largely mirror CUDA:

1. Extend `LINUX_RELEASE_VARIANTS` and deb/AppImage naming (`qube-rocm_*_amd64.deb`, etc.).
2. Add `install_rocm()` in `install_llama_cpp_variant.sh` (source build with HIP flags).
3. Stage ROCm shared libraries beside `libllama` (analog of `stage_cuda_runtime_libs.py` / `nvidia_wheel_lib_dirs.py`).
4. Extend `qube.spec` binary collection when `QUBE_LINUX_VARIANT=rocm`.
5. Add CI matrix leg, smoke/verify scripts, help text, and mutual `.deb` conflicts.

The plumbing is understood; this is incremental work on an existing model.

### Operationally: not easy

ROCm packaging is significantly harder than CUDA or Vulkan in practice.

#### 1. No reliable prebuilt wheels

CUDA Linux releases use abetlen’s prebuilt `cu124` wheels — fast CI and predictable bundles. **There is no equivalent official ROCm wheel index** for llama-cpp-python. We would almost certainly **compile from source on every release**, similar to Vulkan but with heavier toolchain requirements (ROCm SDK, clang, explicit gfx targets).

#### 2. GPU architecture fragmentation

ROCm builds target specific **gfx** ISAs (e.g. `gfx1030`, `gfx1100`). We must choose:

| Strategy | Pros | Cons |
|----------|------|------|
| **One fat build** (many `AMDGPU_TARGETS`) | Single artifact for users | Very large binary; worse GitHub 2 GiB asset risk |
| **Per-architecture artifacts** | Smaller per-file downloads | More CI jobs, more user confusion (“which RX 7900 build?”) |

NVIDIA’s runtime model is more uniform; ROCm is sensitive to card generation and ROCm version pairing.

#### 3. Bundle size (same cliff as CUDA)

ROCm runtimes (`hipBLAS`, `rocBLAS`, etc.) are large. A packaged ROCm `.deb` would likely land in the **~2–3 GiB** range — the same GitHub Releases per-asset limit that blocked `qube-cuda_*_amd64.deb` until pruning and max xz compression. Any ROCm deliverable needs the same size discipline from day one.

#### 4. Limited hardware and OS support

ROCm officially targets **recent AMD discrete GPUs** (RDNA2+, Instinct, etc.) on **specific Linux versions**. Many consumer cards require workarounds (e.g. `HSA_OVERRIDE_GFX_VERSION`). **APUs** are a gray area — Vulkan plus existing unified-memory heuristics in `gpu_layers_cap.py` may remain the better path there.

**Windows ROCm** exists from AMD, but llama-cpp-python HIP builds on Windows are notoriously unreliable. Do not plan a Windows ROCm installer without a separate spike.

#### 5. Support and documentation burden

Users must choose among CPU / Vulkan / CUDA / ROCm on Linux. ROCm failures often involve kernel, driver, gfx version, and ROCm SDK mismatches — higher support load than “install Vulkan build + mesa.”

---

## ROCm vs Vulkan for AMD users

| | **Vulkan (shipped today)** | **ROCm (hypothetical)** |
|--|---------------------------|-------------------------|
| AMD discrete Linux | Supported, packaged | Potentially faster on supported cards |
| Intel GPU Linux | Same `vulkan` build | Not applicable |
| AMD APU | Works; unified-memory tuning exists | Unclear benefit |
| Driver / runtime burden | Mesa + `libvulkan1` (often present) | ROCm stack or bundled libs |
| CI / build | Source build (~tens of minutes) | Similar or worse |
| Typical AppImage size | ~380 MiB (vulkan) | Likely **1–2+ GiB** |

**Vulkan is the portable default.** ROCm is a **performance option for a subset** of AMD discrete Linux users who outgrow Vulkan throughput.

---

## Implications if we pursue it

### Product

- Another install choice in release notes and help (“Vulkan or ROCm?”).
- Clear guidance on which AMD generations are supported.
- Possible future UX: detect RDNA + suggest ROCm build (optional).

### Engineering

- Fourth leg in the Linux release matrix (+ CI minutes).
- New scripts: HIP lib staging, bundle verify (no GPU on CI — layout/`ldd` checks like CUDA).
- Bundle prune / compression pipeline applies to ROCm `.deb`s as well.
- Extend `install_llama_cpp_gpu.sh` vendor detection (`amd` → offer `vulkan` vs document `rocm` source install).

### Release / cost

- No third-party storage required if artifacts stay under GitHub’s 2 GiB cap.
- **CI time and artifact count** increase; risk of release job failure from oversized `.deb` repeats.

---

## Suggested path if we ever implement

Do **not** jump straight to a packaged fourth variant. Prefer incremental validation:

1. **Validate demand** — Are Vulkan users GPU-bound on RDNA, or are CPU/STT/embeddings the bottleneck?
2. **Dev-only ROCm** — Add `rocm` to `install_llama_cpp_variant.sh` for source installs only; document in install-from-source guide.
3. **Benchmark** — Compare Vulkan vs HIP on 2–3 representative systems (e.g. RX 7900, RX 6800, one APU) using real Qube models and layer settings.
4. **Package only if justified** — Add `rocm` AppImage/`.deb` when perf gain clearly outweighs CI size, support, and gfx-matrix costs.

Until then, **investing in Vulkan defaults, layer-cap tuning, and AMD-specific help** likely yields better ROI for most AMD Linux users.

---

## References

- Linux install variants: [install-linux.md](user/install-linux.md)
- llama.cpp HIP build notes: [llama.cpp build docs (HIP)](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- AMD ROCm llama.cpp install guide: [ROCm llama.cpp installation](https://rocm.docs.amd.com/projects/llama-cpp/en/latest/install/llama-cpp-install.html)
- Prior art in repo: Keith’s CUDA `.deb` xz compression fix (PR #45) — same GitHub 2 GiB constraint would apply to ROCm artifacts
- Related internal doc: [release versioning quick reference](release_versioning_quick_reference.md)
