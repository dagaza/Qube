# WinGet CUDA / Microsoft Defender investigation

**Audience:** Maintainers  
**Related:** [`winget_wdsi_submission.md`](winget_wdsi_submission.md), [`winget/README.md`](../winget/README.md)

This document records **what we know**, **what we suspect**, and **what to test** for `dagaza.Qube.CUDA` failing Microsoft WinGet **Installation Validation** (step 08) with `Validation-Defender-Error`.

---

## Confirmed findings (v1.3.50, 2026-09-07)

| Stage | Artifact | Result |
|-------|----------|--------|
| WinGet step **07** Installers Scan | `Qube-1.3.50-cuda-Setup.exe` | **Pass** (static installer scan) |
| WinGet step **08** Installation Validation | Launches installed **`Qube.exe`** | **`Validation-Defender-Error`** |
| CPU/Vulkan Setup (control) | Installer wrappers | Pass WinGet validation |
| CPU/Vulkan Setup on VirusTotal | Setup wrappers | **0 / 75** malicious; Microsoft undetected |
| CUDA **`Qube.exe`** (installed) | Post-install binary | Microsoft: **`Trojan:Win32/Wacatac.B!ml`** (VirusTotal + Windows test VM) |

**WinGet PR:** https://github.com/microsoft/winget-pkgs/pull/429353

### Key hashes (1.3.50)

| File | SHA-256 |
|------|---------|
| `Qube-1.3.50-cuda-Setup.exe` | `4E9F8B49D41AF0EF4667C4371129EC1D9DE4ACB2745FA1A9D69B7065D11CCF08` |
| `Qube.exe` (CUDA install, `%LOCALAPPDATA%\Programs\Qube\`) | `C653FEEB6DE980E92D47F7898C395CD1693A1EDFE44D2AD7A0F27D8111766506` |
| `ggml-cuda.dll` (`_internal\llama_cpp\lib\`) | `81924BB0F75EAF45029114D3F34D32CA7936FD4E9DBD06630497A6A3A39861E8` (886 MiB) |

**WDSI detection name to use:** `Trojan:Win32/Wacatac.B!ml`

### VirusTotal corroboration (`Qube.exe`, 2026-09-07)

| Field | Value |
|-------|--------|
| **VT report** | https://www.virustotal.com/gui/file/c653feeb6de980e92d47f7898c395cd1693a1edfe44d2ad7a0f27d8111766506 |
| **Score** | 4 / 75 malicious (67 undetected at query time) |
| **Popular threat label** | `dropper.` (VT clustering — **not** evidence of dropper behavior) |

| Engine | Detection |
|--------|-----------|
| **Microsoft** | **`Trojan:Win32/Wacatac.B!ml`** |
| Bkav Pro | `W32.Malware.EDE6264E` |
| APEX | `Malicious` |
| Zillya | `Dropper.Agent.Win32.746397` |

The **`!ml`** suffix indicates a **machine-learning / heuristic** classification — common for unsigned PyInstaller apps and often addressable via WDSI false-positive submission.

### `ggml-cuda.dll` (context, not the WDSI upload target)

| Field | Value |
|-------|--------|
| **Size** | 929,903,616 bytes (~886 MiB) — normal for cu124 wheel with multi-arch CUDA kernels |
| **SHA-256** | `81924BB0F75EAF45029114D3F34D32CA7936FD4E9DBD06630497A6A3A39861E8` |
| **VT hash lookup** | 404 (not indexed; too large for VT upload, 650 MiB API cap) |

| Platform | Upload limit | `ggml-cuda.dll` |
|----------|--------------|-----------------|
| **WDSI portal** | ~50 MiB | **Cannot upload** — cite hash + provenance in Additional information |
| **VirusTotal** | 650 MiB (large-file API) | **Cannot upload** (886 MiB) |
| **VT hash search** | — | Works; no public report yet |

WinGet step 08 failure aligns with **`Qube.exe`** (VT-confirmed Microsoft detection), not necessarily with `ggml-cuda.dll` loading during validation. The DLL’s size explains why the CUDA **installer is ~1.04 GiB** while CPU/Vulkan stay ~200 MiB.

**Important:** Step 08 launches **`Qube.exe`**, not the Setup wrapper. WDSI submission should prioritize **installed `Qube.exe`** (~31 MiB). Do **not** ZIP `_internal` for WDSI — cite `ggml-cuda.dll` hash in Additional information and offer a secure direct link only if analysts request the binary ([`winget_wdsi_submission.md`](winget_wdsi_submission.md)).

---

## What is *not* the problem

| Area | Verdict |
|------|---------|
| WinGet manifest / SHA-256 mismatch | Manifests hash the same GitHub Release installers CI publishes |
| GitHub vs WinGet distribution | Same `.exe` artifacts; WinGet is a catalog pointer |
| CPU / Vulkan builds | Same PyInstaller + Inno pipeline; pass validation |
| Install-grace / shell bootstrap (removed in `c280e3b`) | Was a UX workaround, not a Defender fix |

---

## CUDA-specific packaging (repo facts)

CUDA builds add only these **whitelisted** NVIDIA wheel DLLs next to llama.cpp libs (`core/cuda_wheel_bundle.py`, `scripts/stage_cuda_runtime_libs.py`):

- `cudart64_12.dll`
- `cublas64_12.dll`
- `cublasLt64_12.dll`

Plus **`ggml-cuda.dll`** from the official **`llama-cpp-python`** `cu124` wheel (`scripts/windows/install_llama_cpp_variant.ps1`).

Build chain:

```
llama-cpp-python (cu124 wheel) + nvidia-cuda-runtime-cu12 + nvidia-cublas-cu12
        → PyInstaller (qube.spec) → stage_cuda_runtime_libs.py → Inno Setup → GitHub Release → WinGet
```

`llama_cpp` import is **deferred** until explicit model load (`core/llama_cpp_import.py`, empty `pyi_rth_llama_cpp.py`) — helps startup behavior but **does not remove** large CUDA DLLs from the installed tree.

---

## Hypothesis ranking (test before changing production)

| Rank | Suspect | Rationale | Test |
|------|---------|-----------|------|
| 1 | **Installed `Qube.exe` (CUDA)** | Confirmed `Wacatac.B!ml`; step 08 launches it | WDSI submit `Qube.exe`; re-scan after fixes |
| 2 | **`ggml-cuda.dll`** | App-specific 886 MiB CUDA backend; not NVIDIA-signed | Defender scan extracted DLL; compare CPU `Qube.exe` |
| 3 | **`cublasLt64_12.dll`** | Very large; other AVs flag CUDA Lt heuristically | Remove from test copy only; rescan bundle |
| 4 | **PyInstaller + CUDA payload size** | Same packer as CPU, different native payload | Compare installer sizes / scan `Qube.exe` only |
| 5 | **Runtime behavior at launch** | Mic, ONNX wake word, bootstrap downloads | Launch with subsets disabled (diagnostic) |

Do **not** ship without `cublasLt64_12.dll` unless CUDA inference is verified — it is required by `verify_windows_cuda_bundle.ps1` today.

---

## Controlled experiments (maintainer VM)

### Experiment A — Confirm flagged file (done for 1.3.50)

Install CUDA setup → launch once → record Defender threat name and quarantined path.

### Experiment B — Component isolation scan

| ID | Artifact | Action |
|----|----------|--------|
| A | CPU `Qube.exe` | Baseline |
| B | Vulkan `Qube.exe` | GPU variant control |
| C | CUDA `_internal\llama_cpp\lib\` minus `cublasLt64_12.dll` | Diagnostic delete after install |
| D | Full CUDA install | Production layout |

Scan with **Microsoft Defender** and note **Microsoft** column on VirusTotal.

### Experiment C — CPU vs CUDA `Qube.exe` only

Extract `Qube.exe` from each variant install; compare VT Microsoft results.

---

## Recommended action sequence

```
1. WDSI submit Qube.exe (CUDA 1.3.50) — detection Trojan:Win32/Wacatac.B!ml     [see winget_wdsi_submission.md]
2. Comment on winget-pkgs PR #429353 with Submission ID + @wingetbot run
3. Ship 1.3.51+ with bootstrap fix; publish SHA256SUMS.txt on releases
4. If still failing: WDSI submit new Qube.exe hash; optional ggml-cuda.dll hash in text
5. Parallel: SignPath Foundation application; plan cloud signing (not exportable PFX on runner)
6. Defer EV cert purchase until WDSI + signing path are clear
```

---

## Release provenance improvements (implemented / planned)

| Item | Status |
|------|--------|
| **SHA256 for all Windows installers** in GitHub Release notes | Implemented (`write_windows_installer_checksums.py` + release CI) |
| **`SHA256SUMS.txt`** on each GitHub Release | Implemented |
| **CUDA provenance JSON** (pip versions + bundled DLL hashes) | Planned follow-up |
| **Authenticode signing** via SignPath / cloud HSM | Planned; see [`releasing.md`](releasing.md) |

---

## SmartScreen vs Defender

| Symptom | Type | Signing helps? |
|---------|------|----------------|
| “Windows protected your PC” / SmartScreen | Reputation | Yes, over time |
| `Trojan:Win32/...` / step 08 `Validation-Defender-Error` | Antivirus / ML | Partially; **WDSI required** |

Do not treat code signing as a substitute for WDSI false-positive submission.

---

## Code signing note (2026)

Existing CI supports **exportable PFX** signing when `ENABLE_CODE_SIGNING=true`. For **new** public code-signing certificates, prefer **SignPath Foundation** (free for open source) or **cloud/HSM signing** (Azure Trusted Signing, SSL.com eSigner) rather than storing PFX on GitHub Actions runners long term.

---

## External review synthesis (2026-09)

An external repository review correctly identified:

- CUDA-only delta is NVIDIA + `ggml-cuda.dll` stack
- Diagnose before buying certificates
- Publish installer hashes and provenance
- WinGet architecture is sound

We **partially confirm** `cublasLt64_12.dll` as a suspect but **1.3.50 evidence** points first to **`Qube.exe`** / **`Wacatac.B!ml`**, with **`ggml-cuda.dll`** as the likely heavy native component inside that binary’s bundle context.

---

## References

- WDSI portal: https://www.microsoft.com/en-us/wdsi/filesubmission
- Submission playbook: [`winget_wdsi_submission.md`](winget_wdsi_submission.md)
- Bundle verifier: `scripts/release/verify_windows_cuda_bundle.ps1`
- Release checksum script: `scripts/release/write_windows_installer_checksums.py`
