# WinGet CUDA validation & WDSI false-positive submission

**Audience:** Maintainers (Dan Zadobrischi / dagaza)  
**Related:** [`winget/README.md`](../winget/README.md), [`docs/releasing.md`](releasing.md)

Microsoft **WinGet Installation Validation** (step 08) for **`dagaza.Qube.CUDA`** can fail with **`Validation-Defender-Error`** even when step 07 (installer scan) passes. That usually means Defender flagged behavior during the **post-install app launch**, not a manifest typo.

This document is the submission playbook for [Microsoft Security Intelligence (WDSI)](https://www.microsoft.com/en-us/wdsi/filesubmission) and follow-up on the blocked **`winget-pkgs`** PR.

---

## Publisher & contact (fixed fields)

| Field | Value |
|-------|--------|
| **Publisher / developer** | Dan Zadobrischi |
| **Public alias / GitHub org** | [dagaza](https://github.com/dagaza) |
| **Product** | Qube (CUDA build) — package ID `dagaza.Qube.CUDA` |
| **Website** | https://www.qubeapp.eu |
| **Source** | https://github.com/dagaza/Qube (MIT) |
| **WDSI contact email** | dan@zadobrischi.com |
| **Code signing (current)** | Unsigned |
| **Code signing (goal)** | Authenticode via release CI — see [Code signing setup](#code-signing-setup-optional-but-recommended) |

---

## 1. Where to get the Defender detection name

WinGet often reports only the **check failure label** `Validation-Defender-Error`, not the underlying Defender threat string (e.g. `Trojan:Win32/Wacatac.B!ml`). Use **all** sources below; put the best available name in the WDSI **Detection name** field and explain the rest in **Additional information**.

### A. WinGet PR check artifact (try first)

1. Open the **`microsoft/winget-pkgs`** pull request for **`dagaza.Qube.CUDA`** (the PR where validation failed — see [Which WinGet PR?](#2-which-winget-pr-to-reference)).
2. Open **Checks** → **Azure Pipelines** (or **WinGet validation** job) → failed run.
3. Download artifacts named like **`InstallationVerification_Logs`** or **`InstallationVerification_Result.json`**.
4. Search the JSON / logs for:
   - `Validation-Defender-Error`
   - `Threat`, `Detection`, `Malware`, `Defender`
   - Any `Trojan:`, `Program:`, or `PUA:` string

If the artifact only says `Validation-Defender-Error` with no threat name, that is normal — use the label in **Additional information** and obtain the threat name from (B) or (C).

### B. Reproduce on a clean Windows 11 VM (most reliable for threat name)

```powershell
# Install the same CUDA setup that failed WinGet validation, then launch Qube once.

Get-MpThreatDetection | Format-List *
Get-MpThreat | Format-List ThreatName, Resources, InitialDetectionTime

# Or Windows Security → Protection history → filter blocked items
```

Note the **Threat name** column (e.g. `Trojan:Win32/Wacatac.B!ml`).

### C. VirusTotal (optional cross-check)

Upload **`Qube.exe`** from `%LOCALAPPDATA%\Programs\Qube\` (not the whole `_internal` tree). Read the **Microsoft** engine result line on the **Detection** tab.

### D. If you still have no specific threat name

Submit anyway:

- **Detection name:** `Validation-Defender-Error` (or `Unknown — WinGet step 08 only`)
- **Additional information:** State clearly that WinGet step 07 passes, step 08 fails with `Validation-Defender-Error`, and you are requesting classification of the legitimate CUDA inference build.

---

## 2. Which WinGet PR to reference

| Situation | Which PR / version |
|-----------|-------------------|
| **WDSI submission (now)** | Submit the **exact binaries that failed validation** — today that is **`1.3.50`** (`Qube-1.3.50-cuda-Setup.exe` and installed `Qube.exe`). |
| **WDSI after `1.3.51` ships** | If validation still fails on the new build, submit **`1.3.51`** hashes as a **new** WDSI submission (or note in follow-up that the product family is unchanged). |
| **Comment on `winget-pkgs` PR** | Use the PR that is **currently blocked** (likely the **`1.3.50`** CUDA catalog PR). Paste the WDSI **Submission ID** there and `@wingetbot run`. |
| **Future `1.3.51` PR** | If Microsoft clears **`1.3.50`** via WDSI, the **`1.3.51`** PR may pass without a new submission; if not, link the same Submission ID and note that **`1.3.51`** includes the bootstrap fix but the same signed/unsigned CUDA stack. |

**Rule:** WDSI analyzes **files** (SHA-256). WinGet PR comments reference **the PR under review**. They are related but not the same: clear the binary with WDSI, then re-run validation on the open PR (or the next version PR).

**PR URL (fill when submitting):** https://github.com/microsoft/winget-pkgs/pull/429353

---

## 3. Files to submit to WDSI

Submit as **Software developer** → **Incorrectly detected as malware/malicious**.

WinGet step **08 launches the installed app**, so prioritize **`Qube.exe`**. Step **07** scans the installer, so include the **Setup.exe** as a second submission if needed.

| Priority | File | Path / source |
|----------|------|----------------|
| **1** | `Qube.exe` | `%LOCALAPPDATA%\Programs\Qube\Qube.exe` after installing the CUDA setup |
| **2** | `Qube-<version>-cuda-Setup.exe` | GitHub Release asset |

**Do not** upload the full PyInstaller `_internal` folder or a large ZIP (50 MB limit; deprioritized by analysts).

### Hashes (fill per release)

```powershell
Get-FileHash "$env:LOCALAPPDATA\Programs\Qube\Qube.exe" -Algorithm SHA256
Get-FileHash ".\Qube-1.3.50-cuda-Setup.exe" -Algorithm SHA256
```

| File | Version | SHA-256 |
|------|---------|---------|
| `Qube.exe` (installed CUDA) | 1.3.50 | `C653FEEB6DE980E92D47F7898C395CD1693A1EDFE44D2AD7A0F27D8111766506` |
| `Qube-1.3.50-cuda-Setup.exe` | 1.3.50 | `4E9F8B49D41AF0EF4667C4371129EC1D9DE4ACB2745FA1A9D69B7065D11CCF08` |
| `ggml-cuda.dll` (CUDA `_internal`) | 1.3.50 | `81924BB0F75EAF45029114D3F34D32CA7936FD4E9DBD06630497A6A3A39861E8` (886 MiB / 929,903,616 B) |
| `Qube.exe` (installed CUDA) | 1.3.51 | `[FILL after release]` |
| `Qube-1.3.51-cuda-Setup.exe` | 1.3.51 | `[FILL after release]` |

---

## 4. WDSI form — field-by-field

Portal: https://www.microsoft.com/en-us/wdsi/filesubmission

| Field | Value |
|-------|--------|
| Submit as | **Software developer** |
| Company name | **Dan Zadobrischi** (GitHub: dagaza) |
| Product | **Microsoft Defender Antivirus (Windows 11)** |
| Belief | **Incorrectly detected as malware/malicious** |
| Detection name | `Trojan:Win32/Wacatac.B!ml` (confirmed VT + Windows test VM, 2026-09-07) |
| Definition version | `[Optional — output of Get-MpComputerStatus]` |
| File | See [§3](#3-files-to-submit-to-wdsi) |
| Priority | **Medium** |
| Additional information | See [§5](#5-additional-information-final-text) |

**After submit:** Save the **Submission ID** and a **screenshot** of the success page immediately (WDSI history/details pages sometimes fail later).

---

## 5. Additional information (final text)

Copy into the WDSI **Additional information** box (≤1900 characters). Update version and hashes before submit.

```
Product: Qube (CUDA build) — WinGet ID dagaza.Qube.CUDA
Publisher: Dan Zadobrischi (GitHub: dagaza)
Website: https://www.qubeapp.eu
Source: https://github.com/dagaza/Qube (MIT, open source)
Release: https://github.com/dagaza/Qube/releases/tag/v1.3.50

File submitted: Qube.exe (NVIDIA CUDA GPU build), SHA-256: C653FEEB6DE980E92D47F7898C395CD1693A1EDFE44D2AD7A0F27D8111766506.
Installer (same version): Qube-1.3.50-cuda-Setup.exe, SHA-256: 4E9F8B49D41AF0EF4667C4371129EC1D9DE4ACB2745FA1A9D69B7065D11CCF08.
Optional component (hash only — file is 886 MiB, exceeds WDSI 50 MiB upload cap):
ggml-cuda.dll, SHA-256: 81924BB0F75EAF45029114D3F34D32CA7936FD4E9DBD06630497A6A3A39861E8
(929,903,616 bytes; llama-cpp-python cu124 wheel; available on request if analysts need the binary).
Code signing: Currently unsigned; Authenticode signing planned for future releases.

Qube is a legitimate local-first desktop AI assistant. Users run LLM inference
on their own hardware; chat, document library, and memory stay on-device. The
CUDA variant uses llama.cpp with NVIDIA GPU offload (ggml-cuda.dll and bundled
NVIDIA CUDA 12.4 runtime libraries from the official llama-cpp-python cu124
wheel). There is no C2, credential theft, persistence, or unwanted bundling.

False positive context — WinGet catalog validation:
Package dagaza.Qube.CUDA is submitted to microsoft/winget-pkgs. Step 07
(Installers Scan) passes; step 08 (Installation Validation) fails with
Validation-Defender-Error when the app launches after silent install on a
Defender-enabled VM. Blocked PR: https://github.com/microsoft/winget-pkgs/pull/429353.
Defender threat name: Trojan:Win32/Wacatac.B!ml (VirusTotal + local Windows 11 test, 2026-09-07).

Technical mitigations in product:
- llama_cpp/CUDA DLLs are not loaded at process startup; import is deferred
  until the user loads a GGUF model (core/llama_cpp_import.py).
- First-run bootstrap downloads models from Hugging Face only after user consent.
- CPU and Vulkan Windows builds pass WinGet validation; only CUDA bundles
  NVIDIA inference libraries.

Expected behavior: PyQt6 UI, optional mic for wake word, optional Hugging Face
downloads with consent. ggml-cuda.dll loads only when loading a local model.

Contact: dan@zadobrischi.com — additional builds or build steps available on request.
```

---

## 6. WinGet PR comment (after WDSI submit)

Post on the blocked **`winget-pkgs`** PR:

```
Validation-Defender-Error on step 08 (Installation Validation): Microsoft classifies the
installed CUDA Qube.exe as Trojan:Win32/Wacatac.B!ml (VirusTotal + Windows 11 test VM).
Step 07 (Installers Scan) passes. This is not a manifest or installer defect.

Submitted to Microsoft Security Intelligence (WDSI) as software developer false positive:

- Submission ID: [FILL]
- Publisher: Dan Zadobrischi (dagaza)
- File submitted: Qube.exe (CUDA v1.3.50), SHA-256: C653FEEB6DE980E92D47F7898C395CD1693A1EDFE44D2AD7A0F27D8111766506
- Installer: Qube-1.3.50-cuda-Setup.exe, SHA-256: 4E9F8B49D41AF0EF4667C4371129EC1D9DE4ACB2745FA1A9D69B7065D11CCF08
- Detection: Trojan:Win32/Wacatac.B!ml
- Bundled context (hash only; 886 MiB, not uploaded): ggml-cuda.dll 81924BB0F75EAF45029114D3F34D32CA7936FD4E9DBD06630497A6A3A39861E8
- https://www.qubeapp.eu | https://github.com/dagaza/Qube

Please re-run validation after Microsoft clears the submission.

@wingetbot run
```

---

## 7. Code signing setup (optional but recommended)

Unsigned builds are acceptable for WDSI submission but **Authenticode signing** improves SmartScreen/Defender trust and helps future WinGet validation. The release pipeline already supports signing when configured.

### Steps

1. **Obtain a code-signing certificate (PFX)**
   - Providers: DigiCert, Sectigo, SSL.com, etc.
   - **Standard** OV cert: lower cost; SmartScreen reputation builds over time.
   - **EV** cert: faster SmartScreen trust; often requires hardware token / stricter identity verification.
   - Certificate must support **Authenticode** / **Code Signing** for Windows.

2. **Add GitHub repository secrets** (Settings → Secrets and variables → Actions):
   - `WINDOWS_CERT_PFX_BASE64` — base64 of the `.pfx` file  
     ```powershell
     [Convert]::ToBase64String([IO.File]::ReadAllBytes("codesign.pfx")) | Set-Clipboard
     ```
   - `WINDOWS_CERT_PASSWORD` — PFX export password

3. **Enable signing in CI** — repository **variable** (not secret):
   - `ENABLE_CODE_SIGNING` = `true`

4. **Tag a release** — workflow signs `dist\Qube\Qube.exe` and each `Qube-*-Setup.exe` before upload (see [`.github/workflows/release.yml`](../.github/workflows/release.yml)).

5. **After first signed release**, mention in the next WDSI submission:
   - Signer name (CN on certificate)
   - Thumbprint
   - That binaries are Authenticode-signed with timestamp

**Never commit:** the `.pfx`, password, or base64 cert to the repo (see `.gitignore`).

Full release context: [`docs/releasing.md`](releasing.md) § Code signing (Windows, optional).

---

## 8. Submission log (maintainer checklist)

| Date | Version | File submitted | SHA-256 | WDSI Submission ID | WinGet PR | Status |
|------|---------|----------------|---------|-------------------|-----------|--------|
| | 1.3.50 | Qube.exe | | | | |
| | 1.3.50 | Qube-1.3.50-cuda-Setup.exe | | | | |

**Checklist per attempt:**

- [ ] Download `InstallationVerification_Result.json` from failed WinGet check
- [ ] Record Defender threat name (artifact, VM, or VirusTotal)
- [ ] Compute SHA-256 for `Qube.exe` and Setup.exe
- [ ] WDSI submit `Qube.exe` (Software developer / false positive)
- [ ] Save Submission ID + screenshot
- [ ] Optional: second WDSI submit for Setup.exe
- [ ] Comment on `winget-pkgs` PR with Submission ID + `@wingetbot run`
- [ ] Wait for WDSI resolution (days; “In progress” is normal)
- [ ] Re-run WinGet validation when cleared

---

## 9. Why this doc is public (`docs/`) not `docs/private/`

`docs/private/*` is **gitignored** (only `license_signing.md` is tracked). This playbook contains **no secrets** — only a business contact email used on public submissions. Keeping it in **`docs/`** lets any maintainer follow the same process from git history and links from [`winget/README.md`](../winget/README.md).

If you prefer a local-only copy, duplicate to `docs/private/winget_wdsi_submission.local.md` (ignored by git).
