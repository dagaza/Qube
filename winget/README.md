# WinGet manifests

CI renders versioned split manifests into `winget/out/<version>/` during release — one folder per package ID:

| Package ID | Windows installer |
|------------|-------------------|
| `dagaza.Qube` | `Qube-<version>-Setup.exe` (CPU) |
| `dagaza.Qube.Vulkan` | `Qube-<version>-vulkan-Setup.exe` |
| `dagaza.Qube.CUDA` | `Qube-<version>-cuda-Setup.exe` |

Install **one** variant only; all share user data in `%LOCALAPPDATA%\Qube`.

## First-time catalog submission (manual)

`dagaza.Qube` (CPU) may already be in [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs). GPU packages are **separate IDs** and need a one-time PR each before automated `wingetcreate update` works.

1. Tag a release (`v1.2.5`) and wait for the GitHub Actions release workflow (or render locally — see below).
2. Download the `winget-manifests-*` artifact or copy `winget/out/<version>/`.
3. Fork [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs).
4. For each package folder under `winget/out/<version>/`, copy into winget-pkgs:
   - `dagaza.Qube/` → `manifests/d/dagaza/Qube/<version>/`
   - `dagaza.Qube.Vulkan/` → `manifests/d/dagaza/Qube.Vulkan/<version>/`
   - `dagaza.Qube.CUDA/` → `manifests/d/dagaza/Qube.CUDA/<version>/`
5. Validate locally, e.g.:

   ```powershell
   winget validate --manifest manifests/d/dagaza/Qube/1.2.5
   winget validate --manifest manifests/d/dagaza/Qube.Vulkan/1.2.5
   winget validate --manifest manifests/d/dagaza/Qube.CUDA/1.2.5
   ```

6. Open a PR. After merge, users can run:

   ```powershell
   winget install -e --id dagaza.Qube
   winget install -e --id dagaza.Qube.Vulkan
   winget install -e --id dagaza.Qube.CUDA
   ```

### Render manifests locally

```bash
python scripts/render_winget_manifests.py \
  --version 1.2.5 \
  --cpu-sha256 <sha256> \
  --vulkan-sha256 <sha256> \
  --cuda-sha256 <sha256>
```

## Automated updates

Set repository variables:

| Variable | Value |
|----------|-------|
| `WINGET_AUTO_SUBMIT` | `true` |

Set repository secret:

| Secret | Purpose |
|--------|---------|
| `WINGET_SUBMIT_TOKEN` | GitHub PAT with rights to push to your `winget-pkgs` fork and open PRs |

The release workflow runs `scripts/release/submit_winget_packages.py` after each tag. It submits the rendered split manifests under `winget/out/<version>/` for **dagaza.Qube**, **dagaza.Qube.Vulkan**, and **dagaza.Qube.CUDA** via `wingetcreate submit` (one PR per package ID).

### Catch-up without retagging

```bash
gh workflow run winget-submit.yml -f version=1.2.5
```

Requires the GitHub Release to include all three Windows `.exe` assets.

### WinGet `Validation-Defender-Error` (CUDA)

Microsoft's installation validation runs a silent install and launches the app on a Defender-enabled VM. If step **08. Installation Validation** fails with **`Validation-Defender-Error`** while **07. Installers Scan** passes, Defender flagged behavior during startup — not a manifest typo.

For **`dagaza.Qube.CUDA`**, this usually means CUDA backend DLLs (`ggml-cuda.dll`, bundled NVIDIA runtime libs) were loaded into the process during validation. Qube blocks `llama_cpp` import while **CUDA deferral mode** is active:

- **`QUBE_WINGET_VALIDATION=1`** or **`--winget-validation`** (CI smoke tests) — also skips first-run bootstrap consent and uses a shell install for smoke verification
- **20-minute post-install grace** on packaged CUDA builds (`.qube-install-ts` written by the Inno installer) — defers CUDA/native loads only; **first-run bootstrap consent still appears**

In explicit smoke validation mode the app skips native autoload, blocks `get_llama_class()`, defers GPU/NVML probes, and auto-completes first-run bootstrap with a shell install (no model downloads).

Release CI runs `scripts/release/smoke_installed_cuda.ps1` after building the CUDA installer to verify the process stays up without importing `llama_cpp`.

If validation still fails after a rebuild:

1. Download the validation artifact (`InstallationVerification_Result.json`) from the PR checks when available.
2. Submit `Qube-<version>-cuda-Setup.exe` as a **software developer** false positive at [Microsoft WDSI](https://www.microsoft.com/en-us/wdsi/filesubmission).
3. Comment on the winget-pkgs PR with the submission ID and `@wingetbot run` after clearance.

Enabling Authenticode signing (`ENABLE_CODE_SIGNING` — see [`docs/releasing.md`](../docs/releasing.md)) improves SmartScreen/Defender trust for future releases.

## Template files

The files under `winget/templates/` document the manifest shape. Release builds use `scripts/render_winget_manifests.py` instead of editing these directly.
