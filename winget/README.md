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

## Template files

The files under `winget/templates/` document the manifest shape. Release builds use `scripts/render_winget_manifests.py` instead of editing these directly.
