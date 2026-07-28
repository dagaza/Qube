# Chocolatey packages

CI renders versioned package files into `chocolatey/out/<version>/` during release — one package per Windows variant:

| Package ID | Installer |
|------------|-----------|
| `qube` | `Qube-<version>-Setup.exe` (CPU) |
| `qube-vulkan` | `Qube-<version>-vulkan-Setup.exe` |
| `qube-cuda` | `Qube-<version>-cuda-Setup.exe` |

## First-time catalog submission (manual)

1. Tag a release (`v1.0.0`) and wait for the GitHub Actions release workflow.
2. Download the `chocolatey-package-*` artifact or build locally:

   ```powershell
   $v = "1.0.0"
   $cpu = (Get-FileHash "installer\output\Qube-$v-Setup.exe" -Algorithm SHA256).Hash
   $vulkan = (Get-FileHash "installer\output\Qube-$v-vulkan-Setup.exe" -Algorithm SHA256).Hash
   $cuda = (Get-FileHash "installer\output\Qube-$v-cuda-Setup.exe" -Algorithm SHA256).Hash
   python scripts/render_chocolatey_package.py `
     --version $v `
     --cpu-sha256 $cpu `
     --vulkan-sha256 $vulkan `
     --cuda-sha256 $cuda
   foreach ($pkg in @("qube", "qube-vulkan", "qube-cuda")) {
     choco pack "chocolatey\out\$v\$pkg\$pkg.nuspec" --output-directory "chocolatey\out\$v\$pkg"
   }
   ```

3. Create an account at [community.chocolatey.org](https://community.chocolatey.org) and register each package name (`qube` first; then `qube-vulkan` and `qube-cuda`).
4. Push the first version of each package (enters moderation queue):

   ```powershell
   choco push "chocolatey\out\$v\qube\qube.$v.nupkg" --source https://push.chocolatey.org/ --api-key YOUR_API_KEY
   choco push "chocolatey\out\$v\qube-vulkan\qube-vulkan.$v.nupkg" --source https://push.chocolatey.org/ --api-key YOUR_API_KEY
   choco push "chocolatey\out\$v\qube-cuda\qube-cuda.$v.nupkg" --source https://push.chocolatey.org/ --api-key YOUR_API_KEY
   ```

5. After moderation approval, users can run:

   ```powershell
   choco install qube
   choco install qube-vulkan
   choco install qube-cuda
   ```

## Automated updates

Set repository variables:

| Variable | Value |
|----------|-------|
| `CHOCOLATEY_AUTO_PUSH` | `true` |

Set repository secret:

| Secret | Purpose |
|--------|---------|
| `CHOCOLATEY_API_KEY` | Push-only API key from chocolatey.org |

The release workflow pushes all three packages after the GitHub Release is published and the Chocolatey install smoke test passes.

## Local testing

```powershell
$v = "1.0.0"
$cpu = (Get-FileHash "installer\output\Qube-$v-Setup.exe" -Algorithm SHA256).Hash
python scripts/render_chocolatey_package.py --version $v --cpu-sha256 $cpu --vulkan-sha256 $cpu --cuda-sha256 $cpu
choco pack "chocolatey\out\$v\qube\qube.nuspec" --output-directory "chocolatey\out\$v\qube"
choco install qube -y -s "chocolatey\out\$v\qube" --version=$v
choco uninstall qube -y
```

The install script downloads the matching `Qube-<version>-*.exe` from GitHub Releases, so that release asset must exist before `choco install` succeeds.

## Template files

The files under `chocolatey/templates/` document the package shape. Release builds use `scripts/render_chocolatey_package.py` instead of editing these directly.
