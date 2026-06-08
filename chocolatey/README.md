# Chocolatey package

CI renders versioned package files into `chocolatey/out/<version>/` during release.

## First-time catalog submission (manual)

1. Tag a release (`v1.0.0`) and wait for the GitHub Actions release workflow.
2. Download the `chocolatey-package-*` artifact or build locally:

   ```powershell
   $v = "1.0.0"
   $hash = (Get-FileHash "installer\output\Qube-$v-Setup.exe" -Algorithm SHA256).Hash
   python scripts/render_chocolatey_package.py --version $v --sha256 $hash
   choco pack "chocolatey\out\$v\qube.nuspec" --output-directory "chocolatey\out\$v"
   ```

3. Create an account at [community.chocolatey.org](https://community.chocolatey.org) and register the `qube` package name.
4. Push the first version (enters moderation queue):

   ```powershell
   choco push "chocolatey\out\$v\qube.$v.nupkg" --source https://push.chocolatey.org/ --api-key YOUR_API_KEY
   ```

5. After moderation approval, users can run:

   ```powershell
   choco install qube
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

The release workflow runs `choco push` after the GitHub Release is published and the Chocolatey install smoke test passes.

## Local testing

```powershell
$v = "1.0.0"
$hash = (Get-FileHash "installer\output\Qube-$v-Setup.exe" -Algorithm SHA256).Hash
python scripts/render_chocolatey_package.py --version $v --sha256 $hash
choco pack "chocolatey\out\$v\qube.nuspec" --output-directory "chocolatey\out\$v"
choco install qube -y -s "chocolatey\out\$v" --version=$v
choco uninstall qube -y
```

The install script downloads `Qube-<version>-Setup.exe` from GitHub Releases, so that release asset must exist before `choco install` succeeds.

## Template files

The files under `chocolatey/templates/` document the package shape. Release builds use `scripts/render_chocolatey_package.py` instead of editing these directly.
