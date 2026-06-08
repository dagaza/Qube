# Releasing Qube

This document describes how to cut a Windows release and publish it to WinGet and Chocolatey.

## Prerequisites

- Maintainer access to `dagaza/Qube`
- Git tag matching semver: `vMAJOR.MINOR.PATCH` (example: `v1.0.1`)
- Updated [`CHANGELOG.md`](../CHANGELOG.md) entry for the release

## Version flow (single chain)

| Role | Location |
|------|----------|
| **Release trigger** | Git tag `vX.Y.Z` (only tags matching `v*` start the workflow) |
| **Runtime / app** | `core/__version__.py` — imported by `main.py` and embedded in the build |
| **Packaging metadata** | `pyproject.toml` `version` — kept in sync with `__version__.py` |
| **Human release notes** | `CHANGELOG.md` — you edit; not auto-generated |
| **Sync tool** | `scripts/set_version.py` — writes `__version__.py` + `pyproject.toml` |
| **Maintainer helper** | `scripts/prepare_release.py` — runs sync + CHANGELOG check + prints tag commands |

CI does **not** guess the version from files on `main`. It strips the `v` prefix from the tag and runs `set_version.py` before PyInstaller, so the installer, embedded app version, WinGet manifests, and Chocolatey package always match the tag you pushed.

## Pre-release checklist

1. Ensure [`main`](https://github.com/dagaza/Qube) is green (PR CI workflow).
2. Run focused smoke tests locally:

   ```powershell
   pytest tests/test_memory_qa_smoke.py -q
   pytest tests/ -m "not packaging" -q
   ```

3. Optional local packaging parity with CI (uses `core/__version__.py` if `-Version` omitted):

   ```powershell
   python scripts/prepare_release.py 1.0.1
   .\scripts\build_windows.ps1
   ```

## Cut a release

1. Merge release changes to `main`.
2. Prepare version files and CHANGELOG (recommended before tagging):

   ```powershell
   # Edit CHANGELOG.md: move [Unreleased] notes into ## [1.0.1] - YYYY-MM-DD
   python scripts/prepare_release.py 1.0.1
   git add core/__version__.py pyproject.toml CHANGELOG.md
   git commit -m "Release 1.0.1"
   ```

3. Create and push the tag (must match the prepared version):

   ```powershell
   git tag v1.0.1
   git push origin main
   git push origin v1.0.1
   ```

4. GitHub Actions **Build & Release** workflow will:
   - Sync version into `core/__version__.py` and `pyproject.toml`
   - Run pytest
   - Build PyInstaller output and Inno Setup installer
   - Smoke-test dist EXE, silent install, installed EXE launch, and uninstall
   - Compute SHA256 and render WinGet manifests and Chocolatey package
   - Create a GitHub Release with `Qube-<version>-Setup.exe`
   - Smoke-test Chocolatey install/uninstall (after release is published)
   - Optionally push the Chocolatey package to community.chocolatey.org

5. Verify the release asset and SHA256 in the release notes.

## WinGet

### First catalog entry (one-time manual PR)

Follow [`winget/README.md`](../winget/README.md).

### Automated updates

Set repository **variable** `WINGET_AUTO_SUBMIT=true` and secret `WINGET_SUBMIT_TOKEN` (PAT with fork/PR access to `microsoft/winget-pkgs`).

After each tagged release, the workflow opens a WinGet Community PR automatically.

Users install or upgrade with:

```powershell
winget install -e --id dagaza.Qube
winget upgrade -e --id dagaza.Qube
```

## Chocolatey

### First catalog entry (one-time manual submission)

Follow [`chocolatey/README.md`](../chocolatey/README.md).

### Automated updates

Set repository **variable** `CHOCOLATEY_AUTO_PUSH=true` and secret `CHOCOLATEY_API_KEY` (push-only API key from chocolatey.org).

After each tagged release, the workflow pushes `qube.<version>.nupkg` automatically once the GitHub Release is live and the Chocolatey smoke test passes.

Users install or upgrade with:

```powershell
choco install qube
choco upgrade qube
```

## Code signing (optional)

SmartScreen trust improves when binaries are Authenticode-signed.

1. Obtain a standard or EV code-signing certificate (PFX).
2. Add repository secrets:
   - `WINDOWS_CERT_PFX_BASE64`
   - `WINDOWS_CERT_PASSWORD`
3. Set repository **variable** `ENABLE_CODE_SIGNING=true`.

The release workflow signs `dist\Qube\Qube.exe` and the Inno Setup installer when enabled.

## Rollback

1. Mark the bad GitHub Release as **Pre-release** or delete the release asset if necessary.
2. Revert or submit a corrective WinGet manifest pointing to the previous `InstallerUrl`.
3. Push a corrective Chocolatey package version if the bad nupkg was published.
4. Tag a patch release (`v1.0.2`) rather than rewriting history on `main`.

## Artifact naming

| Artifact | Pattern |
|----------|---------|
| Git tag | `v1.0.1` |
| Installer | `Qube-1.0.1-Setup.exe` |
| WinGet folder | `manifests/d/dagaza/Qube/1.0.1/` |
| Chocolatey nupkg | `qube.1.0.1.nupkg` |

## Version source of truth

- **At release time:** the git tag `vX.Y.Z` is canonical; CI derives `X.Y.Z` and runs `set_version.py`.
- **In the running app:** `core.__version__` (via `from core.__version__ import __version__` in `main.py`).
- **Locally:** run `python scripts/prepare_release.py X.Y.Z` so `__version__.py`, `pyproject.toml`, and CHANGELOG align before you tag.

Dry-run packaging may use any version (e.g. `prepare_release.py 9.9.9`); revert with `git checkout -- core/__version__.py pyproject.toml` afterward.
