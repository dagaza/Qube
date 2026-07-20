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
2. **Documentation pass** — see [Launch documentation guidelines](launch_documentation_guidelines.md) (**Final launch pass**). At minimum before a public launch:
   - [ ] README accurate (features, install paths, screenshots)
   - [ ] [`CHANGELOG.md`](../CHANGELOG.md) — `[Unreleased]` moved into the versioned section
   - [ ] [`docs/user/`](user/README.md) install and requirements match shipping platforms
   - [ ] In-app help corpus regenerated if settings/tools changed (checklist below)
   - [ ] GitHub **Social preview** uploaded ([`assets/social/`](../assets/social/README.md)) if image changed
   - [ ] GitHub **Pages** landing live at `https://dagaza.github.io/Qube/` ([setup](pages.md)) if `docs/index.html` changed
3. Run focused smoke tests locally:

   ```powershell
   pytest tests/test_memory_qa_smoke.py -q
   pytest tests/ -m "not packaging" -q
   ```

4. Optional local packaging parity with CI (uses `core/__version__.py` if `-Version` omitted):

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

## macOS

The `macos-build` job runs on the same `v*` tag trigger and produces a `.app`
bundle (via the shared `qube.spec` `BUNDLE()` block) packaged into a DMG, one
per architecture:

| Runner | Architecture | Artifact |
|--------|--------------|----------|
| `macos-14` | Apple Silicon (arm64) | `Qube-<version>-arm64.dmg` |
| `macos-13` | Intel (x86_64) | `Qube-<version>-x86_64.dmg` |

Both DMGs are attached to the same GitHub Release as the Windows installer.

`llama-cpp-python` is rebuilt with `-DGGML_METAL=on` so Apple GPUs are used for
inference (the Windows CUDA path via `pynvml` is excluded from the macOS bundle).

### Signing and notarization

Signing, notarization, and the DMG smoke test only run when the repository
**variable** `ENABLE_MACOS_SIGNING=true`. Add these repository secrets:

| Secret | Purpose |
|--------|---------|
| `MACOS_CERT_P12_BASE64` | Base64 `.p12` (Developer ID Application cert + key) |
| `MACOS_CERT_PASSWORD` | Export password for the `.p12` |
| `KEYCHAIN_PASSWORD` | Password for the ephemeral CI keychain |
| `MACOS_SIGN_IDENTITY` | `Developer ID Application: dagaza (TEAMID)` |
| `MACOS_NOTARY_APPLE_ID` | Apple ID email for `notarytool` |
| `MACOS_NOTARY_TEAM_ID` | 10-character Team ID |
| `MACOS_NOTARY_PASSWORD` | App-specific password (not the Apple ID password) |

Until `ENABLE_MACOS_SIGNING` is set, the job still builds and uploads an
unsigned DMG so the pipeline can be validated end-to-end. Unsigned DMGs will be
blocked by Gatekeeper on end-user machines and are not suitable for a Homebrew
Cask — enable signing before publishing a cask.

### Homebrew Cask

Homebrew Cask distributes the signed, notarized DMGs. See
[`homebrew/README.md`](../homebrew/README.md) for full setup.

**Prerequisite:** signing must be enabled (`ENABLE_MACOS_SIGNING=true`) so the
DMGs are notarized — Gatekeeper and `brew audit` reject unsigned apps. Create a
tap repo `dagaza/homebrew-qube` with a `Casks/` directory.

**Automated updates:** set repository variable `HOMEBREW_AUTO_SUBMIT=true` and
secret `HOMEBREW_TAP_TOKEN` (fine-grained PAT with contents:write on
`dagaza/homebrew-qube`). After each signed release, the `homebrew` job renders
the cask via `scripts/render_homebrew_cask.py`, runs `brew audit`/`brew style`,
and commits the bump to the tap.

Users install or upgrade with:

```bash
brew install --cask dagaza/qube/qube
brew upgrade --cask qube
```

## Code signing (Windows, optional)

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

## In-app help documentation (`@help`)

Help corpus changes ship with the app bundle under `assets/help/en/`. When a PR touches settings registries, composer tools, or help prose, use this checklist (see also [`docs/in_app_help_knowledge_base.md`](in_app_help_knowledge_base.md) §18):

### PR checklist (help-related changes)

- [ ] Ran `python scripts/generate_help_reference.py`
- [ ] Ran `python scripts/compose_help_corpus.py`
- [ ] Ran `python scripts/validate_help_manifest.py`
- [ ] Ran `python scripts/eval_help_golden.py` (Phase 6 golden retrieval eval)
- [ ] Ran `python scripts/eval_help_production.py` (rag_search retrieval path)
- [ ] Optional: `python scripts/export_help_queries.py` on local logs for quarterly doc review
- [ ] Updated human prose / canonical answers if UX intent changed
- [ ] Golden questions still pass (or updated expectations in `tests/fixtures/help_golden_questions.json`)
- [ ] Bumped `corpus_version` in `assets/help/en/manifest.json` when retrieval content changed

CI runs the first four commands automatically on every PR.

### Quarterly documentation priority (post-`@help` launch)

Production `@help` analytics outrank speculative new pages. Each quarter:

1. Export or review top unanswered / low-confidence `@help` queries (local telemetry + `Qube.Help` logs).
2. Rank candidates by `(query frequency) × (1 − retrieval success) × (frustration proxy)` — frustration = rephrase within two turns or user opens Settings without following the cited path.
3. Promote fixes in cost order: **canonical answer** → **FAQ** → **troubleshooting** → **workflow** → **feature section** update.
4. Add or adjust golden questions for recurring themes; re-run `python scripts/eval_help_golden.py`.
5. Bump `corpus_version` and note the change in `assets/help/en/release/whats-new.md`.

During beta, review the top 10 `@help` queries weekly and patch canonical answers first.
