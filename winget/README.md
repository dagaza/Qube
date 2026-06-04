# WinGet manifests

CI renders versioned split manifests into `winget/out/<version>/` during release.

## First-time catalog submission (manual)

1. Tag a release (`v1.0.0`) and wait for the GitHub Actions release workflow.
2. Download the `winget-manifests-*` artifact or copy `winget/out/<version>/`.
3. Fork [microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs).
4. Copy the three YAML files to `manifests/d/dagaza/Qube/<version>/`.
5. Validate locally: `winget validate --manifest manifests/d/dagaza/Qube/<version>`
6. Open a PR. After merge, users can run:

   ```powershell
   winget install -e --id dagaza.Qube
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

The release workflow runs `wingetcreate update dagaza.Qube --submit` after each tag.

## Template files

The files under `winget/templates/` document the manifest shape. Release builds use `scripts/render_winget_manifests.py` instead of editing these directly.
