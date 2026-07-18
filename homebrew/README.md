# Homebrew Cask

CI renders a versioned cask into `homebrew/out/<version>/qube.rb` during
release and (when enabled) commits it to the tap repository.

Homebrew Cask distributes the prebuilt, signed, notarized `Qube.app` DMGs from
the GitHub Release — it does **not** build from source. The DMGs must therefore
be signed and notarized (see [`../docs/releasing.md`](../docs/releasing.md));
Gatekeeper and `brew audit` reject unsigned apps.

## Prerequisite: the tap repository

Create a tap repo named `dagaza/homebrew-qube` with a `Casks/` directory. The
`homebrew` release job commits `Casks/qube.rb` there, so users can run:

```bash
brew install --cask dagaza/qube/qube
# or, after `brew tap dagaza/qube`:
brew install --cask qube
```

## First-time setup (manual)

1. Tag a release and wait for signed, notarized DMGs to attach to the Release.
2. Render the cask locally (digests are the DMG `shasum -a 256` values):

   ```bash
   V=1.0.1
   python scripts/render_homebrew_cask.py \
     --version "$V" \
     --sha256-arm64  "$(shasum -a 256 Qube-$V-arm64.dmg  | awk '{print $1}')" \
     --sha256-x86_64 "$(shasum -a 256 Qube-$V-x86_64.dmg | awk '{print $1}')"
   ```

3. Copy `homebrew/out/$V/qube.rb` into `dagaza/homebrew-qube/Casks/qube.rb`,
   commit, and push.
4. Audit before publishing:

   ```bash
   brew tap dagaza/qube
   brew audit --cask --online qube
   brew style qube
   ```

## Automated updates

Set repository variable `HOMEBREW_AUTO_SUBMIT=true` and secret
`HOMEBREW_TAP_TOKEN` (a fine-grained PAT with contents:write on
`dagaza/homebrew-qube` only). After each tagged release with signing enabled,
the workflow renders the cask, audits it, and commits the bump to the tap.

| Setting | Value |
|---------|-------|
| Variable `HOMEBREW_AUTO_SUBMIT` | `true` |
| Secret `HOMEBREW_TAP_TOKEN` | PAT scoped to `dagaza/homebrew-qube` |

## homebrew-cask core (future)

Once the app is signed, notarized, and has notability, submit the cask to
`homebrew/homebrew-cask` so users can `brew install --cask qube` without adding
the tap.

## Template files

The file under `homebrew/templates/` documents the cask shape. Release builds
use `scripts/render_homebrew_cask.py` instead of editing it directly.
