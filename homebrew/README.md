# Homebrew Cask (custom tap)

CI renders a versioned cask into `homebrew/out/<version>/qube.rb` during
release and (when enabled) commits it to the **`dagaza/homebrew-qube`** tap.

Homebrew distributes the prebuilt `Qube.app` DMGs from GitHub Releases — it
does **not** build from source.

## Unsigned distribution (no Apple Developer account)

Qube does **not** require code signing or notarization for the custom tap.
DMGs from release CI are **unsigned** when `ENABLE_MACOS_SIGNING` is not set.

Users install via the tap, then approve the app in **System Settings → Privacy &
Security** (or run `xattr -dr com.apple.quarantine "/Applications/Qube.app"`).
The rendered cask includes these instructions in `caveats`.

This path is for **your own tap** (`dagaza/homebrew-qube`). Submitting to
**homebrew/homebrew-cask** later would require signing/notarization and
notability review.

## Prerequisite: the tap repository

Create a GitHub repo **`dagaza/homebrew-qube`** with a `Casks/` directory (empty
`Casks/.gitkeep` is fine). The release workflow commits `Casks/qube.rb` there
when `HOMEBREW_AUTO_SUBMIT=true`.

Users install with:

```bash
brew tap dagaza/qube
brew install --cask qube
```

Or without adding the tap to the default list:

```bash
brew install --cask dagaza/qube/qube
```

## First-time setup (manual)

1. Tag a release and wait for DMGs on the GitHub Release (`Qube-<version>-arm64.dmg`
   and `Qube-<version>-x86_64.dmg`).
2. Render the cask locally:

   ```bash
   V=1.2.4
   python scripts/render_homebrew_cask.py \
     --version "$V" \
     --sha256-arm64  "$(shasum -a 256 Qube-$V-arm64.dmg  | awk '{print $1}')" \
     --sha256-x86_64 "$(shasum -a 256 Qube-$V-x86_64.dmg | awk '{print $1}')"
   ```

3. Copy `homebrew/out/$V/qube.rb` → `dagaza/homebrew-qube/Casks/qube.rb`, commit, push.
4. Validate style (unsigned casks skip strict Gatekeeper audit):

   ```bash
   brew tap dagaza/qube
   brew style "$(brew --repository)/Library/Taps/dagaza/homebrew-qube/Casks/qube.rb"
   ```

## Automated updates

Set repository **variable** and **secret** on `dagaza/Qube`:

| Setting | Value |
|---------|-------|
| Variable `HOMEBREW_AUTO_SUBMIT` | `true` |
| Secret `HOMEBREW_TAP_TOKEN` | Fine-grained PAT with **contents: write** on `dagaza/homebrew-qube` only |

After each tagged release, the workflow renders the cask, runs `brew style`, and
pushes the bump to the tap via `scripts/macos/bump_tap.sh`.

Signing/notarization (`ENABLE_MACOS_SIGNING`) is **optional** — the tap workflow
runs for unsigned DMGs as well.

## Template files

The file under `homebrew/templates/` documents the cask shape. Release builds
use `scripts/render_homebrew_cask.py` instead of editing it directly.
