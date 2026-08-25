# Uninstall Qube

## Common questions

- How do I uninstall Qube?
- How do I remove Qube and delete my models?
- Where is user data stored after uninstall?
- How do I uninstall on Windows, macOS, or Linux?

## What it is

**Uninstall Qube** covers removing the application and optionally your local data (models, Library indexes, memory, conversations, settings). User data lives in **`~/.qube`** on macOS and Linux, and **`%LOCALAPPDATA%\Qube`** on Windows — it is separate from the app install on Windows and is not always deleted automatically.

Before a full uninstall, create a **state backup** from **Settings → Backup & restore** if you want to keep conversations, Library indexes, memory, and settings. Optionally export a **knowledge pack** from **Settings → Knowledge → Diagnostics** for Knowledge configuration only (credentials are redacted).

## Where to find it

- **Settings → Help → Uninstall Qube** — platform instructions for Windows, macOS, and Linux, plus quick-uninstall buttons on supported packaged installs (macOS `.app`, Linux `.deb`).
- This workflow — searchable via **Library → Qube** or **`@[tool:help]`**.

## Also called

remove Qube, delete Qube, uninstall app, wipe user data, remove package, clean uninstall

## How to…

### Before you uninstall

1. Create a state backup if you need a full restore later (**Settings → Backup & restore → Create backup now**). Optionally export a knowledge pack for Knowledge settings only (**Settings → Knowledge → Diagnostics → Export knowledge pack**).
2. On **Windows**, choose **Exit Qube** from the system tray before uninstalling — closing the window only hides Qube to the tray. Confirm **Qube.exe** is not running in Task Manager. The Inno uninstaller stops Qube automatically when possible.

### Windows

1. **Release installer** — **Exit Qube** from the tray, then **Settings → Apps → Installed apps** → **Qube** → **Uninstall**. Removes **`%LOCALAPPDATA%\Programs\Qube\`** (including **`Qube.exe`** and **`_internal\`**).
2. Or run: **`%LOCALAPPDATA%\Programs\Qube\unins000.exe`**
3. **WinGet:** `winget uninstall -e --id dagaza.Qube`
4. **Chocolatey:** `choco uninstall qube -y`
5. For a **full wipe**, delete user data at **`%LOCALAPPDATA%\Qube\`** (not removed by the installer alone).

### macOS

1. **DMG** — double-click **`Uninstall Qube.app`** next to **`Qube.app`** on the release disk image.
2. **Packaged `.app`** — use **Settings → Help → Uninstall Qube** buttons (**Uninstall Qube and all data…** or **Remove Qube app only…**).
3. **Homebrew:** `brew uninstall --cask qube` (add **`--zap`** to remove user data and support files).
4. **Manual** — quit Qube, delete **`Qube.app`** from **`/Applications`** or **`~/Applications`**, then remove **`~/.qube`** and related Library files if desired.

### Linux

1. **AppImage (portable)** — delete the AppImage file.
2. **AppImage + install script** — remove **`~/.local/opt/qube/`**, **`~/.local/bin/qube-appimage`**, and **`~/.local/share/applications/qube-appimage.desktop`** if you used **`install_appimage.sh`**. See [Install Qube on Linux](faq/install-linux.md).
3. **`.deb` package** — run **`qube-uninstall`** (or **`qube-uninstall --keep-data`** to keep **`~/.qube`**) or **`sudo apt remove qube`** / **`qube-vulkan`** / **`qube-cuda`** (only one variant installed at a time).
4. **Packaged `.deb`** — use **Settings → Help → Uninstall Qube** buttons (administrator privileges may be requested).
5. **Source install** — remove your virtual environment and repository clone.
6. For a **full data wipe** on any Linux install path, delete **`~/.qube`** after quitting Qube.

## Related

- [Back up or restore Qube state](backup-or-restore-qube-state.md) — full state backup before removal
- [Export or import a knowledge pack](export-or-import-knowledge-pack.md) — Knowledge configuration export
- [Help settings](../features/settings/help.md) — in-app uninstall section and guided tours
- [Knowledge settings](../features/settings/knowledge.md) — export knowledge pack
