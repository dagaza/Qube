# Update Qube

Download a newer release from [GitHub Releases](https://github.com/dagaza/Qube/releases) and install it over your existing copy. **Your local data is kept** — models, Library, memory, conversations, and settings live outside the application install:

| Platform | User data |
|----------|-----------|
| **Windows** | `%LOCALAPPDATA%\Qube\` |
| **macOS / Linux** | `~/.qube/` |

Quit Qube when the installer or replace dialog asks you to close it.

---

## Windows

1. Download **`Qube-<version>-Setup.exe`** from GitHub Releases.
2. Run the installer. Setup detects an existing install and **updates in place** at **`%LOCALAPPDATA%\Programs\Qube\`**.

**Package managers:**

```powershell
winget upgrade -e --id dagaza.Qube
choco upgrade qube -y
```

---

## macOS

1. Download the **`.dmg`** for your architecture (**arm64** or **x86_64**).
2. Open the DMG and drag **`Qube.app`** to **`/Applications`**. Confirm when macOS offers to **replace** the existing app.

**Homebrew:** `brew upgrade --cask qube`

---

## Linux

### `.deb` (Ubuntu / Debian)

Download the matching variant and upgrade:

```bash
sudo apt install ./qube_1.2.0_amd64.deb
# or qube-vulkan_… / qube-cuda_…
```

Launch with **`qube`** or from the application menu. Install only **one** variant at a time.

### AppImage

**Portable:** run the new AppImage; delete the old file when done.

**Menu integration** (from a repo checkout):

```bash
bash scripts/linux/install_appimage.sh ./Qube-1.2.0-x86_64-vulkan.AppImage
```

This updates **`~/.local/opt/qube/Qube.AppImage`**, the **`qube-appimage`** launcher, and the desktop entry, and removes older AppImage files from that folder.

### In the app

**Settings → Help → Software updates → Check for updates** compares your version with GitHub Releases and opens the download when a newer build is available.

See [Install on Linux](install-linux.md) for variant choice (cpu / vulkan / cuda).

---

## Related

- [App update roadmap](../app_update_roadmap.md) — shipped Tiers 1–2 and Tier 3 automatic-update plan
- [Install on Linux](install-linux.md)
- [Uninstall Qube](uninstall.md)
- [Releasing](../releasing.md) — maintainer release process

In-app help: **Library → Qube → Update Qube**, or ask with **`@[tool:help]`**.
