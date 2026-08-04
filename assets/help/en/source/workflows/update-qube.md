# Update Qube

## Common questions

- How do I update Qube to a new version?
- Will updating Qube delete my models, Library, or memory?
- How do I upgrade Qube on Windows, macOS, or Linux?
- Can I run the new installer when Qube is already installed?

## What it is

**Update Qube** means replacing the application files with a newer release while keeping your local data (models, Library indexes, memory, conversations, settings). Data lives outside the install directory:

| Platform | User data location |
|----------|-------------------|
| **Windows** | `%LOCALAPPDATA%\Qube\` |
| **macOS / Linux** | `~/.qube/` |

Quit Qube before updating when the installer asks, or when replacing the app manually.

## Where to find it

- [GitHub Releases](https://github.com/dagaza/Qube/releases) — download the installer or package for your platform
- This workflow — searchable via **Library → Qube** or **`@[tool:help]`**
- Package managers — **WinGet**, **Chocolatey**, or **Homebrew** (see below)

## Also called

upgrade Qube, install new version, update app, replace Qube, bump version

## How to…

### Windows

1. Download **`Qube-<version>-Setup.exe`** from **GitHub Releases** (or use a package manager below).
2. Run the installer. If Qube is already installed, Setup detects it and **updates in place** at **`%LOCALAPPDATA%\Programs\Qube\`**. Close Qube when prompted if it is running.
3. Launch Qube from the Start menu or desktop shortcut as usual.

**WinGet:** `winget upgrade -e --id dagaza.Qube`

**Chocolatey:** `choco upgrade qube -y`

Your models and settings in **`%LOCALAPPDATA%\Qube\`** are kept unless you uninstall and delete that folder separately.

### macOS

1. Download the **`.dmg`** for your Mac architecture (**arm64** or **x86_64**) from **GitHub Releases**.
2. Open the DMG and drag **`Qube.app`** to **`/Applications`**. When macOS asks to **replace** the existing app, confirm.
3. Open Qube from Applications or the Dock.

**Homebrew:** `brew upgrade --cask qube`

User data in **`~/.qube/`** is unchanged.

### Linux — `.deb` (Ubuntu / Debian)

1. Download the matching **`.deb`** variant (**cpu**, **vulkan**, or **cuda**) from **GitHub Releases**.
2. Upgrade in place:

   ```bash
   sudo apt install ./qube_1.2.0_amd64.deb
   # or qube-vulkan_… / qube-cuda_… for your GPU variant
   ```

3. Launch from the application menu or run **`qube`**.

Install **one** variant at a time. To switch GPU variants, remove the current package first, then install the other `.deb`.

### Linux — AppImage

**Portable (no menu entry):** download the new AppImage, make it executable, and run it. Delete the old file when you no longer need it.

**With the install script** (application menu + **`qube-appimage`** launcher):

```bash
bash scripts/linux/install_appimage.sh ./Qube-1.2.0-x86_64-vulkan.AppImage
```

The script copies the new build to **`~/.local/opt/qube/Qube.AppImage`**, updates the launcher and menu entry, and **removes older AppImage files** from that folder. User data stays in **`~/.qube/`**.

### In the app

Open **Settings → About → Software updates** and click **Check for updates**. Qube compares your installed version with the latest GitHub Release and opens the matching download when a newer build is available.

## Related

- [Install Qube on Linux](../faq/install-linux.md) — first install and variant choice
- [Uninstall Qube](uninstall-qube.md) — remove the app or wipe user data
- [What's new in Qube Help](../release/whats-new.md) — help corpus highlights after an upgrade
