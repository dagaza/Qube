# Install Qube on Linux

## Common questions

- How do I install Qube on Linux?
- Which AppImage should I download — CPU, Vulkan, or CUDA?
- What is the difference between AppImage and `.deb` on Linux?
- How do I add Qube to my application menu on Linux?
- Can I install Qube on Ubuntu or Debian without building from source?

## What it is

Official **Linux (amd64)** builds ship from [GitHub Releases](https://github.com/dagaza/Qube/releases) as:

| Format | Best for |
|--------|----------|
| **AppImage** | Portable install — try first; one file per GPU variant |
| **`.deb`** | Ubuntu / Debian integration — `/usr/bin/qube`, app menu, **`qube-uninstall`** |

Three **inference variants** — install **one** that matches your GPU:

| Variant | Hardware | Example filename |
|---------|----------|------------------|
| **cpu** | Any PC; slowest | `Qube-1.1.0-x86_64-cpu.AppImage` |
| **vulkan** | AMD / Intel GPU | `Qube-1.1.0-x86_64-vulkan.AppImage` |
| **cuda** | NVIDIA GPU (recent driver) | `Qube-1.1.0-x86_64-cuda.AppImage` |

**Flatpak** is not shipped today — AppImage + `.deb` are the supported official paths.

## System requirements

- **Ubuntu 22.04+ / Debian 12+** (amd64) for packaged builds
- **16 GB RAM** minimum (**20 GB** recommended)
- **Microphone and speakers** for voice features
- See the [System requirements](https://github.com/dagaza/Qube/blob/main/docs/user/system-requirements.md) doc for model and storage guidance

## AppImage (recommended first try)

1. Download the AppImage matching your GPU from **GitHub Releases**.
2. Make it executable and run:

   ```bash
   chmod +x Qube-*-x86_64-vulkan.AppImage   # pick cpu / vulkan / cuda
   ./Qube-*-x86_64-vulkan.AppImage
   ```

3. **Optional — application menu entry** (copies to `~/.local/opt/qube/`):

   ```bash
   bash scripts/linux/install_appimage.sh ./Qube-*-x86_64-vulkan.AppImage
   qube-appimage
   ```

On systems **without FUSE**, use:

```bash
export APPIMAGE_EXTRACT_AND_RUN=1
./Qube-*-x86_64-vulkan.AppImage
```

The installer script sets `APPIMAGE_EXTRACT_AND_RUN=1` in the desktop entry automatically.

## Debian / Ubuntu (`.deb`)

1. Download the matching `.deb` from **GitHub Releases**.
2. Install **one** variant:

   ```bash
   sudo apt install ./qube_1.1.0_amd64.deb          # CPU
   sudo apt install ./qube-vulkan_1.1.0_amd64.deb   # AMD/Intel (Vulkan)
   sudo apt install ./qube-cuda_1.1.0_amd64.deb     # NVIDIA (CUDA)
   ```

3. Launch from your application menu or run **`qube`**.

Packages install to **`/opt/qube/`** with **`/usr/bin/qube`** and **`qube-uninstall`**. To switch variants, remove the installed package first, then install the other `.deb`.

## Which variant should I pick?

| Your GPU | Download |
|----------|----------|
| No discrete GPU / unsure | **cpu** |
| AMD or Intel graphics | **vulkan** |
| NVIDIA with recent proprietary driver | **cuda** |

When in doubt, start with **cpu** or **vulkan** — you can switch later (AppImage: download another file; `.deb`: remove package, install other variant).

## User data

All variants store data under **`~/.qube/`** (settings, models, Library, memory). Upgrading the app does not remove this folder.

## Updating

See **[Update Qube](workflows/update-qube.md)** for full steps. Summary:

- **`.deb`:** `sudo apt install ./qube_<new>_amd64.deb` (or matching vulkan/cuda package)
- **AppImage + install script:** re-run `bash scripts/linux/install_appimage.sh ./Qube-<new>-x86_64-<variant>.AppImage` — installs to `~/.local/opt/qube/Qube.AppImage` and removes older AppImage files there
- **Portable AppImage:** download and run the new file; delete the old one when done

## Uninstall

- **AppImage:** delete the file; remove `~/.local/bin/qube-appimage` and `~/.local/share/applications/qube-appimage.desktop` if you ran the install script
- **`.deb`:** **`qube-uninstall`** or **`sudo apt remove qube`** (or `qube-vulkan` / `qube-cuda`)

See [Uninstall Qube](workflows/uninstall-qube.md) for a full data wipe.

## From source

Developers can still [install from source](https://github.com/dagaza/Qube/blob/main/docs/user/install-from-source.md) — not required for end users when a release build fits your hardware.

## Related

- [Update Qube](workflows/update-qube.md)
- [Uninstall Qube](workflows/uninstall-qube.md)
- [Hardware tuning (Internal Engine)](hardware-tuning-internal-engine.md) — GPU layers after install
- [Companion on Wayland (Linux)](companion-wayland-linux.md) — floating orb on Linux
- [Web discovery privacy tiers](web-discovery-privacy-tiers.md)
