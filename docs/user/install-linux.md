# Install Qube on Linux

Official Linux builds ship as **AppImage** (portable) and **`.deb`** (Ubuntu/Debian) in three inference variants:

| Variant | Best for | Artifact examples |
|---------|----------|-------------------|
| **cpu** | Any PC; slowest inference | `Qube-1.1.0-x86_64-cpu.AppImage`, `qube_1.1.0_amd64.deb` |
| **vulkan** | AMD / Intel GPUs | `Qube-1.1.0-x86_64-vulkan.AppImage`, `qube-vulkan_1.1.0_amd64.deb` |
| **cuda** | NVIDIA GPUs (recent driver) | `Qube-1.1.0-x86_64-cuda.AppImage`, `qube-cuda_1.1.0_amd64.deb` |

Install **one** `.deb` variant at a time (`qube`, `qube-vulkan`, and `qube-cuda` conflict with each other). AppImages are portable — keep the file that matches your hardware.

For development or bleeding-edge checkouts, use [install from source](install-from-source.md).

---

## System requirements

- **Ubuntu 22.04+ / Debian 12+** (amd64) for the packaged builds
- **16 GB RAM** minimum (**20 GB** recommended)
- **Microphone and speakers** for voice features
- **Vulkan builds:** working Vulkan drivers/mesa stack (`libvulkan1`)
- **CUDA builds:** proprietary NVIDIA driver compatible with CUDA 12.4 wheels (no separate CUDA toolkit required for the packaged app)
- Runtime libraries (`.deb` installs these automatically; AppImage bundles most GUI libs):

  ```bash
  sudo apt install libportaudio2 libgl1 libxcb1 libxkbcommon0
  ```

See [system requirements](system-requirements.md) for model and storage guidance.

---

## AppImage (recommended for first try)

1. Download the AppImage for your GPU from [GitHub Releases](https://github.com/dagaza/Qube/releases).
2. Make it executable and run:

   ```bash
   chmod +x Qube-1.1.0-x86_64-vulkan.AppImage
   ./Qube-1.1.0-x86_64-vulkan.AppImage
   ```

On systems without FUSE, set:

```bash
export APPIMAGE_EXTRACT_AND_RUN=1
./Qube-1.1.0-x86_64-vulkan.AppImage
```

User data is stored under `~/.qube/` (shared across variants).

---

## Debian / Ubuntu (.deb)

1. Download the matching `.deb` from [GitHub Releases](https://github.com/dagaza/Qube/releases).
2. Install **one** variant:

   ```bash
   sudo apt install ./qube_1.1.0_amd64.deb          # CPU
   sudo apt install ./qube-vulkan_1.1.0_amd64.deb   # AMD/Intel (Vulkan)
   sudo apt install ./qube-cuda_1.1.0_amd64.deb     # NVIDIA (CUDA)
   ```

3. Launch from your application menu or run:

   ```bash
   qube
   ```

The package installs the PyInstaller bundle to `/opt/qube/` and a `/usr/bin/qube` wrapper.

To switch variants later, remove the installed package first, then install the other `.deb`.

---

## Uninstall

See [uninstall.md](uninstall.md).

---

## Build locally (maintainers)

```bash
bash scripts/linux/install_build_deps.sh vulkan   # or cpu / cuda / all
bash scripts/linux/build_linux.sh 9.9.9 vulkan
bash scripts/release/smoke_linux_dist.sh
bash scripts/linux/fetch_appimage_tools.sh
bash scripts/linux/build_appimage.sh 9.9.9 vulkan
bash scripts/linux/build_deb.sh 9.9.9 vulkan
```

Revert `core/__version__.py` and `pyproject.toml` after dry runs if needed.

---

## Related

- [Install from source](install-from-source.md)
- [Uninstall Qube](uninstall.md)
- [Releasing](../releasing.md)
