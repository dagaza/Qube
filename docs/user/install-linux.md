# Install Qube on Linux

Official Linux builds ship as an **AppImage** (portable) and a **`.deb`** (Ubuntu/Debian). Both use a CPU-only `llama-cpp-python` wheel by default; GPU acceleration can still be added later via the source workflow.

For development or bleeding-edge checkouts, use [install from source](install-from-source.md).

---

## System requirements

- **Ubuntu 22.04+ / Debian 12+** (amd64) for the packaged builds
- **16 GB RAM** minimum (**20 GB** recommended)
- **Microphone and speakers** for voice features
- Runtime libraries (`.deb` installs these automatically; AppImage bundles most GUI libs):

  ```bash
  sudo apt install libportaudio2 libgl1 libxcb1 libxkbcommon0
  ```

See [system requirements](system-requirements.md) for model and storage guidance.

---

## AppImage (recommended for first try)

1. Download `Qube-<version>-x86_64.AppImage` from [GitHub Releases](https://github.com/dagaza/Qube/releases).
2. Make it executable and run:

   ```bash
   chmod +x Qube-1.1.0-x86_64.AppImage
   ./Qube-1.1.0-x86_64.AppImage
   ```

On systems without FUSE, set:

```bash
export APPIMAGE_EXTRACT_AND_RUN=1
./Qube-1.1.0-x86_64.AppImage
```

User data is stored under `~/.qube/`.

---

## Debian / Ubuntu (.deb)

1. Download `qube_<version>_amd64.deb` from [GitHub Releases](https://github.com/dagaza/Qube/releases).
2. Install:

   ```bash
   sudo apt install ./qube_1.1.0_amd64.deb
   ```

3. Launch from your application menu or run:

   ```bash
   qube
   ```

The package installs the PyInstaller bundle to `/opt/qube/` and a `/usr/bin/qube` wrapper.

---

## Uninstall

See [uninstall.md](uninstall.md).

---

## Build locally (maintainers)

```bash
bash scripts/linux/install_build_deps.sh
bash scripts/linux/build_linux.sh 9.9.9
bash scripts/linux/fetch_appimage_tools.sh
bash scripts/linux/build_appimage.sh 9.9.9
bash scripts/linux/build_deb.sh 9.9.9
bash scripts/release/smoke_linux_dist.sh
```

Revert `core/__version__.py` and `pyproject.toml` after dry runs if needed.

---

## Related

- [Install from source](install-from-source.md)
- [Uninstall Qube](uninstall.md)
- [Releasing](../releasing.md)
