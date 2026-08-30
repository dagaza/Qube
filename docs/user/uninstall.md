# Uninstall Qube

How to remove Qube and its local data on each platform.

---

## Before you uninstall

- **Back up if needed** — In Qube, use **Settings → Knowledge → Diagnostics → Export knowledge pack** to save `~/.qube/knowledge-pack.json` (credentials redacted). Models and library files live under `~/.qube/` and are deleted on a full uninstall.
- **Quit Qube on Windows** — Closing the main window **minimizes to the system tray**; the app keeps running. Use **Exit Qube** from the tray icon, then confirm **Qube.exe** is gone in Task Manager before uninstalling manually. The Inno uninstaller (`unins000.exe`) also stops Qube automatically when you uninstall from **Settings → Apps**.

---

## Windows

### Installed from the release installer (recommended)

1. **Exit Qube** from the system tray (**Exit Qube**), or confirm no **Qube.exe** process in Task Manager.
2. Open **Settings → Apps → Installed apps** (or **Add or remove programs**).
3. Select **Qube** → **Uninstall**.

The uninstaller removes application files under:

```text
%LOCALAPPDATA%\Programs\Qube\
```

(including `Qube.exe` and `_internal\`). If anything remains, Qube was likely still running during uninstall — exit the app and run `unins000.exe` again, or delete that folder after ending **Qube.exe** in Task Manager.

Or run the Inno Setup uninstaller directly:

```text
%LOCALAPPDATA%\Programs\Qube\unins000.exe
```

Silent uninstall (IT scripts):

```powershell
& "$env:LOCALAPPDATA\Programs\Qube\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

### WinGet

```powershell
winget uninstall -e --id dagaza.Qube
```

### Chocolatey

```powershell
choco uninstall qube -y
```

### User data after Windows uninstall

The installer removes the application under `%LOCALAPPDATA%\Programs\Qube\` only. User data is stored in **two** locations on Windows:

```text
%LOCALAPPDATA%\Qube\          — models, library DB, logs
%USERPROFILE%\.qube\          — settings.json and other support files
```

When you uninstall from **Settings → Apps**, the uninstaller asks whether to remove this data as well. Choose **Yes** for a completely clean removal, or **No** (default) to keep models and settings for a future reinstall.

Silent uninstall (`/VERYSILENT`) **never** deletes user data; remove the folders above manually if needed.

---

## macOS

Each release **DMG** includes **`Uninstall Qube.app`** next to **`Qube.app`**. Double-click it and confirm to remove the app and local data.

### From inside Qube (packaged `.app` only)

1. Open **Settings → Help**.
2. Under **Uninstall Qube**, choose:
   - **Uninstall Qube and all data…** — removes the app, `~/.qube`, and related Library files, or
   - **Remove Qube app only…** — removes the app but keeps `~/.qube`.

Qube quits and a background script finishes removal.

### Homebrew Cask

```bash
brew uninstall --cask qube
```

To remove leftover user data as well:

```bash
brew uninstall --cask --zap qube
```

The `zap` paths match the same manifest used by **`Uninstall Qube.app`** (see `core/uninstall_paths.py` in the repository).

### Manual removal

If helpers are unavailable:

1. Quit Qube.
2. Drag **`Qube.app`** from `/Applications` (or `~/Applications`) to the Trash.
3. Delete user data if desired:

```bash
rm -rf ~/.qube
rm -f ~/Library/Preferences/com.dagaza.Qube.plist
rm -rf ~/Library/Saved\ Application\ State/com.dagaza.Qube.savedState
```

---

## Linux (AppImage / .deb / source)

### AppImage

Delete the AppImage file. To remove user data as well:

```bash
rm -rf ~/.qube
```

### Debian / Ubuntu package

From the terminal:

```bash
qube-uninstall
```

Or remove the package only (keeps `~/.qube`):

```bash
qube-uninstall --keep-data
```

Non-interactive removal (automation):

```bash
qube-uninstall --quiet
```

You can also use the package manager directly:

```bash
sudo apt remove qube            # CPU package
sudo apt remove qube-vulkan     # Vulkan package
sudo apt remove qube-cuda       # CUDA package
```

From inside Qube (packaged `.deb` only):

1. Open **Settings → Help**.
2. Under **Uninstall Qube**, choose:
   - **Uninstall Qube and all data…** — removes the package, `~/.qube`, and related files, or
   - **Remove Qube app only…** — removes the package but keeps `~/.qube`.

Administrator privileges may be requested to remove the `.deb` package.

To remove user data after `apt remove` alone:

```bash
rm -rf ~/.qube
```

### Source install

If you installed from a git checkout:

1. Deactivate and delete your virtual environment (`venv/`).
2. Delete your clone of the repository if you no longer need it.
3. Remove user data:

```bash
rm -rf ~/.qube
```

If you built GPU wheels with `./scripts/install_llama_cpp_gpu.sh`, no system-wide uninstall step is required unless you installed extra OS packages for CUDA/Vulkan builds.

---

## Related

- [Install from source](install-from-source.md)
- [System requirements](system-requirements.md)
- [Export or import a knowledge pack](../../assets/help/en/workflows/export-or-import-knowledge-pack.md) (in-app help)
