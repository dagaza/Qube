# Uninstall Qube

How to remove Qube and its local data on each platform.

---

## Before you uninstall

- **Back up if needed** — In Qube, use **Settings → Knowledge → Diagnostics → Export knowledge pack** to save `~/.qube/knowledge-pack.json` (credentials redacted). Models and library files live under `~/.qube/` and are deleted on a full uninstall.
- **Quit Qube** — Close the app before removing files manually. The macOS and Windows uninstallers quit Qube for you.

---

## Windows

### Installed from the release installer (recommended)

1. Open **Settings → Apps → Installed apps** (or **Add or remove programs**).
2. Select **Qube** → **Uninstall**.

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

The installer removes the application under `%LOCALAPPDATA%\Programs\Qube\`. User data (models, library, memory, settings) is stored separately under:

```text
%LOCALAPPDATA%\Qube\
```

Delete that folder if you want a completely clean removal.

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

## Linux (source install)

There is no official Linux package yet. To remove a source install:

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
