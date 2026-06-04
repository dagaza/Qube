# Local validation (matches CI)

Use a dedicated virtual environment with **Python 3.12 or 3.13** — not the system
`C:\Python310` install. The project requires `>=3.12` per `pyproject.toml`; CI uses 3.13.

## One-time setup

```powershell
cd C:\Users\Keith\source\repos\dagaza\Qube

# Create venv with Python 3.13 (recommended)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all deps
python -m pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
```

Always activate the venv before running tests:

```powershell
.\.venv\Scripts\Activate.ps1
python --version   # should show 3.12+ or 3.13
where.exe python   # should point inside .venv\Scripts
```

## PR CI equivalent

```powershell
pytest tests/ -v --tb=short -m "not packaging"
pip-audit -r requirements.txt --ignore-vuln CVE-2025-69872
```

If `pytest` still resolves to `C:\Python310\Scripts\pytest.exe`, use:

```powershell
python -m pytest tests/ -v --tb=short -m "not packaging"
python -m pip_audit -r requirements.txt --ignore-vuln CVE-2025-69872
```

## Full release build (optional)

Closest to the GitHub **Build & Release** workflow. Requires **Inno Setup 6** for the installer step.

**Inno Setup (pick one):**

- Install from [jrsoftware.org](https://jrsoftware.org/isinfo.php) (recommended on non-admin shells), or
- Elevated PowerShell: `choco install innosetup -y`

```powershell
.\scripts\build_windows.ps1 -Version 9.9.9
```

Dist-only (PyInstaller + 10s EXE smoke, no installer):

```powershell
.\scripts\build_windows.ps1 -Version 9.9.9 -SkipInstaller
```

If PyInstaller already succeeded but Inno failed, compile and smoke the installer only:

```powershell
$iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
& $iscc /DMyAppVersion=9.9.9 installer\qube.iss
.\scripts\release\smoke_installed.ps1
```

Use a throwaway version for dry runs; revert `core/__version__.py` and `pyproject.toml` afterward if needed.

## Common mistakes

| Symptom | Cause |
|---------|--------|
| `No module named 'PyQt6'` | Global Python 3.10 used; deps not installed |
| `'PyQt6' is not a package` | Broken/partial PyQt6 on global Python, or wrong interpreter |
| `Unknown config option: qt_api` | `pytest-qt` not installed (old global pytest) |
| `pip-audit` not recognized | Not installed; run `pip install -r requirements-dev.txt` in venv |
| `pip-audit` flags `diskcache` / CVE-2025-69872 | Transitive via `llama-cpp-python`; no patched release yet — audit step ignores this ID until upstream fixes it |
| Circular import in `prompt_blocks` | Fixed — import `render_system_ok_messages` from `core.prompt_renderers` |
| `No module named 'packaging.version'` | Do not use a top-level or `tests/packaging/` folder — it shadows PyPI `packaging`. Release smoke tests live in `tests/release_smoke/` |
| `iscc` not recognized / choco Access denied | Inno Setup not installed; install from [jrsoftware.org](https://jrsoftware.org/isinfo.php) or run `choco install innosetup -y` in **elevated** PowerShell |
| `Access is denied: ...\\pytest-of-<user>` | Corrupted or locked system temp dir; pytest now uses `.pytest_tmp/` in the repo via `--basetemp`. Delete `%TEMP%\\pytest-of-<user>` if errors persist |
