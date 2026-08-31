# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Qube.

Build with:   pyinstaller qube.spec --noconfirm
Output goes to:
  Windows/Linux:  dist/Qube/     (one-dir mode)
  macOS:          dist/Qube.app  (application bundle)
"""
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.path.insert(0, os.getcwd())
from core.__version__ import __version__

_IS_MACOS = sys.platform == "darwin"
_LINUX_VARIANT = os.environ.get("QUBE_LINUX_VARIANT", "")
_WINDOWS_VARIANT = os.environ.get("QUBE_WINDOWS_VARIANT", "")
_BUILD_VARIANT = _LINUX_VARIANT or _WINDOWS_VARIANT

datas = [
    ("assets", "assets"),
    ("system_data", "system_data"),
]
datas += collect_data_files("qtawesome", include_py_files=False)
for _pkg in ("mf2py", "extruct", "recipe_scrapers", "kokoro_onnx", "openwakeword"):
    try:
        datas += collect_data_files(_pkg, include_py_files=False)
    except Exception:
        pass

binaries = []
for package in ("PyAudio", "onnxruntime", "ctranslate2", "llama_cpp"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

if _BUILD_VARIANT == "cuda":
    try:
        from core.nvidia_wheel_lib_dirs import CUDA_WHEEL_PACKAGES, iter_nvidia_wheel_libs

        for path in iter_nvidia_wheel_libs(*CUDA_WHEEL_PACKAGES):
            binaries.append((str(path), "llama_cpp/lib"))
    except Exception as exc:
        print(f"WARNING: could not pre-collect CUDA runtime libs for PyInstaller: {exc}")

# pynvml drives NVIDIA/CUDA telemetry, which does not exist on macOS (Metal).
_hidden_imports = [
    "PyQt6.QtSvg",
    "PyQt6.QtSvgWidgets",
    "lancedb",
    "lance",
    "pyarrow",
    "pynvml",
    "ctranslate2",
    "onnxruntime",
    "tflite_runtime",
    "llama_cpp",
]
_excludes = []
if _IS_MACOS:
    _hidden_imports.remove("pynvml")
    _excludes.append("pynvml")

a = Analysis(
    ["qube_entry.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=_hidden_imports,
    hookspath=[],
    runtime_hooks=["pyi_rth_llama_cpp.py"],
    excludes=_excludes,
)

pyz = PYZ(a.pure)

_icns_path = os.path.join("assets", "logos", "qube.icns")
_ico_path = os.path.join("assets", "logos", "qube.ico")
if _IS_MACOS and os.path.isfile(_icns_path):
    _icon_path = _icns_path
elif os.path.isfile(_ico_path):
    _icon_path = _ico_path
else:
    _icon_path = None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Qube",
    icon=_icon_path,
    console=False,
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Qube",
)

if _IS_MACOS:
    app = BUNDLE(
        coll,
        name="Qube.app",
        icon=_icon_path,
        bundle_identifier="com.dagaza.Qube",
        version=__version__,
        info_plist={
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "NSMicrophoneUsageDescription": (
                "Qube uses the microphone for voice input and wake-word detection."
            ),
        },
    )
