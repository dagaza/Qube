# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Qube.

Build with:   pyinstaller qube.spec --noconfirm
Output goes to:  dist/Qube/  (one-dir mode)
"""
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

sys.path.insert(0, os.getcwd())
from core.__version__ import __version__  # noqa: F401 — build metadata

datas = [
    ("assets", "assets"),
    ("system_data", "system_data"),
]
datas += collect_data_files("qtawesome", include_py_files=False)

binaries = []
for package in ("PyAudio", "onnxruntime", "ctranslate2", "llama_cpp"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
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
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)

pyz = PYZ(a.pure)

_icon_path = os.path.join("assets", "logos", "qube.ico")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Qube",
    icon=_icon_path if os.path.isfile(_icon_path) else None,
    console=False,
    uac_admin=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="Qube",
)
