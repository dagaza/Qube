# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Qube.

Build with:   pyinstaller qube.spec --noconfirm
Output goes to:  dist/Qube/  (one-dir mode)

To add a Windows icon, place an .ico file at assets/logos/qube.ico and
uncomment the icon= line in the EXE block below.
"""
import sys
import os

sys.path.insert(0, os.getcwd())
from core.__version__ import __version__

a = Analysis(
    ["main.py"],
    pathex=["."],
    datas=[
        ("assets", "assets"),
        ("system_data", "system_data"),
    ],
    hiddenimports=[
        # qtawesome renders SVG icons through QtSvg
        "PyQt6.QtSvg",
        "PyQt6.QtSvgWidgets",
        # lancedb / lance / pyarrow use dynamic C-extension loading
        "lancedb",
        "lance",
        "pyarrow",
        # nvidia-ml-py exposes as pynvml
        "pynvml",
        # faster-whisper backend
        "ctranslate2",
        # onnxruntime for kokoro-onnx TTS
        "onnxruntime",
        # openwakeword pulls tflite_runtime dynamically
        "tflite_runtime",
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
