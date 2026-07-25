"""Tests for core/linux_bundle_prune.py and core/linux_cuda_bundle.py."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from core.linux_bundle_prune import prune_pyinstaller_bundle
from core.linux_cuda_bundle import REQUIRED_CUDA_WHEEL_LIBS, missing_cuda_wheel_libs


def _touch(path: Path, *, content: bytes = b"x", executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_prune_removes_packaging_bloat_but_keeps_runtime_files(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    internal = dist / "_internal" / "pkg"
    _touch(internal / "module.py", content=b"print('ok')\n")
    _touch(internal / "__pycache__" / "module.pyc", content=b"dead")
    _touch(internal / "tests" / "test_module.py", content=b"import pytest\n")
    _touch(internal / "README.md", content=b"# docs\n")
    _touch(internal / "libfoo.so", content=b"\x7fELF" + b"0" * 128)

    report = prune_pyinstaller_bundle(dist, variant="cpu", strip_binaries=False)

    assert (internal / "module.py").is_file()
    assert not (internal / "__pycache__").exists()
    assert not (internal / "tests").exists()
    assert (internal / "README.md").is_file()
    assert report.removed_dirs >= 2
    assert report.bytes_removed > 0


def test_cuda_prune_drops_extra_wheel_libs_but_keeps_required(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    for name in REQUIRED_CUDA_WHEEL_LIBS:
        _touch(lib_dir / name, content=b"\x7fELF" + b"c" * 64)
    _touch(lib_dir / "libllama.so", content=b"\x7fELF" + b"l" * 64)
    _touch(lib_dir / "libggml-cuda.so", content=b"\x7fELF" + b"g" * 64)
    _touch(lib_dir / "libnvrtc.so.12", content=b"\x7fELF" + b"n" * 64)

    report = prune_pyinstaller_bundle(dist, variant="cuda", strip_binaries=False)

    assert missing_cuda_wheel_libs(dist) == []
    assert (lib_dir / "libggml-cuda.so").is_file()
    assert not (lib_dir / "libnvrtc.so.12").exists()
    assert report.skipped_cuda_libs == ["libnvrtc.so.12"]


def test_cuda_prune_fails_if_required_libs_would_be_removed(tmp_path: Path) -> None:
    dist = tmp_path / "Qube"
    lib_dir = dist / "_internal" / "llama_cpp" / "lib"
    _touch(lib_dir / "libcudart.so.12", content=b"x")
    _touch(lib_dir / "libllama.so", content=b"y")

    with pytest.raises(RuntimeError, match="libcublas"):
        prune_pyinstaller_bundle(dist, variant="cuda", strip_binaries=False)


def test_prune_can_strip_shared_libraries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = tmp_path / "Qube"
    lib = dist / "_internal" / "llama_cpp" / "lib" / "libfoo.so"
    outside = dist / "_internal" / "numpy.libs" / "libscipy_openblas64_.so"
    _touch(lib, content=b"x" * 256)
    _touch(outside, content=b"y" * 256)

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        target = Path(cmd[-1])
        before = target.stat().st_size
        target.write_bytes(b"x" * (before // 2))
        class Result:
            returncode = 0

        return Result()

    fake_strip = tmp_path / "strip"
    fake_strip.write_text("stub", encoding="utf-8")
    monkeypatch.setattr("core.linux_bundle_prune.shutil.which", lambda _name: str(fake_strip))
    monkeypatch.setattr("core.linux_bundle_prune.subprocess.run", fake_run)

    report = prune_pyinstaller_bundle(dist, variant="cpu", strip_binaries=True)

    assert len(calls) == 1
    assert calls[0][-1].endswith("llama_cpp/lib/libfoo.so")
    assert outside.stat().st_size == 256
    assert report.stripped_files == 1
    assert report.bytes_stripped > 0
