"""Safe size reductions for PyInstaller one-dir Linux release bundles."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.linux_cuda_bundle import REQUIRED_CUDA_WHEEL_LIBS, missing_cuda_wheel_libs, missing_llama_lib
from core.linux_release_variants import normalize_linux_variant

# Directory names that are safe to drop from vendored Python/native trees.
_REMOVABLE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        "tests",
        "test",
        "testing",
        "testdata",
        "docs",
        "doc",
        "examples",
        "example",
        "benchmarks",
        "include",
    }
)

_REMOVABLE_FILE_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyi"})
_REMOVABLE_STATIC_SUFFIXES: frozenset[str] = frozenset({".a", ".la"})


@dataclass
class PruneReport:
    removed_dirs: int = 0
    removed_files: int = 0
    bytes_removed: int = 0
    stripped_files: int = 0
    bytes_stripped: int = 0
    skipped_cuda_libs: list[str] = field(default_factory=list)

    def total_bytes_saved(self) -> int:
        return self.bytes_removed + self.bytes_stripped


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _remove_path(path: Path, report: PruneReport) -> None:
    try:
        if path.is_symlink() or path.is_file():
            size = path.stat().st_size
            path.unlink()
            report.removed_files += 1
            report.bytes_removed += size
            return
        if path.is_dir():
            size = _dir_size(path)
            shutil.rmtree(path)
            report.removed_dirs += 1
            report.bytes_removed += size
    except OSError:
        pass


def _prune_tree(root: Path, report: PruneReport) -> None:
    if not root.is_dir():
        return

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        current = Path(dirpath)
        for name in filenames:
            path = current / name
            if path.suffix in _REMOVABLE_FILE_SUFFIXES or path.suffix in _REMOVABLE_STATIC_SUFFIXES:
                _remove_path(path, report)
        for name in dirnames:
            if name in _REMOVABLE_DIR_NAMES:
                _remove_path(current / name, report)


def _strip_shared_libraries(root: Path, report: PruneReport) -> None:
    strip_bin = shutil.which("strip")
    if not strip_bin:
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        name = path.name
        if ".so" not in name:
            continue
        before = path.stat().st_size
        try:
            subprocess.run(
                [strip_bin, "--strip-debug", str(path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        after = path.stat().st_size
        if after < before:
            report.stripped_files += 1
            report.bytes_stripped += before - after


def _is_nvidia_wheel_shared_lib(name: str) -> bool:
    return name.startswith(("libcudart", "libcublas", "libnv"))


def _prune_extra_cuda_libs(dist_dir: Path, report: PruneReport) -> None:
    lib_dir = dist_dir / "_internal" / "llama_cpp" / "lib"
    if not lib_dir.is_dir():
        return
    allowed = set(REQUIRED_CUDA_WHEEL_LIBS)
    for path in sorted(lib_dir.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if ".so" not in path.name:
            continue
        if not _is_nvidia_wheel_shared_lib(path.name):
            continue
        if path.name in allowed:
            continue
        report.skipped_cuda_libs.append(path.name)
        _remove_path(path, report)


def prune_pyinstaller_bundle(
    dist_dir: Path,
    *,
    variant: str = "cpu",
    strip_binaries: bool = True,
) -> PruneReport:
    """Remove packaging-only bloat without touching runtime-critical files."""
    dist_dir = dist_dir.resolve()
    variant = normalize_linux_variant(variant)
    report = PruneReport()

    internal = dist_dir / "_internal"
    if internal.is_dir():
        _prune_tree(internal, report)

    if variant == "cuda":
        _prune_extra_cuda_libs(dist_dir, report)
        missing = missing_cuda_wheel_libs(dist_dir)
        if missing:
            raise RuntimeError(f"CUDA bundle prune would break runtime; missing: {', '.join(missing)}")
        missing_llama = missing_llama_lib(dist_dir)
        if missing_llama:
            raise RuntimeError(f"CUDA bundle prune would break runtime; {missing_llama}")

    if strip_binaries:
        _strip_shared_libraries(dist_dir, report)

    return report
