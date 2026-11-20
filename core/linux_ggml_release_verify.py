"""Release checks for Linux libggml-cpu (AVX-VNNI vs AVX512-VNNI on Zen4)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_VEX_VP_DP_BUSD = re.compile(
    r"\{vex\}\s+vpdpbusd|c4\s+e[0-9a-f]{2}\s+[0-9a-f]{2}\s+50\s+[0-9a-f]{2}",
    re.IGNORECASE,
)
_EVEX_VP_DP_BUSD = re.compile(
    r"\b62\s+[0-9a-f]{2}\s+[0-9a-f]{2}\s+2[08]\s+50\b",
    re.IGNORECASE,
)
_TRUE_STUB = re.compile(
    r"mov\s+\$0x1,%eax",
    re.IGNORECASE,
)
_FALSE_STUB = re.compile(
    r"xor\s+%eax,%eax",
    re.IGNORECASE,
)


def _disassemble_symbol(lib: Path, symbol: str) -> str:
    proc = subprocess.run(
        ["objdump", "-d", f"--disassemble={symbol}", str(lib)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout or ""


def _symbol_range_disasm(lib: Path, symbol: str) -> str:
    nm = subprocess.run(
        ["nm", "-D", str(lib)],
        capture_output=True,
        text=True,
        check=True,
    )
    addr: str | None = None
    for line in nm.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == symbol:
            addr = parts[0]
            break
    if not addr:
        return ""
    start = int(addr, 16)
    stop = start + 0x200
    proc = subprocess.run(
        [
            "objdump",
            "-d",
            str(lib),
            f"--start-address=0x{addr}",
            f"--stop-address=0x{stop:x}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout or ""


def verify_no_avx_vnni_true_stub(lib: Path) -> None:
    blob = _disassemble_symbol(lib, "ggml_cpu_has_avx_vnni")
    if not blob.strip():
        raise ValueError(f"{lib}: missing ggml_cpu_has_avx_vnni")
    if _TRUE_STUB.search(blob) and not _FALSE_STUB.search(blob):
        raise ValueError(
            f"{lib}: ggml_cpu_has_avx_vnni is a compile-time TRUE stub "
            "(GGML_AVX_VNNI=ON); expected FALSE on Linux release builds."
        )


def verify_q8_0_dot_avoids_vex_avx_vnni(lib: Path) -> None:
    """Q8 matmul must not use VEX ``vpdpbusd`` (SIGILL on AVX512-VNNI-only AMD)."""
    blob = _symbol_range_disasm(lib, "ggml_vec_dot_q8_0_q8_0")
    if not blob.strip():
        raise ValueError(f"{lib}: missing ggml_vec_dot_q8_0_q8_0")
    if _VEX_VP_DP_BUSD.search(blob):
        raise ValueError(
            f"{lib}: ggml_vec_dot_q8_0_q8_0 contains VEX vpdpbusd "
            "(AVX-VNNI); use GGML_AVX_VNNI=OFF and AVX512-VNNI paths instead."
        )


def verify_avx512_vnni_kernels_present(lib: Path) -> None:
    """Ensure AVX512-VNNI was not stripped alongside AVX-VNNI."""
    proc = subprocess.run(
        ["objdump", "-d", str(lib)],
        capture_output=True,
        text=True,
        check=True,
    )
    if not _EVEX_VP_DP_BUSD.search(proc.stdout or ""):
        raise ValueError(
            f"{lib}: no EVEX vpdpbusd opcodes found; "
            "AVX512-VNNI kernels may have been disabled unintentionally."
        )


def verify_release_libggml_cpu(lib: Path) -> None:
    if not lib.is_file():
        raise FileNotFoundError(lib)
    verify_no_avx_vnni_true_stub(lib)
    verify_q8_0_dot_avoids_vex_avx_vnni(lib)
    verify_avx512_vnni_kernels_present(lib)


def bundled_libggml_cpu_paths(dist_dir: Path) -> list[Path]:
    internal = dist_dir / "_internal"
    candidates = [
        internal / "libggml-cpu.so.0",
        internal / "llama_cpp" / "lib" / "libggml-cpu.so.0",
    ]
    return [p for p in candidates if p.is_file()]


def configure_bundled_llama_import(dist_dir: Path) -> Path:
    """Point imports and dynamic loader at the PyInstaller ``_internal`` tree only."""
    internal = dist_dir.resolve() / "_internal"
    if not internal.is_dir():
        raise FileNotFoundError(f"dist internal dir missing: {internal}")
    internal_str = str(internal)
    sys.path = [
        p
        for p in sys.path
        if "site-packages" not in p and "dist-packages" not in p
    ]
    if internal_str not in sys.path:
        sys.path.insert(0, internal_str)
    lib_dirs = [
        internal_str,
        str(internal / "llama_cpp" / "lib"),
    ]
    prev = sys.environ.get("LD_LIBRARY_PATH", "")
    joined = ":".join(lib_dirs + ([prev] if prev else []))
    sys.environ["LD_LIBRARY_PATH"] = joined
    return internal


def mapped_libggml_cpu_paths() -> set[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return set()
    hits: set[str] = set()
    for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "libggml-cpu.so" not in line:
            continue
        parts = line.split()
        if parts:
            hits.add(parts[-1])
    return hits


def assert_libggml_loaded_from_dist(internal: Path) -> None:
    resolved = str((internal / "libggml-cpu.so.0").resolve())
    mapped = mapped_libggml_cpu_paths()
    if not mapped:
        raise RuntimeError("libggml-cpu.so not mapped after llama load")
    if not any(p.startswith(str(internal)) for p in mapped):
        raise RuntimeError(
            f"unexpected libggml-cpu mapping(s) {sorted(mapped)!r}; "
            f"expected under {internal}"
        )
    # Prefer the root _internal copy (PyInstaller layout); llama_cpp/lib may match.
    if not any(internal.as_posix() in m for m in mapped):
        raise RuntimeError(
            f"libggml-cpu not loaded from dist tree; maps={sorted(mapped)!r}"
        )