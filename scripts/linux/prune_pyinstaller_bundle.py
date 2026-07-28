#!/usr/bin/env python3
"""Prune a PyInstaller one-dir Linux bundle before packaging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.linux_bundle_prune import prune_pyinstaller_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        type=Path,
        default=_REPO / "dist" / "Qube",
        help="PyInstaller output directory (default: dist/Qube)",
    )
    parser.add_argument(
        "--variant",
        default="cpu",
        help="Linux build variant: cpu, vulkan, or cuda",
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Skip strip --strip-debug on shared libraries",
    )
    args = parser.parse_args(argv)

    report = prune_pyinstaller_bundle(
        args.dist_dir,
        variant=args.variant,
        strip_binaries=not args.no_strip,
    )
    saved_mb = report.total_bytes_saved() / (1024 * 1024)
    print(
        "Pruned bundle:"
        f" removed {report.removed_dirs} dirs / {report.removed_files} files"
        f" ({report.bytes_removed / (1024 * 1024):.1f} MiB),"
        f" stripped {report.stripped_files} shared libs"
        f" ({report.bytes_stripped / (1024 * 1024):.1f} MiB),"
        f" total saved ~{saved_mb:.1f} MiB",
    )
    if report.skipped_cuda_libs:
        print("Removed extra CUDA wheel libraries:", ", ".join(report.skipped_cuda_libs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
