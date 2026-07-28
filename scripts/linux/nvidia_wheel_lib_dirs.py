"""Linux CUDA PyInstaller bundle requirements — re-export for scripts."""

from core.nvidia_wheel_lib_dirs import CUDA_WHEEL_PACKAGES, iter_nvidia_wheel_libs

__all__ = ["CUDA_WHEEL_PACKAGES", "iter_nvidia_wheel_libs"]
