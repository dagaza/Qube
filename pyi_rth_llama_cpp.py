"""PyInstaller runtime hook (intentionally empty).

``llama_cpp`` DLL paths and imports are deferred until
:func:`core.llama_cpp_import.get_llama_class` runs during an explicit model load.
Registering CUDA/Vulkan backend DLL directories at process start caused
WinGet ``Validation-Defender-Error`` on ``dagaza.Qube.CUDA`` when the sidecar
loaded a GGUF immediately after first-run bootstrap.
"""
