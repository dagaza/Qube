## [Unreleased]

## [1.3.48] - 2026-09-03

### Fixed
- **TTS (installed builds):** Bundle Kokoro dependency assets in PyInstaller — `language_tags`, `phonemizer`, and `espeakng_loader` (native library + espeak-ng-data) — so TTS loads on Windows, macOS, and Linux frozen builds.
- **Bootstrap Nemotron download:** Point Nemotron 3 Nano 4B Q8 at the public `unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF` repo (fixes 401 Unauthorized from the gated `bartowski/...-BF16-GGUF` catalogue entry).

## [1.3.47] - 2026-09-03

### Fixed
- **Windows audio devices:** Filter duplicate PortAudio endpoints (prefer WASAPI) so Settings input/output lists align with Windows Sound; refresh device menus when opened so hot-plugged USB/Bluetooth devices appear without restart.
- **TTS voice settings (Windows):** Load Kokoro when ONNX + voices are on disk even if the PyAudio output stream fails at boot; retry with the system default output device; auto-reload when opening Voice & Audio or using TTS Refresh; disable the voice picker and preview buttons until the engine is ready; surface specific load/preview errors instead of a generic message.
- **Bootstrap consent:** Allow feasibility checks to skip RAM enforcement when appropriate so low-memory sessions can still proceed with disk-only guidance.
- **Light theme splash:** Improve splash card text contrast on light backgrounds.

## [1.3.46] - 2026-09-02

### Fixed
- **Kokoro TTS download (bootstrap + Settings):** ONNX assets now download from the official `thewh1teagle/kokoro-onnx` GitHub release instead of removed files on `hexgrad/Kokoro-82M` (fixes 404 on first-run bootstrap and **Download base TTS model**).
- **Composer `@[file:…]` mentions:** exclude the built-in Qube help corpus from file picker results; ignore tool/category routing tokens when filtering filenames.
- **Frameless modals (Windows/Linux):** centralize prestige-style dialog chrome so borders and translucency re-apply reliably after the native window handle exists.
- **Windows silent uninstall wipe:** Inno Setup supports `/DELETEUSERDATA=1` to remove `%LOCALAPPDATA%\Qube` and `%USERPROFILE%\.qube` during silent uninstall; fix `{userprofile}` expansion via `{%USERPROFILE}`.
- **WinGet install grace:** post-install validation runs without mock bootstrap downloads; diagnostics distinguish smoke vs install-grace paths and write grace boot traces for CUDA release CI.
- **Dependencies:** bump `pypdf` to 6.16.1 (CVE-2026-84309/84310/84311).

### Added
- **Release CI (CUDA):** richer WinGet validation smoke failure output (mode, grace trace) and clearer wait/retry handling in `smoke_installed_cuda.ps1`.

## [1.3.45] - 2026-08-31

### Added
- **Native model reload feedback:** changing GPU layers, CPU threads, or context limit in Settings → AI & Models shows a success toast when the model reload completes (including CPU-fallback loads).
- **Native load CPU fallback:** when GPU offload or context allocation fails (``Failed to create llama_context``), automatically retry on CPU before surfacing an error.

### Fixed
- **Model Manager branding (installed builds):** publisher logos, Official badges, and “Official model by / Modified by” detail lines resolve bundled assets via ``resource_path()`` (fixes missing branding on Windows/macOS/Linux PyInstaller installs).
- **STT Whisper (legacy cache layout):** resolve bundled Whisper from flat ``stt/small/`` or Hugging Face hub-cache snapshots so Linux dev installs with only ``models--Systran--…`` weights load correctly.
- **Chat without a conversational model:** universal “open Model Manager” prompt replaces the hardcoded Qwen 3.5 9B bootstrap download when no GGUF is loaded.
- **Native GGUF load errors:** modal dialog with actionable hints for missing shards, non-chat models (ASR/embed), and memory/context failures (previously log-only).
- **Windows GPU defaults:** when VRAM cannot be detected (typical Vulkan/iGPU installs), default GPU layers to **0** (CPU) and cap the slider at 32 instead of ~74 layers that commonly fail ``llama_context`` creation.
- **PyInstaller voice assets:** bundle ``kokoro_onnx`` (``config.json``) and ``openwakeword`` pretrained ONNX models so TTS and wakeword work on installed builds.
- **Release CI (Windows CPU):** silent install after the upgrade smoke no longer hangs on Inno ``AppMutex``. Setup now terminates running ``Qube.exe`` before the mutex check (same as uninstall), and install smokes stop the process tree and wait for exit before launching the next Setup.exe.

## [1.3.44] - 2026-08-31
