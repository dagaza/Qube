## [Unreleased]

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
