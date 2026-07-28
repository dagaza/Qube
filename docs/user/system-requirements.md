# System requirements

Summary tables for planning hardware and storage. Qube targets **local-first** operation: most RAM should go to models and context, not the UI shell.

---

## Quick reference

| | Minimum | Recommended |
|---|---------|-------------|
| **RAM** | 16 GB | 20 GB |
| **OS** | Windows 10+, macOS 12+ (Apple Silicon or Intel), Linux (source install) | Same |
| **Storage** | ~2 GB for app + voice models; additional space per chat model | SSD strongly recommended |
| **Audio** | Microphone and speakers (or headset) for voice | Same |
| **GPU** | Optional | Discrete GPU or Apple Silicon with enough VRAM for your chosen model |
| **Python** | 3.12+ (source install only) | 3.13 |

---

## RAM and models

Qube is designed around a **strict on-device memory budget** (roughly **10–15 GB usable** for models + context on a 16 GB machine, depending on OS overhead).

| RAM | Guidance |
|-----|----------|
| **16 GB** | Use a small SLM (e.g. **Nemotron 3 Nano 4B**). Lower GPU offload layers if load fails. Enable **Settings → Help → Suggest models for my hardware** in Model Manager. |
| **20 GB+** | More headroom for larger quants and longer context. |
| **32 GB+** | Comfortable for mid-size models; still match quant to VRAM when using GPU offload. |

Disk swapping during inference makes voice and chat unusably slow — prefer a smaller model over exceeding RAM.

---

## Operating systems

| Platform | Distribution |
|----------|--------------|
| **Windows** | Installer via [GitHub Releases](https://github.com/dagaza/Qube/releases), WinGet (`dagaza.Qube`), or Chocolatey (`qube`) |
| **macOS** | Signed/notarized `.dmg` from GitHub Releases when available; Homebrew cask via `dagaza/homebrew-qube` when published |
| **Linux** | **AppImage**, **`.deb`**, **`.rpm`**, and **`.tar.gz`** (amd64) from GitHub Releases; [install guide](install-linux.md). Source install still supported. |
| **macOS** | `.dmg` from GitHub Releases, or **`brew tap dagaza/qube && brew install --cask qube`** (custom tap; unsigned builds supported) |

---

## GPU acceleration

**Internal Engine (native)** supports **GPU offload layers** (Settings → AI & Models → Hardware tuning).

| Hardware | Notes |
|----------|-------|
| **NVIDIA (Windows/Linux)** | Windows: **`cuda`** Setup.exe from GitHub Releases; Linux: **`cuda`** AppImage/`.deb`; or GPU source install |
| **Apple Silicon (macOS)** | Metal build in release DMGs |
| **AMD / Intel (Windows/Linux)** | Windows: **`vulkan`** Setup.exe; Linux: **`vulkan`** AppImage/`.deb`; or `install_llama_cpp_gpu.sh` |
| **CPU only** | Supported — STT (faster-whisper) and TTS (Kokoro) run well on CPU; chat will be slower |

**ROCm (AMD HIP on Linux)** is not shipped today. See [ROCm support exploration](../rocm_support_exploration.md) for feasibility and trade-offs vs Vulkan.

**External Server** mode delegates inference to LM Studio / Ollama — tune GPU settings in that host app instead.

---

## Storage planning

| Component | Typical size |
|-----------|----------------|
| Application | Varies by platform package |
| Kokoro TTS (first run) | ~400 MB |
| STT / wake-word / embedding models | Additional hundreds of MB (downloaded as needed) |
| Chat `.gguf` models | ~2 GB – 40 GB+ depending on model and quant |
| Library documents | Your PDFs/EPUBs + LanceDB index |
| Long-term memory | LanceDB rows under `~/.qube/` |

Model Manager performs **disk-space checks** before large Hugging Face downloads.

---

## Network

Qube runs **offline** for chat, Library, and memory. Network is used only when **you** choose to:

- Download models (Model Manager → Hugging Face)
- Run web search, Live Sources, or `@research`
- Fetch bootstrap assets on first run

No third-party **chat API** is required.

---

## UI shell

Qube uses **PyQt6** native widgets (not Electron or a browser tab). On memory-constrained machines, that matters because **model weights** consume most RAM — a lighter UI shell leaves more headroom for context and retrieval.

---

## Related

- [Install from source](install-from-source.md)
- In-app: **Hardware tuning (Internal Engine)** FAQ (`Library → Qube`)
- [Architecture stack](../architecture/stack.md) — component-level detail
