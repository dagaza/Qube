# Pipeline and UI

**Audience:** Contributors and advanced users.  
**Extracted from:** [archived README](../archive/readme-pre-launch-rewrite.md) (pre–launch rewrite).  
**Canonical router reference:** [cognitive_router.md](../cognitive_router.md) (more complete and maintained)

---

### ⚡ Real-Time Interruption (Barge-In)

Qube supports true conversational interruption without crashing the UI thread:

- Speech can interrupt TTS playback instantly.
    
- Wake-word detection triggers immediate cancellation signals via thread-safe booleans.
    
- TTS is micro-chunked (~85ms segments) for fast interruption response without blocking `stream.write()`.
    
- Employs a ~0.75s "Deaf Window" immediately following a wakeword trigger to allow hardware speaker buffers to clear, preventing echo feedback.
    

---

### 🧭 Intent-Aware Routing System (Cognitive Router v4)

Qube uses an adaptive routing system that selects between:

- CHAT (direct LLM response)
    
- RAG (document retrieval)
    
- WEB/TOOL (external/local tools)
    
- MEMORY (long-term memory retrieval)
    

**Key properties:**

- Built on a semantic centroid-based scoring system (`IntentRouter`).
    
- Detects conversation intent drift and adjusts retrieval thresholds dynamically.
    
- Self-tunes using real-time telemetry, applying load penalties if latency spikes.
    
- Deterministic decision making with a <10ms latency target.
    
- Safe fallback to CHAT under uncertainty.

- **Semantic RECALL intent (Memory v6 Phase B):** "tell me about X", "remind me about X", "who is Y" style queries are scored against a recall-intent centroid and forced into the **HYBRID** memory + RAG fusion path automatically — so a thin name-stub memory is always answered with the actual document context behind it instead of just the bare name.
    
- No DAGs, multi-step planners, or recursive loops (intentional simplicity to protect hardware constraints).
    

---

### 🤖 Local LLM routing (dual mode)

Qube no longer *depends* on a separate inference app. Pick your backend in **Settings → Inference engine**:

| Mode | What it is |
| :--- | :--- |
| **Internal Engine (native)** | **llama-cpp-python** inference runs **inside Qube** on a dedicated worker thread—load **.gguf** models, set **GPU offload layers**, and stream tokens with the same low-latency path as external mode. No LM Studio or Ollama required. Includes **execution policy** (Think toggle, reasoning strip/display), **model-aware prompt bundles** for validation and logging (template detection for ChatML, Llama&nbsp;3, Phi, Mistral, etc.—structurally safe reasoning hints), **model-name template overrides** (extra stop tokens + assistant-anchor hints for common families), and **self-healing overrides** persisted under **`~/.qube/model_overrides.json`** when the diagnostic ablation harness detects bad first-token or leakage patterns (applied on later loads—load-time behavior profiling skips a repeat ablation when an override already exists). Optional **load-time behavior profiling** still classifies difficult models for automatic policy tweaks when ablation runs. Chat inference still uses the normal **`messages`** → formatter path; bundles are for observability and parity, not a second sampling stack. |
| **External Server (localhost)** | Classic stack: **LM Studio**, **Ollama**, or any **OpenAI-compatible** server on `localhost` (e.g. ports `1234` / `11434`). |

- **Streaming-first** in both modes (TTFB-friendly, sentence-chunked for TTS).
    
- External mode uses OpenAI-style SSE; internal mode uses the same UI and cancellation semantics via a **thread-safe queue handoff** from the native engine.
    
- Strict timeouts and `finally`-style teardown so the chat UI **always unlocks** if a stream aborts or the server drops.
    

#### 🏪 Model Manager (“App Store” for weights)

Open **Model Manager** from the nav to **search the Hugging Face Hub** (GGUF-oriented results), browse **Qube Verified / Editor’s Picks**, read repo **README** Markdown in-app, pick a **quantization** from the live file list, and **download** directly into Qube’s model storage—**with disk-space checks** before large downloads and clean teardown of partial files if you cancel or something fails.

![Qube Model Manager](../../assets/screenshots/qube_model_manager_dark_mode.png)

---

### 🎙️ Speech-to-Text (STT)

- Powered by `faster-whisper`.
    
- CPU-efficient transcription pipeline.
    
- Streaming-compatible chunk processing.
    
- Optimized for low-latency voice input.
    

---

### 🗣️ High-Fidelity Text-to-Speech (TTS)

- Default engine: **Kokoro ONNX** (30+ voices via `voices-v1.0.bin`).
    
- Optional swap: **Piper ONNX** via Settings → Voice & Audio → Advanced TTS (`.onnx` + `.onnx.json` under `~/.qube/models/tts/`).
    
- Micro-chunk streaming for fast interrupt response.
    
- Strips bracketed citations via regex before audio synthesis to ensure fluid speech.
    
- Designed for real-time conversational playback on CPU.
    

---

### 📚 Advanced RAG Engine

- LanceDB-based vector retrieval system.
    
- PyMuPDF-based document parsing.
    
- Semantic chunking (overlapping window strategy capped at ~1500 chars to protect the C++ engine).
    
- **Strict context budgeting:** max memory characters and max result caps enforced.
    
- **UI-safe retrieval contract:** guarantees `filename` and `content` payloads to prevent UI crashes.
    

---

### 🎛️ Responsive Multithreaded GUI

- Built with **PyQt6** (native widgets—not a RAM-heavy embedded browser), keeping headroom for models and long context.
    
- Fully asynchronous worker architecture (UI thread is strictly isolated).
    
- Escapes model citations into native Markdown (e.g., `[1]`) to bypass `heightForWidth` geometry recalculation loops that would freeze the Qt layout engine.
    
- Real-time telemetry (latency, VU meter, system stats).
    
- Wake-word support (multiple configurable triggers).

![Qube Telemetry Dashboard](../../assets/screenshots/qube_telemetry_dark_mode.png)
