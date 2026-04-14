# Qube: Local Hardware-Accelerated AI Assistant

| **Dark Theme** | **Light Theme** |
| :---: | :---: |
| ![Dark Theme](assets/screenshots/qube_conversations_dark_mode.png) | ![Light Theme](assets/screenshots/qube_conversations_light_mode.png) |

Qube is a fully local, privacy-first, voice-to-voice AI desktop assistant built on a multithreaded, streaming-first pipeline. Operating entirely offline under a strict memory budget, it integrates state-of-the-art voice processing, adaptive cognitive routing, and asynchronous semantic memory enrichment directly into your hardware environment. Run inference with a **built-in native llama.cpp engine** *or* plug in LM Studio / Ollama—load your files either way—and experience a genuinely intelligent second brain.

Unlike traditional chat-based assistants, Qube is designed around a **low-latency streaming architecture**, combining:

- adaptive cognitive routing
    
- retrieval-augmented generation (RAG)
    
- live web search integration
    
- async long-term semantic memory enrichment
    
- strict RAM-aware execution constraints (~10–15GB usable budget)
    

Inference and RAG stay on-device—**no** third-party chat API. (Optional **Model Manager** downloads talk to Hugging Face only when **you** choose to fetch weights.)

✨ Quick Overview

    🧠 Long-Term Semantic Memory & RAG: Qube doesn't just hold temporary context; it learns. Using a background async worker, Qube extracts Atomic Facts from your conversations, manages contradictions, applies reinforcement scoring, and stores them permanently in LanceDB. This prevents the "Amnesia Bug" and gives Qube true long-term recall alongside your documents.

    ⚡ Real-Time Interruption (Barge-In): Experience true conversational fluidity. Qube supports "Barge-In" capabilities, allowing you to interrupt the assistant mid-sentence by calling it out.

    🤖 **Dual-mode LLM routing:** Choose **Internal Engine (native llama.cpp)** for a self-contained app with no separate server, or **External Server (localhost)** for LM Studio / Ollama-style OpenAI-compatible APIs—same streaming pipeline either way. Intelligent NLP triggers and dashboard toggles for RAG routing.

    🎙️ Lightning-Fast STT: Powered by faster-whisper, Qube offers incredibly fast and accurate Speech-to-Text transcription right on your hardware (excellent on CPU alone).

    🗣️ High-Fidelity TTS: Uses the cutting-edge Kokoro engine for ultra-realistic Text-to-Speech, with over 30 voices included. In the Settings area you can load your own engine if you prefer something like Voxtral or Qwen TTS, but be prepared to keep an eye out on the Dashboard telemetry as these require more beefy hardware like a dedicated GPU (or a solid APU) acceleration.

    📚 Advanced RAG Engine: Built on LanceDB for blazing-fast vector storage and PyMuPDF for aggressive text extraction from complex PDFs, eBooks, and text files.

    🌐 Live Web Search Integration: Qube can break out of its offline shell when explicitly requested. Using the internet_tool, it performs real-time web searches, parses the data into the context window, and provides beautifully formatted, clickable [W] citations right next to your local document sources.

    🎛️ **Responsive native GUI:** A lean **PyQt6** desktop shell (not an Electron wrapper)—so more of your RAM stays available for models and context. Includes a real-time VU meter, dynamic settings, custom wake-word support (currently over 4 different wake-words available), and **Model Manager**: search Hugging Face, browse Editor’s Picks, read READMEs, and download **.gguf** quantizations with **disk-space guardrails** (pre-flight free-space check + safe **.part** cleanup on cancel or failure).

    🎚️ **Hardware controls:** Per-model **GPU offload layers** for the native engine, plus granular audio and generation settings—tuned for real hardware, not abstract “cloud” tiers.


## 🏗️ Deep Dive: Architecture & Features

### 🧠 Dual Memory System (RAG + Long-Term Atomic Memory)

Qube uses two complementary memory layers:

#### 1. Document RAG Memory

- Built on **LanceDB vector storage**.
    
- Ingests PDFs, EPUBs, TXT, and Markdown files.
    
- Retrieves semantic chunks for grounding responses.
    
- Injects retrieved context directly into LLM prompts using sequential numeric citations (e.g., `[1]`).
    

#### 2. Long-Term Atomic Memory (v5.1)

- Extracts durable “facts” from conversations asynchronously.
    
- Runs in a background **QThread Enrichment Worker** that yields to the main LLM to prevent local server deadlocks.
    
- Stores structured atomic memory in LanceDB using a dedicated `qube_memory::%` namespace.
    

**Key properties:**

- Atomic fact extraction (no summaries stored as memory).
    
- JSON-structured memory payloads.
    
- Confidence-based filtering.
    
- Semantic deduplication using vector similarity thresholds before insertion.
    
- Reinforcement system (frequent facts become stronger).
    
- Contradiction handling (schema-safe updates to existing facts instead of duplicating them).
    

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

---

### 🎙️ Speech-to-Text (STT)

- Powered by `faster-whisper`.
    
- CPU-efficient transcription pipeline.
    
- Streaming-compatible chunk processing.
    
- Optimized for low-latency voice input.
    

---

### 🗣️ High-Fidelity Text-to-Speech (TTS)

- Uses **Kokoro ONNX engine**.
    
- Micro-chunk streaming for fast interrupt response.
    
- Strips bracketed citations via regex before audio synthesis to ensure fluid speech.
    
- Designed for real-time conversational playback.
    

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


---


## 🚀 Getting Started

### Prerequisites
* Python 3.12 or higher (Linux/Windows)
* **LLM backend (pick one):** use Qube’s **Internal Engine** with downloaded **.gguf** models (see **Model Manager**), *or* run **[LM Studio](https://lmstudio.ai/)** / **[Ollama](https://ollama.com/download)** (or any OpenAI-compatible server on `localhost`, e.g. `:1234` / `:11434`) if you prefer **External Server** mode.
* **Hardware:** Minimum 16GB RAM (20GB recommended to avoid disk swapping).
* **Suggested SLM at 16GB RAM:** Nemotron 3 Nano 4B
* A microphone and speakers for STT & TTS interactions

### 1. Installation

Clone the repository and navigate into the directory:
```bash
git clone [https://github.com/dagaza/Qube.git](https://github.com/dagaza/Qube.git)
cd Qube
```

Create a virtual environment and activate it:

Bash

```
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

Bash

```
pip install -r requirements.txt
```

### 2. First Run & Auto-Download

Start the application:

Bash

```
python main.py
```

_Note: On the very first run, Qube will automatically connect to Hugging Face and download the necessary Kokoro TTS models (approx. 400MB) directly into the `models/` directory. Optional chat weights are **not** pulled automatically—use **Model Manager** when you want to fetch **.gguf** files (with on-device disk checks). Grab a coffee while TTS finishes, then you’re ready to chat._

---

## 🛠️ How to Use Qube

### Voice Interaction

1. **Inference:** In **Settings**, choose **Internal Engine** (after selecting a **.gguf** in **Model Manager**) *or* **External Server** and start your local LLM server (e.g., LM Studio or Ollama) if you use external mode.
    
2. Say the wake word (Default: _"Hey Alexa"_) (training your own custom wake word in the app is coming soon).
    
3. Speak your prompt. Qube uses a smart sliding-window VAD (Voice Activity Detection) threshold—it listens as long as you speak and processes your request after 2 seconds of silence. You can change this cut-off time setting at any time from the Settings screen.
    

### RAG (Document Retrieval)

Want Qube to answer questions based on a specific book or PDF?

1. Open the **Library View** and ingest your documents (PDF, EPUB, TXT, or MD). Qube will parse and embed them into the local LanceDB store.
    
2. Use the **RAG Toggle** in the tools pane or use trigger phrases like _"According to my files..."_ which **you can define yourself** in the settings area! Yes, you heard that right, Qube has NLP-triggered RAG functionality inside! 
    
3. Ask your question. Qube will retrieve the most relevant chunks and inject them into the LLM's context window, which also showing you the sources and citations
    
4. **Conversational Turn:** Because Qube saves context to its internal "RAG Memory," you can ask follow-up questions about your documents without re-triggering a search.
---

## 🏗️ Architecture Stack

- **UI Framework:** PyQt6 (Frameless, Thread-Isolated)
    
- **Chat inference (internal mode):** llama-cpp-python (**GGUF**), long-lived native worker thread + streaming queue handoff to the main LLM pipeline; execution policy + template-aware prompt representation for logs/validation; **template_override** (built-in name heuristics) + **model_override_store** (learned JSON at **`~/.qube/model_overrides.json`**) adjust merged stop lists and assistant anchoring in the prompt bundle only; optional one-shot ablation on model load for behavior classification when no persisted self-heal entry exists (diagnostic **`python -m tools.run_ablation`** can also write the same store).
    
- **Vector Database:** LanceDB (Disk-native, zero-copy)
    
- **Embeddings:** Nomic v1.5 GGUF via llama-cpp-python (Vulkan/CPU).
    
- **Wake Word:** OpenWakeWord
    
- **STT:** Faster-Whisper
    
- **TTS:** Kokoro-ONNX with Micro-Chunking

## 💖 Support the Project

Qube is built with passion and released as free, open-source software. If this app makes your life easier, helps you study, or saves you time, consider supporting its continued development!

* ☕ **[Support me on Patreon](https://patreon.com/Dagaza)** ---

Any help in the form or feedback, feature requests, issue reporting, or any other type of participatory involvement with the project is equally appreciated! <3 

## 🙏 Acknowledgements

Qube stands on the shoulders of giants. A massive thank you to the brilliant developers and teams behind the open-source stack that makes this app possible:

* **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M):** For the breathtakingly realistic TTS engine (by Hexgrad).
* **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper):** For blazing-fast speech recognition (by SYSTRAN).
* **[Nomic AI](https://www.nomic.ai/)**: For the high-performance **Nomic Embed v1.5** model powering our hardware-accelerated RAG pipeline.
* **[LanceDB](https://lancedb.com/):** For the incredibly efficient, serverless vector database.
* **[PyMuPDF](https://pymupdf.readthedocs.io/):** For the industrial-strength document parsing.
* **[OpenWakeWord](https://github.com/dscripka/openWakeWord):** For lightweight, customizable wake word detection.
* **[Hugging Face](https://huggingface.co/):** For the Hub APIs and model artifacts used by **Model Manager** (search, READMEs, **.gguf** downloads).
* **[LM Studio](https://lmstudio.ai/) & [Ollama](https://ollama.com/):** For optional external local LLM hosting when you’re not using the built-in engine.
* **[PyQt6](https://riverbankcomputing.com/software/pyqt/):** For the robust framework powering the Qube UI.
* **All the wonderful people around me who have encouraged me with the project, you rock!**

---

## 📄 License

This project is licensed under the **MIT License**.

You are completely free to use, modify, distribute, and even use this code in commercial projects. The only requirement is that you **must include the original copyright notice and permission notice** (giving proper attribution to this repository) in any copy or substantial reuse of the software. See the `LICENSE` file for more details.
