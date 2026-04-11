# Qube: Local Hardware-Accelerated AI Assistant

| **Dark Theme** | **Light Theme** |
| :---: | :---: |
| ![Dark Theme](assets/screenshots/qube_dark_mode.png) | ![Light Theme](assets/screenshots/qube_light_mode.png) |

Qube is a fully local, privacy-first, voice-to-voice AI desktop assistant built on a multithreaded, streaming-first pipeline. Operating entirely offline under a strict memory budget, it integrates state-of-the-art voice processing, adaptive cognitive routing, and asynchronous semantic memory enrichment directly into your hardware environment. Bring your own local LLM engine, load your files, and experience a genuinely intelligent second brain.

Unlike traditional chat-based assistants, Qube is designed around a **low-latency streaming architecture**, combining:

- adaptive cognitive routing
    
- retrieval-augmented generation (RAG)
    
- live web search integration
    
- async long-term semantic memory enrichment
    
- strict RAM-aware execution constraints (~10–15GB usable budget)
    

Everything runs locally with no external API dependency.

✨ Quick Overview

    🧠 Long-Term Semantic Memory & RAG: Qube doesn't just hold temporary context; it learns. Using a background async worker, Qube extracts Atomic Facts from your conversations, manages contradictions, applies reinforcement scoring, and stores them permanently in LanceDB. This prevents the "Amnesia Bug" and gives Qube true long-term recall alongside your documents.

    ⚡ Real-Time Interruption (Barge-In): Experience true conversational fluidity. Qube supports "Barge-In" capabilities, allowing you to interrupt the assistant mid-sentence by calling it out.

    🤖 Local LLM Routing: Interfaces directly with local LLM providers (like LM Studio) for private, fast text generation. Features intelligent NLP triggers and UI dashboard toggles for RAG routing.

    🎙️ Lightning-Fast STT: Powered by faster-whisper, Qube offers incredibly fast and accurate Speech-to-Text transcription right on your hardware (excellent on CPU alone).

    🗣️ High-Fidelity TTS: Uses the cutting-edge Kokoro engine for ultra-realistic Text-to-Speech, with over 30 voices included. In the Settings area you can load your own engine if you prefer something like Voxtral or Qwen TTS, but be prepared to keep an eye out on the Dashboard telemetry as these require more beefy hardware like a dedicated GPU (or a solid APU) acceleration.

    📚 Advanced RAG Engine: Built on LanceDB for blazing-fast vector storage and PyMuPDF for aggressive text extraction from complex PDFs, eBooks, and text files.

    🌐 Live Web Search Integration: Qube can break out of its offline shell when explicitly requested. Using the internet_tool, it performs real-time web searches, parses the data into the context window, and provides beautifully formatted, clickable [W] citations right next to your local document sources.

    🎛️ Responsive GUI: A clean, multithreaded PyQt6 interface featuring a real-time VU meter, dynamic settings, and custom wake-word support (currently over 4 different wake-words available).

---
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

### 🤖 Local LLM Routing

- Fully local inference via LM Studio / Ollama compatible servers.
    
- OpenAI-style streaming interface support.
    
- Streaming-first response design (TTFB optimized).
    
- Wrapped in strict timeouts and `finally` blocks to guarantee UI unlocking even if the local server crashes mid-stream.
    

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

- Built with PyQt6 using a frameless, `qt_material` design.
    
- Fully asynchronous worker architecture (UI thread is strictly isolated).
    
- Escapes model citations into native Markdown (e.g., `[1]`) to bypass `heightForWidth` geometry recalculation loops that would freeze the Qt layout engine.
    
- Real-time telemetry (latency, VU meter, system stats).
    
- Wake-word support (multiple configurable triggers).


---


## 🚀 Getting Started

### Prerequisites
* Python 3.12 or higher (Linux/Windows)
* [LM Studio](https://lmstudio.ai/) or [Ollama] (https://ollama.com/download) (or a compatible local LLM server running on `localhost:1234`)
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

_Note: On the very first run, Qube will automatically connect to Hugging Face and download the necessary Kokoro TTS models (approx. 400MB) directly into the `models/` directory. Grab a coffee, and it will boot up as soon as the download finishes!_

---

## 🛠️ How to Use Qube

### Voice Interaction

1. Start your local LLM server (e.g., LM Studio or Ollama).
    
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
* **[LM Studio] & [Ollama]:** For making local LLM hosting accessible to everyone.
* **[PyQt6](https://riverbankcomputing.com/software/pyqt/):** For the robust framework powering the Qube UI.
* **All the wonderful people around me who have encouraged me with the project, you rock!**

---

## 📄 License

This project is licensed under the **MIT License**.

You are completely free to use, modify, distribute, and even use this code in commercial projects. The only requirement is that you **must include the original copyright notice and permission notice** (giving proper attribution to this repository) in any copy or substantial reuse of the software. See the `LICENSE` file for more details.
