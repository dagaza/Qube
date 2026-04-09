# Qube: Local Hardware-Accelerated AI Assistant

| **Dark Theme** | **Light Theme** |
| :---: | :---: |
| ![Dark Theme](assets/screenshots/qube_dark_mode.png) | ![Light Theme](assets/screenshots/qube_light_mode.png) |

Qube is a fully local, privacy-first, voice-to-voice AI desktop assistant built with a complete multimodal pipeline. It operates entirely offline with real-time conversational streaming capabilities. By integrating state-of-the-art voice processing and a powerful Retrieval-Augmented Generation (RAG) engine, Qube allows you to interact directly with your personal documents without any data leaving your hardware.

## ✨ Key Features

* **🧠 Conversational Memory & Dynamic RAG:** Qube doesn't just answer; it remembers. Using a custom SQLite "RAG Memory" injection, the assistant maintains context across multiple turns, preventing the "Amnesia Bug" common in basic RAG implementations.

* **⚡ Real-Time Interruption (Barge-In):** Experience true conversational fluidity. Qube supports "Barge-In" capabilities, allowing you to interrupt the assistant mid-sentence by calling it out.

* **🤖 Local LLM Routing:** Interfaces directly with local LLM providers (like LM Studio) for private, fast text generation. Features intelligent NLP triggers and UI dashboard toggles for RAG routing.

* **🎙️ Lightning-Fast STT:** Powered by `faster-whisper`, Qube offers incredibly fast and accurate Speech-to-Text transcription right on your hardware (excellent on CPU alone).

* **🗣️ High-Fidelity TTS:** Uses the cutting-edge **Kokoro** engine for ultra-realistic Text-to-Speech, with over 30 voices included. In the Settings area you can load your own engine if you prefer something like Voxtral or Qwen TTS, but be prepared to keep an eye out on the Dashboard telemetry as these require more beefy hardware like a dedicated GPU (or a solid APU) acceleration.

* **📚 Advanced RAG Engine:** Built on **LanceDB** for blazing-fast vector storage and **PyMuPDF** for aggressive text extraction from complex PDFs, eBooks, and text files. 

* **🎛️ Responsive GUI:** A clean, multithreaded PyQt6 interface featuring a real-time VU meter, dynamic settings, and custom wake-word support (currently over 4 different wake-words available).

---

## 🚀 Getting Started

### Prerequisites
* Python 3.12 or higher (Linux/Windows)
* [LM Studio](https://lmstudio.ai/) or [Ollama] (https://ollama.com/download) (or a compatible local LLM server running on `localhost:1234`)
* **Hardware:** Minimum 16GB RAM (20GB recommended to avoid disk swapping).
* A microphone and speakers

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

---

## 📄 License

This project is licensed under the **MIT License**.

You are completely free to use, modify, distribute, and even use this code in commercial projects. The only requirement is that you **must include the original copyright notice and permission notice** (giving proper attribution to this repository) in any copy or substantial reuse of the software. See the `LICENSE` file for more details.
