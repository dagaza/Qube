# Qube: Local Hardware-Accelerated AI Assistant

Qube is a fully local, privacy-first AI assistant built with a complete multimodal pipeline. It integrates state-of-the-art voice processing, hardware-accelerated inference, and a powerful Retrieval-Augmented Generation (RAG) engine to let you talk directly to your personal documents without sending a single byte of data to the cloud.

## ✨ Key Features

* **🧠 Local LLM Routing:** Interfaces directly with local LLM providers (like LM Studio) for private, fast text generation. Features intelligent NLP triggers and UI dashboard toggles for RAG routing.

* **🎙️ Lightning-Fast STT:** Powered by `faster-whisper`, Qube offers incredibly fast and accurate Speech-to-Text transcription right on your hardware (excellent on CPU alone).

* **🗣️ High-Fidelity TTS:** Uses the cutting-edge **Kokoro** engine for ultra-realistic Text-to-Speech. Features a smart **Auto-Downloader** that seamlessly fetches the required model weights on the first boot to keep the repository lightweight. In the Settings area you can load your own engine if you prefer something like Voxtral or Qwen TTS, but be prepared to keep an eye out on the Dashboard telemetry as these require more beefy hardware like GPU (or a solid APU) acceleration.

* **📚 Advanced RAG Engine:** Built on **LanceDB** for blazing-fast vector storage and **PyMuPDF** for aggressive text extraction from complex PDFs, eBooks, and text files. 

* **🎛️ Responsive GUI:** A clean, multithreaded PyQt6 interface featuring a real-time VU meter, dynamic settings, and custom wake-word support (for the moment Hey Jarvis, but more and custom options will come soon).

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* [LM Studio](https://lmstudio.ai/) or [Ollama] (https://ollama.com/download) (or a compatible local LLM server running on `localhost:1234`)
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

1. Start your local LLM server (e.g., LM Studio).
    
2. Say the wake word (Default: _"Hey Jarvis"_) (more & custom options coming soon - train your own custom wake word in the app).
    
3. Speak your prompt. Qube uses a smart sliding-window VAD (Voice Activity Detection) threshold—it listens as long as you speak and processes your request after 2 seconds of silence. You can change this setting at any time from the Settings screen.
    

### RAG (Document Retrieval)

Want Qube to answer questions based on a specific book or PDF?

1. Open the **Settings Dialog** in the UI.
    
2. Click **Add Document** and select your PDF, EPUB, TXT, or MD file. Qube will parse, chunk, and embed the text into the local LanceDB vector database.
    
3. Check the **"🧠 Force Document Search"** box (or use trigger phrases like _"Check my notes"_ when speaking).
    
4. Ask your question. Qube will retrieve the most relevant chunks and inject them into the LLM's context window.
    

---

## 🏗️ Architecture Stack

- **UI Framework:** PyQt6
    
- **Vector Database:** LanceDB
    
- **Embeddings:** Sentence-Transformers / ONNX
    
- **Wake Word:** OpenWakeWord
    
- **STT:** Faster-Whisper
    
- **TTS:** Kokoro-ONNX / Piper

## 💖 Support the Project

Qube is built with passion and released as free, open-source software. If this app makes your life easier, helps you study, or saves you time, consider supporting its continued development!

* ☕ **[Support me on Patreon](https://patreon.com/Dagaza)** ---

## 🙏 Acknowledgements

Qube stands on the shoulders of giants. A massive thank you to the brilliant developers and teams behind the open-source stack that makes this app possible:

* **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M):** For the breathtakingly realistic TTS engine (by Hexgrad).
* **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper):** For blazing-fast speech recognition (by SYSTRAN).
* **[LanceDB](https://lancedb.com/):** For the incredibly efficient, serverless vector database.
* **[PyMuPDF](https://pymupdf.readthedocs.io/):** For the industrial-strength document parsing.
* **[OpenWakeWord](https://github.com/dscripka/openWakeWord):** For lightweight, customizable wake word detection.
* **[LM Studio] & [Ollama]:** For making local LLM hosting accessible to everyone.
* **[PyQt6](https://riverbankcomputing.com/software/pyqt/):** For the robust framework powering the Qube UI.

---

## 📄 License

This project is licensed under the **MIT License**.

You are completely free to use, modify, distribute, and even use this code in commercial projects. The only requirement is that you **must include the original copyright notice and permission notice** (giving proper attribution to this repository) in any copy or substantial reuse of the software. See the `LICENSE` file for more details.
