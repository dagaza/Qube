# Install Qube from source

For most users, prefer a packaged install from [GitHub Releases](https://github.com/dagaza/Qube/releases) (Windows installer, macOS `.dmg`, Linux AppImage/`.deb`) or `winget install -e --id dagaza.Qube` / `choco install qube`. Linux packages: [install-linux.md](install-linux.md).

Use this guide when you are developing Qube, running on Linux, or need a bleeding-edge checkout.

---

## Prerequisites

- **Python 3.12+** (3.13 recommended; see `pyproject.toml`)
- **Git**
- **16 GB RAM** minimum (**20 GB** recommended to avoid swap during model load)
- **Microphone and speakers** (or headset) for voice features
- **LLM backend (pick one):**
  - **Internal Engine** — download a `.gguf` via in-app **Model Manager**, or
  - **External Server** — [LM Studio](https://lmstudio.ai/), [Ollama](https://ollama.com/download), or any OpenAI-compatible server on `localhost` (e.g. `:1234` / `:11434`)

At **16 GB RAM**, a small model such as **Nemotron 3 Nano 4B** is a practical starting point.

---

## Clone and install

```bash
git clone https://github.com/dagaza/Qube.git
cd Qube
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Optional: GPU-accelerated internal engine (Linux)

For AMD / Intel Vulkan or NVIDIA CUDA builds of `llama-cpp-python`:

```bash
./scripts/install_llama_cpp_gpu.sh
```

See script comments for build dependencies. Default parallelism is memory-safe on 16 GB machines.

### Optional: development dependencies

```bash
pip install -r requirements-dev.txt
```

---

## First run

```bash
python main.py
```

On the **first run**, Qube downloads Kokoro TTS weights (roughly **400 MB**) into your models directory. **Chat `.gguf` weights are not pulled automatically** — use **Model Manager** when you are ready.

Packaged Windows builds use a first-run bootstrap consent dialog and phased downloads; source runs follow the same model layout under `~/.qube/` (see `core/paths.py`).

---

## Configure inference

1. Open **Settings → AI & Models**.
2. Choose **Internal Engine (native)** or **External Server (localhost)**.
3. **Internal:** download/load a `.gguf` in **Model Manager**.
4. **External:** start LM Studio or Ollama, then set the server URL in Settings.

Full walkthrough: in-app **Set up local models** workflow (`Library → Qube → workflows/set-up-local-models.md`).

---

## Developer flags

| Flag | Purpose |
|------|---------|
| `python main.py --routing-debug` | Detached routing debug side tool |
| `python main.py --mock-bootstrap-download` | Mock bootstrap downloads (testing only) |

See [logging and diagnostics](../logging_and_diagnostics.md) for environment variables and log locations.

---

## Verify your setup

```bash
pytest tests/ -m "not packaging" -q
```

Full CI parity: [local_validation.md](../local_validation.md).

---

## Related

- [System requirements](system-requirements.md)
- [How to use Qube](how-to-use.md)
- [README.md](../../README.md) — project overview and download links
