# Qube

<p align="center">
  <img src="assets/logos/qube_logo_256.png" alt="Qube logo" width="80">
</p>

**Your private, voice-first AI assistant** — runs on your machine, remembers what matters, and grounds answers in your files and trusted sources.

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="https://dagaza.github.io/Qube/">Landing page</a> ·
  <a href="#features">Features</a> ·
  <a href="#see-it-in-action">See it in action</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#built-in-help">Help</a> ·
  <a href="#support-the-project">Support</a>
</p>

| **Dark theme** | **Light theme** |
| :---: | :---: |
| ![Conversations — dark theme](assets/screenshots/qube_conversations_dark_mode.png) | ![Conversations — light theme](assets/screenshots/qube_conversations_light_mode.png) |

---

## What is Qube?

Qube is a desktop AI assistant built for **privacy, voice, and real work on your own hardware**. Talk naturally, chat with text, search your documents, and let Qube build a memory of what you care about — all without sending conversations to a cloud chat API.

Inference stays local. Optional web and research tools run **only when you ask**. Your Library, memories, and models live on your machine.

---

## What makes Qube different?

**LM Studio** is an excellent model runner — browse GGUF weights, chat, serve an API, plug in MCP. **SillyTavern** is a deep LLM frontend for prompt craft, lorebooks, and character workflows. **Odysseus** is a broad self-hosted workspace — agents, email, calendar, Docker. Qube can even use LM Studio or Ollama as its backend.

**Qube is built for a different job:** a **voice-first desktop assistant** that aims to be **approachable on first install** — Recommended preset downloads sidecar, **search embeddings**, **voice models**, and an optional **main LLM** in one consent flow — while still routing each turn to the right knowledge: your **Library**, **memory**, or **live institutional sources**. You stay in control with **`@` composer tools**, **`@help`** inside the app, **`?` guided tours**, numbered citations, and a **Memory Manager** you can edit.

Competitors like LM Studio, SillyTavern, and Odysseus excel at **models, prompts, or workspace breadth**; they generally expect you to **bring or wire your own stack** and read **external docs** when stuck ([details](docs/user/competitive-landscape.md#verified-onboarding--enthusiast-vs-approachable)).

What that looks like in practice:

- **First-run Recommended preset** — sidecar cognition, balanced **embedding** search, **Whisper** + **Kokoro** voice, optional **main chat model** — with disk/memory feasibility checks (Advanced path for alternates)
- **`@help` + guided tours** — shipped help in **Library → Qube**, searchable from chat; **`?`** on every major screen (competitors link out to web wikis)
- **`@` composer routing** — attach `@library`, `@evidence`, `@finance`, `@legal`, `@research`, `@memory`, or your own **`@[tool:user:…]` presets**; add **`@[skill:…]`** for reasoning without changing route
- **Cognitive router** — Qube infers Memory vs Library vs web vs plain chat from how you ask (override anytime with `@` tokens or toggles); weak hits are dropped so the model is not forced to fake citations
- **Library as a real corpus** — ingest PDFs/EPUBs, preview indexed text, **Chat with document**, and browse shipped help under **Library → Qube** (searchable via **`@help`** — same pipeline, separate scope)
- **Live Sources, not just “search the web”** — cited pathways to Wikipedia/trusted catalogs, scientific literature, SEC filings, case law, and async **`@research`** evidence reports
- **Memory with an editor** — facts distilled over time, reviewable in **Memory Manager**; delete means delete
- **Full voice loop** — wake word, push-to-talk, streaming TTS, **barge-in**, optional **Desktop Companion** orb

[Full competitive comparison →](docs/user/competitive-landscape.md)

---

## Why Qube?

- **Private by default** — chat, documents, and long-term memory stay on your device
- **See what happened** — **Telemetry** dashboard, **INSPECT RETRIEVAL** on citations, and opt-in audit logs (local only — not cloud analytics)
- **Control web search** — default **private** tier (DuckDuckGo + Wikipedia); optional **your SearXNG** instance
- **Voice-first** — wake word, push-to-talk, and barge-in for natural back-and-forth
- **Smart routing** — cognitive router picks Memory, Library, web, or chat; `@` tools override when you want precision
- **Ready after install** — Recommended bootstrap bundles embeddings, voice, and an optional main model — not a blank shell
- **Help inside the app** — **`@help`** and **`?` tours**; no hunting through external wikis for settings locations
- **Remembers you** — long-term **Memory Manager** with automatic fact extraction, tiers, and delete-that-sticks (negative list); separate from **Library** documents
- **Grounded answers** — Library, Live Sources, and clickable citations — not guesswork dressed up as confidence
- **Your hardware** — built-in GGUF engine or plug in LM Studio / Ollama

---

## Features

<a id="features"></a>

**Voice** — Fast speech-to-text and natural text-to-speech with 30+ voices. Interrupt the assistant mid-sentence without breaking the flow.

**Composer & `@` routing** — Per-message control: `@library`, `@file`, `@evidence`, `@finance`, `@legal`, `@research`, `@internet`, `@memory`, `@help`, custom **`@[tool:user:…]` presets**, and **`@[skill:…]`** reasoning frameworks. The cognitive router handles everyday phrasing; attach `@` when you want a specific pathway.

**Library** — A persistent document corpus: ingest PDFs, EPUBs, and text; search titles and indexed body text; preview from the vector index; **Chat with document** prefills `@file`. Shipped help lives in **Library → Qube** and is reachable via **`@help`** without mixing into your uploads.

**Memory Manager** — Long-term facts distilled from chat — preferences, projects, knowledge — with tier filters, edit/flag/delete, export, and a negative list so deleted memories stay gone.

**Live Sources** — Institutional adapters beyond generic web search: trusted/Wikipedia, scientific literature, SEC EDGAR, U.S. case law, page **fetch**, and multi-step **`@research`** reports (async, non-blocking). **58+ adapters** with per-source toggles; build your own via **My knowledge** presets and **Custom sources** (REST, GraphQL, MCP, …).

**Transparency** — **Advanced Telemetry** (local hardware + routing stats), per-reply **INSPECT RETRIEVAL**, and **Settings → Advanced** diagnostic logs with redaction options. Web discovery defaults to **private** search (DuckDuckGo + Wikipedia); point at **your SearXNG** when you want self-hosted SERP.

**Model Manager** — Search Hugging Face, browse curated picks, read model READMEs in-app, and download **.gguf** quantizations with disk-space guardrails. Run natively or point at an external server.

**Desktop Companion** — Optional floating orb for quick voice turns and glanceable status without bringing the main window forward.

**Built-in help** — Full guides in **Library → Qube**, searchable with **`@[tool:help]`** from chat (same retrieval pipeline as your docs, separate scope). **`?` guided tours** on every major screen — unlike competitors that primarily link to external documentation sites.

| **Library** | **Model Manager** | **Telemetry** |
| :---: | :---: | :---: |
| ![Library — dark theme](assets/screenshots/qube_library_dark_mode.png) | ![Model Manager](assets/screenshots/qube_model_manager_dark_mode.png) | ![Telemetry](assets/screenshots/qube_telemetry_dark_mode.png) |

> **Before launch:** capture a Desktop Companion orb screenshot for this section — see [launch documentation guidelines](docs/launch_documentation_guidelines.md) Phase 4.

---

## Quick Start

<a id="quick-start"></a>

### Download (recommended)

| Platform | Install |
|----------|---------|
| **Windows** | [`winget install -e --id dagaza.Qube`](https://github.com/dagaza/Qube/releases) or `choco install qube` |
| **macOS** | Download the `.dmg` for your Mac from [GitHub Releases](https://github.com/dagaza/Qube/releases) |
| **All platforms** | Latest installer or bundle from [GitHub Releases](https://github.com/dagaza/Qube/releases) |

### First launch

1. **Complete setup** — On first run, Qube shows a **Recommended** preset: required sidecar + **search embeddings**, plus **Whisper**, **Kokoro TTS**, and (by default) a **main chat model** sized for ~16 GB RAM — with disk/memory feasibility checks. Uncheck what you do not want, or switch to **Advanced** for alternates.
2. **Or use an external backend** — Skip the main LLM download and point Qube at **LM Studio** or **Ollama** under **Settings → AI & Models**; add more weights anytime in **Model Manager**.
3. **Start talking** — Enable voice input, choose a wake word in **Settings → Voice & Audio**, or type in **Conversations**. Press **?** on any screen for a guided tour, or ask **`@help`** in chat.

> **Tip:** At 16 GB RAM, start with a small model (for example Nemotron 3 Nano 4B). See **Settings → Help** to enable hardware-fit suggestions in Model Manager.

### From source (developers)

See **[Install from source](docs/user/install-from-source.md)** for clone, venv, GPU builds, and developer flags.

Quick start:

```bash
git clone https://github.com/dagaza/Qube.git && cd Qube
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python main.py
```

---

## See it in action

<a id="see-it-in-action"></a>

1. **Ask by voice** — Say your wake phrase or use push-to-talk, then ask a question. Qube transcribes, thinks, and replies out loud.
2. **Ground on your files** — Drop a PDF into **Library**, then ask naturally or attach **`@[tool:library]`** / **`@[file:…]`**. Citations link back to source chunks.
3. **Route with precision** — Try **`@[tool:evidence]`** for papers, **`@[tool:finance]`** for SEC filings, or **`@[tool:research]`** for an async evidence report — or let the router infer from your wording.
4. **Curate memory** — Open **Memory Manager** to see what Qube filed away; edit or delete anything you do not want kept.
5. **Stay in flow** — Enable the **Desktop Companion** orb for quick turns while another app is fullscreen.

---

## Built-in help

<a id="built-in-help"></a>

Qube ships with a full help library — no browser required.

- Open **Library → Qube** to browse guides, workflows, and troubleshooting.
- Type **`@[tool:help]`** in chat and ask how to do something (*"How do I set GPU layers?"*, *"Memory vs Library?"*).

Guided tours (**?** buttons) on each screen walk you through the layout step by step.

More workflows: [How to use Qube](docs/user/how-to-use.md).

---

## System requirements

| | Minimum | Recommended |
|---|---------|-------------|
| **RAM** | 16 GB | 20 GB |
| **OS** | Windows 10+, macOS 12+ (Apple Silicon or Intel), Linux (source) | Same |
| **Storage** | ~2 GB for app + voice models; plan extra for each chat model | SSD strongly recommended |
| **Audio** | Microphone and speakers (or headset) for voice | Same |
| **GPU** | Optional — speeds up the internal engine via GPU offload layers | Discrete GPU or Apple Silicon with enough VRAM for your chosen model |

Full hardware guidance (models, GPU paths, storage): **[System requirements](docs/user/system-requirements.md)**.

Qube uses a native **PyQt6** desktop shell — not Electron and not a browser tab — so less of your **16 GB budget** goes to UI overhead before models load. Inference still runs in native code (`llama-cpp-python`, Whisper, Kokoro); token speed depends on your GPU/CPU and quant, not on “Python magic.” [Runtime comparison →](docs/user/competitive-landscape.md#verified-runtime-ui-shell-and-ram)

---

## Documentation

<a id="documentation"></a>

| Audience | Start here |
|----------|------------|
| **Users** | [docs/user/](docs/user/README.md) — install, requirements, workflows |
| **Positioning** | [Competitive landscape](docs/user/competitive-landscape.md) — vs LM Studio, SillyTavern, Odysseus |
| **In-app** | **Library → Qube** or **`@[tool:help]`** (see below) |
| **Contributors** | [docs/architecture/](docs/architecture/README.md) — memory, pipeline, stack |
| **Contributing** | [CONTRIBUTING.md](CONTRIBUTING.md) — setup, tests, PRs |
| **Launch doc playbook** | [docs/launch_documentation_guidelines.md](docs/launch_documentation_guidelines.md) — re-run before official launch |
| **Landing page** | [dagaza.github.io/Qube](https://dagaza.github.io/Qube/) · [setup](docs/pages.md) |
| **Social preview** | [assets/social/qube-social-preview.png](assets/social/qube-social-preview.png) — upload in repo Settings |
| **Release notes** | [CHANGELOG.md](CHANGELOG.md) — what changed in each version |

The pre–launch rewrite README (453 lines of technical detail) is preserved at [docs/archive/readme-pre-launch-rewrite.md](docs/archive/readme-pre-launch-rewrite.md).

**What's new:** [v1.0.1](CHANGELOG.md#101---2026-06-28) — first-run bootstrap, phased model downloads, WinGet first-launch fixes.

---

## Support the project

<a id="support-the-project"></a>

Qube is free, open-source software built with care. If it saves you time or helps you learn, consider supporting continued development:

- ☕ **[Support on Patreon](https://patreon.com/Dagaza)**
- 🐛 **[Report a bug or request a feature](https://www.qubeapp.eu)** — or use **Settings → Contact & Feedback** in the app
- 💬 **GitHub [Issues](https://github.com/dagaza/Qube/issues)** — bug reports and discussions welcome

---

## Acknowledgements

Qube stands on the shoulders of excellent open-source projects:

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) · [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) · [Nomic Embed](https://www.nomic.ai/) · [LanceDB](https://lancedb.com/) · [PyMuPDF](https://pymupdf.readthedocs.io/) · [OpenWakeWord](https://github.com/dscripka/openWakeWord) · [Hugging Face Hub](https://huggingface.co/) · [LM Studio](https://lmstudio.ai/) · [Ollama](https://ollama.com/) · [PyQt6](https://riverbankcomputing.com/software/pyqt/) · [llama.cpp](https://github.com/ggerganov/llama.cpp)

Thank you to everyone who encouraged this project along the way.

---

## License

This project is licensed under the **MIT License**. You may use, modify, and distribute it freely — including in commercial projects — as long as you include the original copyright notice. See [`LICENSE`](LICENSE) for details.

---

## Developers

Clone, test, and contribute: **[CONTRIBUTING.md](CONTRIBUTING.md)** · [Install from source](docs/user/install-from-source.md) · [Architecture](docs/architecture/README.md)
