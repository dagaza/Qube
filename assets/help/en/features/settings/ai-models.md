# AI & Models

## Common questions

- Where are GPU layers in settings?
- What is the difference between Internal Engine and External Server?
- How do I connect to LM Studio or Ollama?
- Which chat template should I use for my local model?
- How do I tune temperature and context limit?
- What is the difference between context limit and max reply tokens?
- How does chat history interact with Memory?

## What it is

**AI & Models** is the control centre for how Qube runs language models. Choose between **Internal Engine (native)** (local `.gguf` models on your machine) and **External Server (localhost)** (OpenAI-compatible servers such as LM Studio or Ollama). Configure startup behaviour, generation limits, chat style, reasoning skills, hardware tuning, chat templates, and optional auxiliary cognition models.

Internal inference keeps data on-device; External inference forwards prompts to a server you configure. **Think** mode and reasoning display live in **Conversations**, not on this page.

## Generation parameters (how Qube uses them)

These settings apply to **main chat completions** in Conversations (not sidecar/auxiliary models). See [Generation parameters FAQ](../../faq/generation-parameters.md) for the full educative guide; essentials:

| Control | Qube behaviour |
|---------|----------------|
| **Temperature** (default **0.8**) | Baseline sampling randomness for replies. Qube may scale it down slightly on high-risk turns (long threads, structured lists) via an internal risk profile. |
| **Context limit** (default **32,000** tokens) | Shared token pool for system text, retrieval, **Chat history**, your message, and the reply. On the Internal Engine this sets `n_ctx` and **reloads the model** when changed. |
| **Limit maximum reply length** + **Max reply tokens** (default **on** / **4,096**) | Caps new tokens per assistant message. When off, replies grow until the context window is full minus the prompt. The hint below these controls estimates effective reply room. |
| **Chat history** (default **10** messages) | Sliding window of recent session messages in each prompt — not the same as Memory Manager. Synced with the Conversations tools panel. |
| **Show advanced generation settings** | Top-K, Top-P, Min-P, repeat penalty, presence penalty — passed to native and external backends. |

**Conversations → tools panel → GENERATION PARAMETERS** exposes Temperature, Context Limit, and Chat History only (two-way sync with this page). **Max reply tokens** is Settings-only.

## Hardware tuning (Internal Engine only)

**GPU offload layers** and **CPU thread pool** live under **Show advanced hardware settings**. They apply only when **Internal Engine (native)** is selected — External Server hosts manage their own GPU/CPU. Changing layers or threads **reloads the active `.gguf`**. See [Hardware tuning FAQ](../../faq/hardware-tuning-internal-engine.md).

## Where to find it

Open **Settings → AI & Models** (settings section `ai.models`). Press **?** for the guided tour (`settings.ai_models`).

**GPU offload layers** and **CPU thread pool** appear under **Hardware tuning** after you enable **Show advanced hardware settings**. **Chat template (internal)** appears after **Show advanced chat template settings**.

## Also called

AI models settings, native engine, local LLM, external server, GPU offload, LM Studio, Ollama, NATIVE ENGINE & LOCAL LIBRARY

## How to…

1. **Select an engine** — Use **AI Engine** to switch between **Internal Engine (native)** and **External Server (localhost)**, then pick **External Provider** when using a local server.
2. **Manage local model files** — Under **Local models**, review **Model storage**, files **On this device**, and the **Active model** label. Use **Use selected**, **Refresh**, or **Delete** as needed.
3. **Load on startup** — Toggle **Load last used model on startup** if you want Qube to reopen your last Internal Engine model automatically.
4. **Tune generation** — Adjust **Temperature**, **Context limit**, **Limit maximum reply length**, and **Chat history**. Read [Generation parameters FAQ](../../faq/generation-parameters.md) for how these share the context window. Unlock **Show advanced generation settings** for sampling penalties.
5. **Tune GPU layers** — Enable **Show advanced hardware settings**, then adjust **GPU offload layers** and **CPU thread pool**. Read [Hardware tuning FAQ](../../faq/hardware-tuning-internal-engine.md) for VRAM tradeoffs and unified-memory notes.
6. **Match chat templates** — Enable **Show advanced chat template settings**, pick **Chat template (internal)**, and use **Reset** if a model family needs the default again.
7. **Download or load from Model Manager** — Use [Model Manager](../../features/model-manager.md) to fetch `.gguf` files; loading also works from the Conversations tools panel.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → AI & Models**.


### Engine & routing

- **AI Engine**
- **External Provider**

### Local models

- **Use selected**
- **Refresh**
- **Delete**
- **Model storage**
- **On this device**
- **Active model**

### Startup

- **Load last used model on startup**

### Generation

- **Temperature**
- **Context limit**
- **Limit maximum reply length**
- **Max reply tokens**
- **Chat history**
- **Show advanced generation settings**
- **Top-K sampling**
- **Top-P sampling**
- **Min-P sampling**
- **Repeat penalty**
- **Presence penalty**

### Chat style

- **Encourage brief follow-ups on general chat**

### Reasoning skills

- **Enable compositional reasoning skills**
- **Show advanced hardware settings**
- **GPU offload layers**
- **CPU thread pool**
- **Show advanced chat template settings**
- **Reset**
- **Chat template (internal)**

### Auxiliary cognition

- **Download base cognition model**
- **Show advanced engine settings**
- **Use selected**
- **Reset to default**
- **Delete**

- **Reset to default configuration** — restores all settings on this page

## Related

- [Hardware tuning FAQ](../../faq/hardware-tuning-internal-engine.md) — GPU offload layers, CPU threads, VRAM
- [Generation parameters FAQ](../../faq/generation-parameters.md) — temperature, context window, max reply tokens, chat history
- [Model Manager feature](../../features/model-manager.md) — browse and download GGUF models
- [Set up local models workflow](../../workflows/set-up-local-models.md) — end-to-end local model setup
- [Internal engine vs external server FAQ](../../faq/internal-engine-vs-external-server.md) — when to use each
- [Model won't load troubleshooting](../../troubleshooting/model-wont-load.md) — load failures and VRAM issues
