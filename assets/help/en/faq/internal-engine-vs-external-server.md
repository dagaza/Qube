# Internal engine vs external server

## Common questions

- Should I use Internal or External inference?
- What is the difference between the native engine and LM Studio?
- Does External send my data off the machine?
- Why doesn’t Qube’s **Context limit** match my LM Studio window?
- Do **GPU offload layers** apply on External Server?

## What it is

**Internal Engine (native)** runs `.gguf` models locally through Qube’s in-process **llama.cpp** engine — data stays on your hardware subject to model and OS constraints. You tune **GPU offload layers**, **CPU thread pool**, **Context limit** (`n_ctx`), and **Chat template (internal)** in Qube.

**External Server (localhost)** sends chat completions to an **OpenAI-compatible HTTP API** on your machine — typically **LM Studio (port 1234)** or **Ollama (port 11434)**. The external host loads its own model, chooses its own context window, and manages its own GPU/CPU allocation.

Choose **Internal** for offline-first privacy and integrated hardware tuning. Choose **External** when you already host models in another tool, need a model Qube cannot load natively, or want to share one server across apps.

## Where to find it

Switch engines in **Settings → AI & Models → Engine & routing** with **AI Engine**. Pick **External Provider** when External is selected. Internal model files live under **Local models** / **Model Manager**; hardware tuning is under **Show advanced hardware settings**.

## What Qube controls on each engine

| Concern | Internal Engine | External Server |
|---------|-----------------|-----------------|
| Model file | Qube `.gguf` in Model Manager | Model loaded **inside LM Studio / Ollama** |
| **GPU offload layers** / **CPU threads** | **Settings → AI & Models → Hardware tuning** | Configure in the **external host** — Qube ignores these |
| **Context limit** | Sets llama.cpp **`n_ctx`**; **reloads `.gguf`** when changed | **Does not configure the server’s window** (see below) |
| **Chat template (internal)** | Qube applies template when building messages | Host applies its own template |
| **Temperature**, **Max reply tokens**, sampling | Qube sends on every chat request | Qube sends on every chat request |
| **Chat history** window | Qube trims messages before the request | Qube trims messages before the request |
| VRAM while chatting | Qube holds the loaded `.gguf` | Qube **unloads** native model when you switch to External |

## External Server: context limit quirks

This is a common source of confusion:

1. **Qube does not send `n_ctx` (context limit) to the external API.** The HTTP payload contains `messages`, `temperature`, `max_tokens`, `stream`, and sampling fields — not a context-window size.

2. **The server’s real context window** is whatever you configured in **LM Studio**, **Ollama**, or another host (often 4k–128k depending on model and host settings). Qube cannot raise it from Settings.

3. **Qube still uses your Context limit setting on External** — but only on the **client side** to compute **`max_tokens`** (reply budget) via its output-token budget logic. If Qube assumes a 32k window while the server only allows 8k, the **server** may truncate the prompt, return an error, or stop early even though Qube’s UI suggested more reply room.

4. **Symptoms when host window is smaller than your prompt:**
   - Missing early chat turns (server-side truncation)
   - `finish_reason: length` on short answers
   - HTTP errors from the host on very large RAG prompts

   **Fix:** Raise context in **LM Studio/Ollama**, or lower Qube **Chat history**, retrieval scope, or **Context limit** so the assembled prompt fits the **host** window — not just Qube’s slider.

5. **Changing Context limit on External** updates Qube’s stored setting and reply-budget math but **does not reload** anything (unlike Internal, where it reloads the `.gguf`).

See [Generation parameters FAQ](generation-parameters.md) for how prompt pieces share the window.

## External Server: other Qube-specific behaviour

- **One brain at a time** — switching **AI Engine** to External **unloads** Qube’s native `.gguf` to free VRAM (`Engine: External — native model unloaded`).
- **Local llama.cpp servers** — when Qube detects a localhost OpenAI-compatible service, it sets `cache_prompt: false` on requests to reduce unbounded prompt-prefix / KV reuse across unrelated chat turns.
- **Model reload** — the Conversations **reload** action on External emits **Model Context Updated** status only; it does **not** reload the remote host’s weights (configure that in LM Studio/Ollama).
- **Privacy** — External to `localhost` keeps traffic on your machine; pointing a provider at another PC on the network sends prompts to that machine instead.

## Internal Engine: quick tuning pointers

- Start **GPU offload layers** conservative; raise until loads fail, then back off (see [Hardware tuning FAQ](hardware-tuning-internal-engine.md)).
- Match **Chat template (internal)** to the model family when replies look malformed.
- Use **Eject loaded model (free VRAM)** before heavy External sessions on the same GPU.

## Also called

local vs external LLM, native engine vs Ollama, on-device vs server inference, GGUF vs API server, LM Studio localhost

## How to…

1. Open **Settings → AI & Models**.
2. Select **Internal Engine (native)** to run local `.gguf` models. Download or load files in **Model Manager**; tune **GPU offload layers** under **Show advanced hardware settings**.
3. Select **External Server (localhost)** and pick **External Provider** (**LM Studio** or **Ollama**).
4. Confirm the external server is running and its model context window fits your Qube prompts before long RAG chats.
5. Switch back to Internal if latency, privacy, or integrated tuning requirements change.

## Related

- [Hardware tuning FAQ](hardware-tuning-internal-engine.md) — GPU layers, CPU threads, VRAM
- [Generation parameters FAQ](generation-parameters.md) — temperature, context, max reply tokens
- [AI & Models settings](../features/settings/ai-models.md) — engine and provider controls
- [Set up local models workflow](../workflows/set-up-local-models.md) — Internal path
- [Model won't load troubleshooting](../troubleshooting/model-wont-load.md) — when Internal fails
