# Generation parameters (temperature, context, replies)

## Common questions

- What does **Temperature** do in Qube?
- What is **Context limit** and why does changing it reload my model?
- What is the difference between **Context limit** and **Max reply tokens**?
- How does **Chat history** work — and how is it different from Memory?
- Why was my answer cut off mid-sentence?

## What it is

**Generation parameters** control how Qube samples text from the **main chat model** for assistant replies in **Conversations**. They apply to the Internal Engine (local `.gguf`) and External Server paths. Auxiliary cognition models (sidecar tasks) use separate fixed budgets — not these controls.

Each chat turn builds one prompt and asks the model for a completion. Four settings dominate everyday use:

| Setting | Default | What it controls in Qube |
|---------|---------|---------------------------|
| **Temperature** | 0.8 | Randomness/creativity of sampling (0.0–2.0) |
| **Context limit** | 32,000 tokens | Total token window for prompt **and** reply |
| **Limit maximum reply length** + **Max reply tokens** | On / 4,096 | Cap on new tokens per assistant message |
| **Chat history** | 10 messages | How many recent session messages are included |

## Where to find them

- **Settings → AI & Models → Generation** — all controls, including **Limit maximum reply length**, **Max reply tokens**, and **Show advanced generation settings** (Top-K, Top-P, Min-P, repeat/presence penalties).
- **Conversations → tools panel → GENERATION PARAMETERS** — **Temperature**, **Context Limit**, and **Chat History** only. These spinboxes stay **in sync** with Settings (changing either side updates the other).

## How one turn uses the context window

Qube does not reserve a fixed slice per ingredient. The **context limit** is one shared pool counted in **tokens** (roughly word pieces, not characters). Everything below competes for space **before** the model writes the reply:

1. **System and routing text** — instructions, skill guidance, safety/format hints.
2. **Retrieval** — Library chunks (`@[tool:library]`, RAG), web/Live Source excerpts, Memory hits, referenced conversations, help corpus (`@[tool:help]`), etc.
3. **Chat history** — the last *N* user/assistant messages from this session (see below).
4. **Your current message**.
5. **The assistant reply** — whatever tokens remain after the prompt is assembled.

On the **Internal Engine**, **Context limit** sets the model’s `n_ctx`. Increasing it uses more RAM/VRAM. Changing it **reloads the loaded `.gguf` model** so the new window takes effect.

On **External Server**, Qube **does not send context limit to the host** — LM Studio/Ollama use their own window. Qube still uses your Context limit value **client-side** to compute **`max_tokens`** (reply budget). If the host window is smaller than your assembled prompt, the server may truncate or error even when Qube’s slider is high. See [Internal engine vs external server](internal-engine-vs-external-server.md).

The live hint under **Max reply tokens** in Settings summarizes effective reply room for your current numbers (via Qube’s output-token budget logic).

## Temperature

**Temperature** is Qube’s baseline sampling randomness for chat completions:

- **Lower (≈0.1–0.3)** — stricter, more deterministic wording; can sound flat.
- **≈0.7–0.8 (default 0.8)** — balanced natural replies for most chats.
- **Higher (≈0.9–1.0+)** — more varied/creative wording; less predictable.

Qube may **scale temperature down slightly on risky turns** (for example long threads, structured list requests, or after an unreliable prior reply) using an internal **generation risk profile**. Your Settings value is still the baseline — risk adjustment is temporary per turn, not a second hidden setting.

Temperature does **not** change routing, retrieval, or Memory. It only affects how the model chooses the next token during the reply.

## Context limit vs max reply tokens

These are related but not the same knob:

- **Context limit** — ceiling for **prompt + reply together** in one inference call.
- **Limit maximum reply length** — when **on** (default), each reply stops after at most **Max reply tokens** (default 4,096), even if the window has spare room.
- When **Limit maximum reply length** is **off**, the reply may grow until the context window is full minus whatever the prompt already consumed.

Large prompts shorten replies: long **Chat history**, big Library retrievals, or pasted text leave fewer tokens for the answer. If replies truncate unexpectedly, try lowering **Chat history**, raising **Context limit** (if hardware allows), turning **off** the reply cap, or shortening the question.

On the native engine, Qube **re-counts prompt tokens** after assembly and may clamp `max_tokens` again so the completion fits `n_ctx`.

## Chat history (session window)

**Chat history** is how many **recent messages** from the **current conversation** Qube includes in each prompt (default **10**, range **2–100**). Each user or assistant line counts as one message.

Qube **windows** long threads: older messages drop out of the prompt even though they remain visible in the transcript. Very long single messages are truncated with a safety cap before windowing.

This is **not** long-term **Memory**:

- **Chat history** — short-term continuity inside one conversation session.
- **Memory Manager** — durable facts Qube may inject across sessions when relevant (see [Conversations vs memory context](conversations-vs-memory-context.md)).

When history is windowed and Memory salvage is enabled, Qube may enqueue extraction from dropped messages so important facts can still reach Memory — but the model no longer sees those turns directly.

## Advanced generation settings

Under **Show advanced generation settings** in **Settings → AI & Models**:

| Control | Role in Qube |
|---------|----------------|
| **Top-K sampling** | Consider only the K most likely next tokens (0 = off) |
| **Top-P (nucleus)** | Nucleus sampling cutoff (default 0.95) |
| **Min-P** | Drop tokens below a relative probability floor (default 0.05) |
| **Repeat penalty** | Discourage repeating recent words (default 1.1; may be nudged by risk profile) |
| **Presence penalty** | Discourage reusing tokens already in the output (default 0.0) |

Defaults work for most models. Change these when tuning repetition or diversity on a specific `.gguf`.

## Truncated or cut-off replies

If a reply stops abruptly:

1. Check whether **Limit maximum reply length** is on with a low **Max reply tokens**.
2. Check whether **Context limit** is tight relative to retrieval + **Chat history** size.
3. Qube may emit a turn notice when output likely hit a token ceiling (`max_tokens` / length finish).

Use **Advanced Telemetry** or generation debug logs if you need per-turn token diagnostics.

## Also called

temperature setting, context window, max tokens, reply length cap, chat history messages, sampling parameters, generation settings, inference settings

## Related

- [AI & Models settings](../features/settings/ai-models.md) — full Generation section and hardware tuning
- [Conversations](../features/conversations.md) — tools panel mirrors Temperature, Context Limit, Chat history
- [Conversations vs memory context](conversations-vs-memory-context.md) — session history vs Memory Manager
- [Internal engine vs external server](internal-engine-vs-external-server.md) — context limit on External vs Internal
