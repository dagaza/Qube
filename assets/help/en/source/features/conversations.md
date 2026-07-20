# Conversations

## Common questions

- Where is the chat screen?
- How do I start a new conversation?
- How do I find an old chat?
- How do I attach files or `@` tools in chat?
- What do the **● RAG**, **● WEB**, and **● HYBRID** indicators mean?
- How is **Web** different from **Hybrid Internet Mode**?
- How do I export or copy a conversation?
- How do I open sources or citations in a reply?
- Can I regenerate or edit a message?

## What it is

**Conversations** is Qube's main chat view. Each thread keeps its own message history, composer, and reading area. Use it to ask questions, attach `@[tool:…]` tokens, and read grounded answers with citations.

The **left sidebar** lists folders and chats (including the built-in **Main** folder). The **main stage** shows the transcript, a readability toolbar, and the composer. The **right-hand tools panel** (global, not chat-specific) holds model, voice, generation, and retrieval toggles you can adjust while chatting. **Per-message sources** and citation previews live on assistant replies—not inside the tools panel.

Press **?** in the sidebar header for a spatial guided tour (`conversations`). This page summarizes controls in retrieval-friendly prose; the tour walks the layout button-by-button.

## Where to find it

Click **Conversations** in the left navigation (chat icon). Press **?** in the sidebar header for the guided tour.

## Also called

chat, chat screen, main chat, threads, conversation list, composer, new chat, message input

## How to…

1. **Start a new chat** — Click **New conversation** (+ icon) in the sidebar header. The **Web** toggle resets to off for each new chat. Titles update automatically after your first messages.
2. **Find a chat** — Type in **Search titles or messages…** to search conversation titles and message text, or browse folders when search is empty.
3. **Organize threads** — Create folders with **New folder**. Use the sort icon (**Sort folders and items**) for **By Name** or **By Date**. Open a folder's **⋮** menu to **Rename Folder**, **Export** (ZIP of all chats in that folder), or **Delete Folder** (not available on built-in **Main**).
4. **Manage one chat** — On a conversation row, open **⋮** (**Conversation actions**) for **Rename Chat**, **Move to folder**, **Export** (Markdown file), or **Delete Chat**.
5. **Attach context** — Click **Attach (@)** or type `@` in the input. Pick files, tools, skills, commands, or another conversation. Only **one routing source** (library/web/memory-style attachment) is allowed per message; extra attachments show a toast. Remove chips with **×** on the strip above the input.
6. **Search the web** — See [Web vs Hybrid Internet Mode](#web-vs-hybrid-internet-mode) below and [Cognitive Router — how routing works](../faq/cognitive-router-how-routing-works.md).
7. **Ground on your Library** — Attach `@[tool:library]`, enable **Local Knowledge Base** in the tools panel, and/or rely on **NLP Auto-Activator** or Knowledge settings triggers (see [Knowledge settings](settings/knowledge.md)).
8. **Dictate one message** — Click **Speak your message (push-to-talk)** beside the input (this is push-to-talk, not the wakeword). Always-on listening is controlled under **Enable Voice Input** in the tools panel.
9. **Export the thread** — Use **Download conversation as Markdown** or **Copy conversation to clipboard** in the reading toolbar, or **Export** from a chat's **⋮** menu.
10. **Inspect citations** — On an assistant reply, click **Sources** (when citations exist) or click numbered citation links in the text.

**Not supported in v1:** drag-and-drop reorder of chats, editing or regenerating individual messages after send.

## Controls

Grouped top-to-bottom like the Conversations layout. Readability choices apply to the **current session only** (they are not saved to Settings).

### Sidebar (conversation list)

| Control | What it does |
|---------|----------------|
| **?** (guided tour) | Starts the Conversations tour |
| **New conversation** | New thread in the active folder |
| **New folder** | Create a folder for grouping chats |
| Sort icon | **By Name** or **By Date** for folders and chats |
| **Search titles or messages…** | Filter by title or message body |
| Folder row | Click to set active folder; chevron expands/collapses |
| Folder **⋮** | **Rename Folder**, **Export** (ZIP), **Delete Folder** (user folders only) |
| Chat row | Click to load the thread |
| Chat **⋮** | **Rename Chat**, **Move to folder**, **Export**, **Delete Chat** |
| **Main** folder | Built-in; cannot rename or delete |

### Reading toolbar (above transcript)

| Control | What it does |
|---------|----------------|
| **A−** / **A+** | Decrease / increase chat font (Shift+click: larger step) |
| Line spacing icon | Cycles **Compact**, **Comfortable**, **Relaxed** line spacing |
| Text alignment icon | Toggles **Left** and **Justified** alignment |
| Reader focus | Dims other messages; hover a message to focus it |
| High contrast | Stronger markdown contrast in the transcript |
| Layout width icon | **Narrow column** (~800px) vs **Wide column** (~1200px) |
| Download icon | **Download conversation as Markdown** |
| Copy icon | **Copy conversation to clipboard** |

Export buttons stay disabled until the thread has at least one message.

### Composer

| Control | What it does |
|---------|----------------|
| **Web** | When on, web search applies to **every following message** in this chat until turned off |
| **Think** | Shows model reasoning in replies (**Internal Engine** only, when the loaded model supports it) |
| Context chips | Attached `@` items above the input; **×** removes one |
| **Attach (@)** | Opens the `@` palette (Files, Conversations, Tools, Skills, Commands) |
| Push-to-talk mic | **Speak your message (push-to-talk)** — one-shot dictation |
| Message input | **Enter** sends; **Shift+Enter** new line; `@` opens inline attach palette |
| Send / Stop | **Send message** while idle; becomes **Stop** during generation, voice capture, TTS wait, or deep research |

First use of `@` may show a one-time discovery hint. Tool and command lists are in [Composer tools reference](../reference/composer-tools.md).

### Tools panel (right column, global)

Collapse with the chevron (**Hide tools panel** / **Show tools panel**). These controls affect chat behavior app-wide; detailed voice device setup lives in [Voice & Audio settings](settings/voice-audio.md).

**LOCAL LLM**

| Control | What it does |
|---------|----------------|
| Model selector | Choose / load a `.gguf` model |
| Eject button | **Eject loaded model (free VRAM)** |
| **Load model on startup** | Auto-load the last used local model |

**Audio & TTS Voice**

| Control | What it does |
|---------|----------------|
| **Enable Voice Input** | Master switch for microphone / wakeword pipeline |
| **Silence Cutoff** | Pause after speech ends before capture closes |
| **Noise Suppression** | Background noise gate |
| **Trigger Threshold** | Wakeword sensitivity |
| **Enable TTS Voice** | Speak assistant replies aloud |
| Voice selector | **Choose text-to-speech voice** |

**GENERATION PARAMETERS**

Synced with **Settings → AI & Models → Generation** (two-way). **Max reply tokens** is Settings-only. See [Generation parameters FAQ](../faq/generation-parameters.md) for how Qube budgets tokens.

| Control | What it does |
|---------|----------------|
| **Temperature** | Baseline reply randomness (default 0.8). Qube may lower it slightly on risky turns. |
| **Context Limit** | Total token window for prompt + reply. Internal Engine reloads the model when this changes. |
| **Chat History** | Recent session messages included each turn (default 10). Not the same as Memory Manager. |

**RAG ENGINE**

| Control | What it does |
|---------|----------------|
| **Local Knowledge Base** | Master switch for Library retrieval in chat |
| **NLP Auto-Activator** | Prompt-based Library wake even when master RAG is off |
| **Strict Isolation Mode** | Answers must cite retrieved Library chunks |

**MCP TOOLS**

| Control | What it does |
|---------|----------------|
| **Hybrid Internet Mode** | Lets Qube auto-route to web search when context warrants it (pairs with **● HYBRID**) |

### Top bar (status while on Conversations)

| Indicator | Meaning |
|-----------|---------|
| Status bubble | **Idle**, **Listening**, **Working**, **Speaking**, and similar states |
| **● RAG** | Library retrieval: grey = off, blue = ready, green = retrieving |
| **● WEB** | Web search state for the current chat / turn |
| **● HYBRID** | Hybrid Internet Mode: grey = off, coloured when enabled or actively searching |
| Mic level + chevron | Input level; chevron opens **Select microphone input** |
| **⏸ DDG** | Shown only during DuckDuckGo backoff (rare) |

Memory retrieval uses `@[tool:memory]` or related routing—there is **no separate MEMORY status dot**.

### Web vs Hybrid Internet Mode

All three controls route **live web search** — they are **not** the same as a **HYBRID** execution route in **Telemetry** (Memory + Library together). See [Cognitive Router FAQ](../faq/cognitive-router-how-routing-works.md#web-vs-hybrid-internet-mode).

| Control | Scope | When to use |
|---------|-------|-------------|
| **Web** toggle (composer) | Sticky for **this chat** | You want **every following message** to use web search until you turn it off |
| **`@[tool:internet]`** (or related `@` tools) | **This message only** | One explicit web or discovery turn |
| **Hybrid Internet Mode** (tools panel) | App setting; **● HYBRID** dot | Let Qube **decide per turn** when live-web intent warrants search — not forced on every message |

**● WEB** reflects web search state. **● HYBRID** reflects **Hybrid Internet Mode** (auto web routing), **not** “Memory + Library HYBRID route” on the Telemetry dashboard.

### Assistant messages

| Control | What it does |
|---------|----------------|
| Citation links in text | Open in-app source preview (or external browser for `http(s)`) |
| **Sources** button | Citation list dialog; may include a research map for evidence routes |
| Copy menu | **Copy as plain text** or **Copy as Markdown** |
| Help action chips | From `@help` answers—jump to a Settings section |
| **STT**, **TTFT**, **TTS**, **TPS** | Per-turn latency / throughput labels when available |

User messages: right-click **Copy**. Routing/skill chips on sent bubbles are read-only.

## Related

- [Chat with a library document workflow](../workflows/chat-with-a-library-document.md) — RAG from uploaded files
- [What do `@` mentions do FAQ](../faq/what-do-at-mentions-do.md) — composer tokens
- [Generation parameters FAQ](../faq/generation-parameters.md) — temperature, context, reply caps, chat history
- [Conversations vs memory context FAQ](../faq/conversations-vs-memory-context.md) — chat history vs long-term memory
- [Chat history vs export FAQ](../faq/chat-history-vs-export.md) — export formats and scope
- [Composer tools reference](../reference/composer-tools.md) — full `@` tool list
- [Knowledge settings](settings/knowledge.md) — search quality and automatic Library triggers
- [Cognitive Router — how routing works](../faq/cognitive-router-how-routing-works.md) — pathways, overrides, why citations may be missing
- [Voice & Audio settings](settings/voice-audio.md) — devices, wakeword, TTS defaults
- [AI & Models settings](settings/ai-models.md) — engine mode, GPU layers, thinking models
- [Library](../library.md) — where documents live before chat retrieval
