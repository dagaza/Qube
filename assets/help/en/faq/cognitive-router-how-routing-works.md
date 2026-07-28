# Cognitive Router — how Qube routes a chat turn

## Common questions

- How does Qube decide between Memory, Library, web, and plain chat?
- What is the **cognitive router**?
- What is the difference between **● HYBRID** and a **HYBRID** route in Telemetry?
- How is the composer **Web** toggle different from **Hybrid Internet Mode**?
- Why did Qube answer without citations even though it searched?
- Where can I see why a specific turn routed the way it did?

## What the cognitive router is

Before the main chat model replies, Qube picks a **knowledge pathway** — what context to retrieve (if any). The **cognitive router** (`CognitiveRouterV4`) scores your message against Memory, Library (RAG), web, and self-knowledge lanes using phrase triggers and embedding centroids.

**It is not the only decider.** Pre-router rules (composer attachments, “remember this”, file-search phrasing) and post-router rules (recall fusion, custom Library triggers, web/RAG vetoes) can override or adjust the router’s pick. **Skills** and the **sidecar** assist with prompts and query rewrite but **do not choose routes**.

Everyday chat uses this automatically — you do not open a “router settings” page.

## Route vocabulary (execution routes)

These names appear in **Telemetry → Router Intelligence**, **Routing debug log** JSONL, and developer traces. The UI shows **● RAG / ● WEB / ● HYBRID** dots for retrieval *activity*, not the full route name.

| Route | Plain meaning | What usually runs |
|-------|---------------|-------------------|
| **NONE** | **Self-knowledge** — answer from the model’s weights (plus optional light preference injection). | No Memory/Library/web retrieval tools required |
| **MEMORY** | **Long-term Memory** — saved facts from past chats (`Memory Manager`). | Memory search |
| **RAG** | **Library** — your imported documents. | Library / RAG search |
| **WEB** | **Live web** — DuckDuckGo SERP snippets (not full page fetches). | Internet search tool |
| **HYBRID** | **Memory + Library together** — both retrievers run; the model synthesizes across hits. | Memory search **and** Library search |

`INTERNET` is treated as an alias of **WEB** in execution paths.

> **Important — two different “HYBRID” words**
>
> | You see… | It means… |
> |----------|-----------|
> | **● HYBRID** dot + **Hybrid Internet Mode** toggle (Conversations tools panel) | Qube may **auto-route to web search** when context warrants it — **not** Memory+Library fusion. |
> | **HYBRID** in **Telemetry → Router Intelligence → Routes** | A turn that ran **Memory and Library RAG** together. |
>
> Same label, different mechanisms. See [Web vs Hybrid Internet Mode](#web-vs-hybrid-internet-mode) below.

## End-to-end flow (simplified)

1. **Discourse grounding** — follow-ups may expand the *internal* routing query (“Regarding Paris: what about hotels?”) without changing what you typed.
2. **Pre-router overrides** — beat the cognitive router when they fire (attachments, explicit remember, file-only intent, narrative recap).
3. **Cognitive router** — if nothing above applied, scores lanes and picks an initial route.
4. **Post-router overrides** — recall fusion, custom Library trigger phrases, Hybrid Internet / manual web, discourse downgrades, web/RAG capability vetoes.
5. **Tool execution** — Memory, Library, and/or web search run for the chosen route.
6. **Relevance gates** — low-quality hits are dropped per channel.
7. **Empty-source downgrade** — if a retrieval route finished with **zero** surviving sources, Qube **downgrades to NONE** before prompt build so the model is not forced into “cite your sources” mode with nothing to cite. Telemetry may still record the **original** route.
8. **Prompt + LLM** — system suffixes match the **final** route and whether sources exist.

## Pre-router overrides (user-visible levers)

These run **before** the cognitive router and win when they match:

| Trigger | Typical route | Where it comes from |
|---------|---------------|---------------------|
| Explicit “remember this…” | **NONE** (write turn, no retrieval) | Natural language |
| File / document scoped question | **RAG** | Phrasing + **`@[file:…]`** |
| Session narrative recap | **MEMORY** (+ episodes) | Recap-style phrasing |
| Composer routing attachment | varies | **First** `@[file:…]`, `@[chat:…]`, or `@[tool:…]` in the message controls routing |

See [Composer attachments](../reference/composer-attachments.md) and [What do `@` mentions do](what-do-at-mentions-do.md).

## Post-router overrides (why the route changed after scoring)

| Override | When | Effect |
|----------|------|--------|
| **Recall fusion** | “Tell me about X”, “who is Y”, similar recall phrasing | **NONE**, **MEMORY**, or **RAG** → **HYBRID** (Memory **and** Library) |
| **Discourse downgrade** | High-confidence follow-up with an active topic in thread | **MEMORY** / **RAG** / **HYBRID** → **NONE** (answer from chat context) |
| **Custom Library triggers** | Your phrases under **Settings → Knowledge** | Upgrade toward **RAG** (or keep **HYBRID**) |
| **Manual / auto web** | **Web** toggle, `@[tool:internet]`, live-web phrasing, or **Hybrid Internet Mode** + router web intent | Force **WEB** |
| **Proactive web veto** | Router picked **WEB** but internet tool is off and no explicit web trigger | **WEB** → **NONE** |
| **RAG capability veto** | Library route but **Local Knowledge Base** master switch off (no bypass) | **RAG** → **NONE**; **HYBRID** → **MEMORY** only |
| **Discourse web veto** | Ungrounded “search for this” with no topic | **WEB** → **NONE** |

**Bypasses for Library when master RAG is off:** NLP Auto-Activator phrases, custom trigger lines, explicit file-search intent, **`@[file:…]`** attachments.

## Web vs Hybrid Internet Mode

All three are about **web**, not Memory+Library **HYBRID** routes.

| Control | Scope | Behaviour |
|---------|-------|-----------|
| **Web** toggle (above composer) | **This chat**, sticky | **Every following message** uses web search until you turn it off |
| **`@[tool:internet]`** (or related `@` tools) | **This message** | Explicit single-turn web / discovery routing |
| **Hybrid Internet Mode** (tools panel **MCP TOOLS**) | **App-wide** setting | Qube’s cognitive router may **auto-pick WEB** per turn when live-web intent is detected — you do not force web on every message |

**● WEB** dot — web search state for the chat/turn. **● HYBRID** dot — **Hybrid Internet Mode** enabled or actively searching the web on this turn (not “Memory + Library HYBRID route”).

Pair **Hybrid Internet Mode** with sensible **Settings → Knowledge → Web search discovery** privacy tier and DDG pacing before relying on automatic web. See [Web discovery privacy tiers](web-discovery-privacy-tiers.md).

## Settings and tools that influence routing

| Lever | Location | Routing effect |
|-------|----------|----------------|
| **Local Knowledge Base** | Tools panel + **Settings → Knowledge** | Master Library retrieval switch; off → RAG veto unless bypass |
| **NLP Auto-Activator** | Tools panel + Knowledge | One-turn Library wake even when master switch off |
| **Custom trigger phrases** | **Settings → Knowledge** | Phrase → Library route upgrade |
| **Strict Isolation Mode** | Tools panel **RAG ENGINE** | Does **not** pick routes — when Library sources exist, prompt requires citing retrieved chunks |
| **`@[tool:memory]`** / **`@[tool:library]`** | Composer | Explicit routing targets (first routing attachment wins) |
| **Hybrid Internet Mode** | Tools panel **MCP TOOLS** | Enables auto **WEB** when router detects live-web intent |

**Adaptive tuner weights** (`h` / `m` / `r` on **Telemetry → Router Intelligence**) are **automatic and read-only** — Qube adjusts sensitivities from recent turn outcomes; there is no Settings slider for them.

## Why no citations on a “search” turn

Common causes after routing:

1. **Relevance gates** dropped all Memory/Library/web hits (semantic floor, overlap gate, etc.).
2. **Empty-source downgrade** — retrieval ran on **MEMORY**, **RAG**, **HYBRID**, or **WEB**, but nothing survived → final answer is plain **NONE**-style chat without citation discipline.
3. **RAG capability veto** — Library was expected but master **Local Knowledge Base** was off.
4. **Web veto or empty SERP** — internet disabled, ungrounded follow-up, or zero web hits after gating.
5. **Discourse downgrade** — follow-up answered from thread text without re-retrieval.

Check **Sources** on the reply; if empty, the final route was treated as non-grounded even when telemetry shows an earlier lane.

## Where to inspect routing (three layers)

| Layer | Where | Best for |
|-------|-------|----------|
| **Session aggregate** | **Telemetry → Router Intelligence** | Route mix, avg retrieval phase, tuner weights, health flags over ~200 recent turns |
| **Per-reply retrieval** | **Sources → INSPECT RETRIEVAL** (when shown) | Adapters, preset, pipeline detail for **that assistant message** |
| **Per-turn router JSONL** | **Settings → Advanced → Routing debug log** | Full route, strategy, intent scores, policy trace — enable recording, send one message, **View** log |

See [Advanced Telemetry — interpreting](advanced-telemetry-interpreting.md) and [Diagnostic logs — Advanced](diagnostic-logs-advanced-settings.md).

## Related

- [Conversations](../features/conversations.md) — **Web** toggle, status dots, tools panel
- [Knowledge settings](../features/settings/knowledge.md) — Library triggers, web discovery, retrieval trace panel
- [Web discovery privacy tiers](web-discovery-privacy-tiers.md) — SERP provider tiers for web turns
- [Memory vs Library](memory-vs-library.md) — two stores, two pipelines
- [What do `@` mentions do](what-do-at-mentions-do.md) — routing attachments vs skills
- [INSPECT RETRIEVAL](inspect-retrieval.md) — per-reply inspector tabs and routing block
- [Advanced Telemetry](../features/telemetry.md) · [Interpreting telemetry](advanced-telemetry-interpreting.md)
- [Composer attachments](../reference/composer-attachments.md) — `@[file:…]`, `@[chat:…]`, order rules
