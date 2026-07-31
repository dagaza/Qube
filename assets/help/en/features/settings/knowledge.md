# Knowledge

## Common questions

- How do I search my Library from chat?
- What are Live Sources?
- Where is search quality mode (Fast / Balanced / Power)?
- What is retrieval profile?
- What's the difference between retrieval profile and search quality?
- How do I create a knowledge preset?
- How does web search discovery work?
- What is Library Pro depth (precision ingest / precision retrieval)?
- Do I need a Pro license for precision ingest?

## What it is

**Knowledge** settings connect chat to your information across several subsections:

- **Library search phrases** — master **Local Knowledge Base** switch, **NLP Auto-Activator**, and custom trigger phrases
- **Search quality** — **Fast**, **Balanced**, or **Power** embedding/rerank presets for **Library indexing** (download models first)
- **Library Pro depth** — optional **Default precision ingest on import** and **Precision retrieval** toggles (**Qube Pro** license required)
- **Retrieval profile** — **global orchestration** for knowledge turns (adapter fan-out, timeouts, cache, web fetch depth)—not Library-only and not ranking
- **Web search discovery** — privacy tier, DuckDuckGo pacing/limits, optional SearXNG
- **Live sources** — adapter toggles for structured online catalogs
- **Custom sources** and **My knowledge** — REST/GraphQL/MCP connectors and bundled presets (`@[tool:…]` or capability bundles)
- **Diagnostics** — retrieval trace, knowledge pack import/export
- **Advanced embedding** — optional custom embedder override

**My knowledge** presets bundle **API adapter ids** or **web-fetch domains**, not Library folders. Attach `@[tool:library]` or enable the master RAG switch for document search.

## Retrieval profile

**Retrieval profile** is a **global orchestration knob** for knowledge turns—not a Library-only switch and not how Qube **ranks** passages (ranking uses separate profiles and adapters).

It controls **how hard and how fast** Qube searches when any knowledge path runs: **Library** (`@[tool:library]`), **Live Sources**, **My knowledge** presets, **`@[tool:internet]`** / Hybrid Internet Mode, and related pipelines.

| Knob | Examples |
|------|----------|
| Fan-out & budgets | Parallel adapter calls, max results, latency caps |
| Cache | More aggressive SERP caching on **Fast** |
| Web fetch depth | **Fast** = SERP snippets only; **Balanced** / **Thorough** = fetch top result pages |
| Ordering hints | **Local-first** prefers local connectors; **Evidence-first** favors citation quality |

**Profiles:** **Fast**, **Balanced**, **Thorough**, **Evidence-first**, **Local-first**.

**Not the same as Search quality** — **Fast / Balanced / Power** there picks embedding models for Library indexing only. See [Retrieval profile vs search quality](../../faq/retrieval-profile-vs-search-quality.md).

Open **Settings → Knowledge → Retrieval profile**. The Conversations tools panel does not include this control.

## Library Pro depth (Pro license)

**Library Pro depth** adds optional accuracy modes for serious Library collections. Both require a **Qube Pro** (or Team) license imported under **Settings → License**.

| Toggle | Effect |
|--------|--------|
| **Default precision ingest on import** | Pre-selects **Precision indexing** in the Library import dialog (you still choose per upload). Semantic re-segmentation at indexing time. **Re-import documents** to change their mode. |
| **Precision retrieval** | Second bi-encoder rerank after hybrid search + MMR on each Library query. |

Standard structural chunking, hybrid search, MMR, and **Search quality** presets remain **free**. See [Library Pro depth FAQ](../../faq/library-pro-depth.md) and [Enable Library Pro depth workflow](../../workflows/enable-library-pro-depth.md).

## Where to find it

Open **Settings → Knowledge** (settings section `knowledge`). Press **?** for the guided tour (`settings.knowledge`) — including **Library Pro depth** (default precision ingest, precision retrieval). See also the generated [Live sources overview](../../reference/live-sources-overview.md).

## Also called

knowledge base settings, RAG settings, library search, document search, NLP RAG TRIGGERS, embeddings, internet search adapters, live sources, retrieval orchestration, fetch depth settings

## How to…

1. **Prepare search models** — On **Search quality**, use **Prepare search models** or **Download all search presets** before expecting Library hits (see workflow below).
2. **Enable library search** — Turn on **Enable Local Knowledge Base** under **Library search phrases**. Add custom phrases and/or enable **Enable NLP Auto-Activator** for one-turn searches even when the master switch is off.
3. **Set search quality** — Pick **Mode**: **Fast**, **Balanced**, or **Power** to match latency vs depth for **Library embeddings**.
4. **Set retrieval profile** — Pick **Fast**, **Balanced**, **Thorough**, or a hint profile to tune orchestration and web fetch depth for **all knowledge turns** (Library, Live Sources, presets, `@internet`).
5. **Configure web discovery** — **Privacy tier** and **Hybrid Internet Mode** live on **Settings → Privacy & data**; this page holds DDG limits, provider setup, and SearXNG. Choose a tier, review **Live DDG usage**, and use **Open Privacy & data** for audit logs. See [Web discovery privacy tiers FAQ](../../faq/web-discovery-privacy-tiers.md).
6. **Enable Live Sources** — Toggle the adapters you need; use **Configure** where API keys are required.
7. **Create a preset** — In **My knowledge**, choose **API adapters (scientific, finance, legal)** or **Web fetch (source profile)**, then **Save preset** for repeatable `@[tool:…]` bundles. Presets can also bundle **integration capability URNs** saved from MCP grant review.
8. **Connect MCP (optional)** — Under **Custom sources**, add connector **mcp** (command JSON + **namespace**), **Test/Save**, then grant capabilities under **Settings → Integrations**. See [Connect an MCP server](../../workflows/connect-mcp-server.md).
9. **Chat with documents** — Attach `@[tool:library]` in **Conversations**, enable **Local Knowledge Base** in the tools panel, and/or rely on custom trigger phrases. Routing behaviour is explained in [Cognitive Router — how routing works](../../faq/cognitive-router-how-routing-works.md).
10. **Enable Library Pro depth (Pro)** — Import a license under **Settings → License**, then toggle **Default precision ingest on import** and/or **Precision retrieval** under **Library Pro depth**. On each **Import (+)**, choose **Normal** or **Precision indexing**; precision-indexed docs show a **gem** badge.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → Knowledge**.


### Library search phrases

- **Enable Local Knowledge Base**
- **Enable NLP Auto-Activator**

### Search quality

- **Mode**
- **Prepare search models**
- **Download Fast, Balanced, and Power presets for offline mode switching**
- **Download all search presets**

### Library Pro depth

- **Default precision ingest on import**
- **Precision retrieval**

### Retrieval profile

- **Profile**

### Advanced embedding

- **Show advanced embedding settings**
- **Use selected**
- **Refresh**
- **Delete**
- **Model storage**
- **On this device**
- **Custom override**
- **Configure**

### Live sources

- **Recommended setup**
- **Dismiss**
- **Configure**
- **Set up SearXNG…**

### My knowledge

- **Save preset**
- **Delete selected**
- **Explain selected**

### Custom sources

- **Base URL**
- **Search path**
- **Command**
- **Namespace**
- **Tool name**
- **Source id**
- **Label**
- **Connector**
- **New source**
- **Save source**
- **Test**
- **Delete selected**

### Diagnostics

- **Refresh last retrieval trace**
- **Export knowledge pack**
- **Import knowledge pack**

### Source status

- **Open Knowledge → Web search discovery**

### Web search discovery

- **Privacy tier**
- **Slow down live DuckDuckGo searches slightly (recommended)**
- **Live DDG usage**
- **Show advanced discovery limits**
- **Session limit override**
- **SearXNG base URL**
- **Reset discovery health**
- **Open Privacy & data**

- **Reset to default configuration** — restores all settings on this page

## Related

- [Web discovery privacy tiers FAQ](../../faq/web-discovery-privacy-tiers.md) — tier comparison and what leaves your device
- [Retrieval profile vs search quality FAQ](../../faq/retrieval-profile-vs-search-quality.md) — Fast/Balanced naming disambiguation
- [Library feature](../../features/library.md) — document storage and ingest
- [Conversations feature](../../features/conversations.md) — chat and composer attachments
- [Prepare search models workflow](../../workflows/prepare-search-models-for-library.md) — embeddings and rerankers
- [Create knowledge preset workflow](../../workflows/create-knowledge-preset.md) — bundle sources
- [Live sources vs Library search FAQ](../../faq/live-sources-vs-library-search.md) — two retrieval paths
- [Cognitive Router — how routing works](../../faq/cognitive-router-how-routing-works.md) — triggers, vetoes, and route vocabulary
- [Library search returns nothing troubleshooting](../../troubleshooting/library-search-returns-nothing.md) — empty results
- [Library Pro depth FAQ](../../faq/library-pro-depth.md) — precision ingest, precision retrieval, licensing
- [Enable Library Pro depth workflow](../../workflows/enable-library-pro-depth.md) — step-by-step Pro setup
- [Integrations settings](integrations.md) — MCP capability permissions after Custom sources setup
- [Connect an MCP server workflow](../../workflows/connect-mcp-server.md) — filesystem MCP example
