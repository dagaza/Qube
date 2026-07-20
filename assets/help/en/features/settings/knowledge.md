# Knowledge

## Common questions

- How do I search my Library from chat?
- What are Live Sources?
- Where is search quality mode (Fast / Balanced / Power)?
- How do I create a knowledge preset?
- How does web search discovery work?

## What it is

**Knowledge** settings connect chat to your information across several subsections:

- **Library search phrases** — master **Local Knowledge Base** switch, **NLP Auto-Activator**, and custom trigger phrases
- **Search quality** — **Fast**, **Balanced**, or **Power** embedding/rerank presets (download models first)
- **Retrieval profile** — how aggressively Qube retrieves and ranks library chunks
- **Web search discovery** — privacy tier, DuckDuckGo pacing/limits, optional SearXNG
- **Live sources** — adapter toggles for structured online catalogs
- **Custom sources** and **My knowledge** — REST connectors and bundled presets (`@[tool:…]` workflows)
- **Diagnostics** — retrieval trace, knowledge pack import/export
- **Advanced embedding** — optional custom embedder override

**My knowledge** presets bundle **API adapter ids** or **web-fetch domains**, not Library folders. Attach `@[tool:library]` or enable the master RAG switch for document search.

## Where to find it

Open **Settings → Knowledge** (settings section `knowledge`). Press **?** for the guided tour (`settings.knowledge`). See also the generated [Live sources overview](../../reference/live-sources-overview.md).

## Also called

knowledge base settings, RAG settings, library search, document search, NLP RAG TRIGGERS, embeddings, internet search adapters, live sources

## How to…

1. **Prepare search models** — On **Search quality**, use **Prepare search models** or **Download all search presets** before expecting Library hits (see workflow below).
2. **Enable library search** — Turn on **Enable Local Knowledge Base** under **Library search phrases**. Add custom phrases and/or enable **Enable NLP Auto-Activator** for one-turn searches even when the master switch is off.
3. **Set search quality** — Pick **Mode**: **Fast**, **Balanced**, or **Power** to match latency vs depth.
4. **Configure web discovery** — Under **Web search discovery**, choose a **Privacy tier** and review **Live DDG usage** before relying on `@internet` or Hybrid Internet Mode.
5. **Enable Live Sources** — Toggle the adapters you need; use **Configure** where API keys are required.
6. **Create a preset** — In **My knowledge**, choose **API adapters (scientific, finance, legal)** or **Web fetch (source profile)**, then **Save preset** for repeatable `@[tool:…]` bundles.
7. **Chat with documents** — Attach `@[tool:library]` in **Conversations**, enable **Local Knowledge Base** in the tools panel, and/or rely on custom trigger phrases. Routing behaviour is explained in [Cognitive Router — how routing works](../../faq/cognitive-router-how-routing-works.md).

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

### My knowledge

- **Save preset**
- **Delete selected**
- **Explain selected**

### Custom sources

- **Source id**
- **Label**
- **Connector**
- **Base URL**
- **Search path**
- **Save source**
- **Test**
- **Delete selected**

### Diagnostics

- **Refresh last retrieval trace**
- **Export knowledge pack**
- **Import knowledge pack**

### Source status


### Web search discovery

- **Privacy tier**
- **Slow down live DuckDuckGo searches slightly (recommended)**
- **Live DDG usage**
- **Show advanced discovery limits**
- **Session limit override**
- **SearXNG base URL**
- **Reset discovery health**

- **Reset to default configuration** — restores all settings on this page

## Related

- [Library feature](../../features/library.md) — document storage and ingest
- [Conversations feature](../../features/conversations.md) — chat and composer attachments
- [Prepare search models workflow](../../workflows/prepare-search-models-for-library.md) — embeddings and rerankers
- [Create knowledge preset workflow](../../workflows/create-knowledge-preset.md) — bundle sources
- [Live sources vs Library search FAQ](../../faq/live-sources-vs-library-search.md) — two retrieval paths
- [Cognitive Router — how routing works](../../faq/cognitive-router-how-routing-works.md) — triggers, vetoes, and route vocabulary
- [Library search returns nothing troubleshooting](../../troubleshooting/library-search-returns-nothing.md) — empty results
