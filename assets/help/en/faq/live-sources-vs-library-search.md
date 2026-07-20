# Live sources vs Library search

## Common questions

- What is the difference between `@library` and `@internet`?
- Do Live Sources search my uploaded files?
- When should I use Live Sources instead of Library?

## What it is

**Library search** retrieves passages from documents **you ingested** in **Library → Main** via **`@[tool:library]`**, **Local Knowledge Base**, or custom **Library search phrases**.

**Live Sources** query **online catalogs**—scientific papers, finance filings, legal opinions—when tools such as **`@[tool:evidence]`**, **`@[tool:finance]`**, or **`@[tool:legal]`** route to adapters configured under **Settings → Knowledge → Live sources**.

**`@[tool:internet]`** uses the **Web search discovery** pipeline (DuckDuckGo/Brave/SearXNG)—not Live Sources adapters.

They complement each other: Library for your private corpus; Live Sources for structured external datasets; **`@internet`** for general web discovery.

## Where to find it

Configure Library search under **Settings → Knowledge → Library search phrases** and ingest files in **Library → Main**. Enable Live Sources adapters on the same page; configure web discovery separately under **Web search discovery**. See [Live sources overview](../reference/live-sources-overview.md) for the adapter catalog.

## Also called

internet search vs document search, online sources vs RAG, evidence adapters vs library, web search vs my files

## How to…

1. Use **`@[tool:library]`** (or **Local Knowledge Base** / custom trigger phrases) for your own PDFs and notes.
2. Use **`@[tool:evidence]`** / **`@finance`** / **`@legal`** / similar tools for Live Sources catalogs.
3. Use **`@[tool:internet]`** for general web search (configured under **Web search discovery**).
4. Combine **`@[tool:library]`** with a Live Source or preset tool in one message when research spans private notes and public datasets.
5. Verify API keys for adapters that require them before expecting hits.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — all paths configured here
- [Live sources overview reference](../reference/live-sources-overview.md) — adapter list
- [Chat with a library document workflow](../workflows/chat-with-a-library-document.md) — Library path
- [Create knowledge preset workflow](../workflows/create-knowledge-preset.md) — custom scoped tools
