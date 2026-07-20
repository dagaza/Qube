<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
# Composer tools (@tool)

## Common questions

- What `@` tools can I attach in chat?
- What does `@[tool:library]` do?
- What is the difference between `@library` and `@internet`?
- Why do some tools not appear in the `@` palette until I search?

## What these tokens do

Composer **tools** route a chat turn to a specific capability. Insert a token like `@[tool:library]` in your message, or pick one from the `@` palette under **Tools**.

Only the **first routing attachment** in your message controls behaviour — the first token among `@[file:…]`, `@[chat:…]`, or `@[tool:…]` in left-to-right order. See [Composer attachments](composer-attachments.md) for files, chats, and mixing rules.

## Built-in tools

### Trusted — `@[tool:trusted]`

Wikipedia and allowlisted sources

Use for general facts from vetted allowlisted sources.

### Scientific literature — `@[tool:evidence]`

Peer-reviewed papers and preprints across disciplines

Use when you need cited papers; discipline-specific sources are chosen automatically.

### Finance — `@[tool:finance]`

SEC EDGAR company filings (10-K, 10-Q, 8-K)

Use for SEC filings, company financials, and regulatory disclosures.

### Legal — `@[tool:legal]`

U.S. case law opinions via CourtListener

Use for U.S. court opinions and case law.

### Deep research — `@[tool:research]`

Multi-step evidence report (async, non-blocking)

Use for a multi-step async literature review report.

### Internet — `@[tool:internet]`

Live web search

Use for timely web information beyond your library.

### Fetch — `@[tool:fetch]`

Discover a page and extract readable content

Use when you need full page content, not just search snippets.

### Recipe — `@[tool:recipe]` (advanced palette)

Recipe sites with structured ingredients and steps

Use for structured recipe ingredients and steps from recipe sites.

### Library — `@[tool:library]`

Search your documents

Use to search only your uploaded documents.

### Help — `@[tool:help]`

Search Qube's built-in documentation

Use for how-to questions, settings locations, and troubleshooting.

### Memory — `@[tool:memory]`

Search stored memories

Use to recall facts saved from past chats.

### Scientific literature — `@[tool:science]` (advanced palette)

Alias for @evidence (same routing)

Same routing as @evidence; prefer @evidence in the palette.

### Wikipedia — `@[tool:wikipedia]` (advanced palette)

Wikipedia intro extracts only

Use for quick encyclopedia summaries only.

### PubMed — `@[tool:pubmed]` (advanced palette)

Biomedical literature abstracts

Use for biomedical papers and clinical research.

### arXiv — `@[tool:arxiv]` (advanced palette)

Preprint abstracts (CS, physics, math)

Use for CS, physics, and math preprints.

## Advanced palette tools

These built-in tools are hidden from the default **Tools** browse list until you type `@` and search for the tool id: `recipe`, `science`, `wikipedia`, `pubmed`, `arxiv`.

Type the id (for example `pubmed`) or pick the token once it appears.

## My knowledge presets (dynamic)

Presets you create under **Settings → Knowledge → My knowledge** appear in the `@` palette as **`@[tool:user:…]`** tokens (for example `@[tool:user:biology]`). They bundle selected Live Source adapters or web-fetch domains — not Library folders.

Preset tokens are not listed here because they depend on your configuration. See [Create a knowledge preset](../workflows/create-knowledge-preset.md).

## Single-adapter pins (`@[tool:source:…]`)

Pin one Live Source adapter manually with `@[tool:source:adapter_id]` (for example `@[tool:source:pubmed]`). These tokens are not shown in the palette; use a knowledge preset when you want a repeatable scoped bundle instead.

## Settings and prerequisites

- **Library** (`@[tool:library]`) — enable document search in **Settings → Knowledge** and prepare search models (see [Prepare search models for Library](../workflows/prepare-search-models-for-library.md)).
- **Web / evidence tools** (`@[tool:internet]`, `@[tool:evidence]`, `@[tool:trusted]`, etc.) — configure **Live Sources** and optional API keys in **Settings → Knowledge**. See [Live sources overview](live-sources-overview.md).
- **Help** (`@[tool:help]`) — searches the **Qube Documentation** collection only. Help docs are excluded from normal Library search unless you attach `@help`.
- **Memory** (`@[tool:memory]`) — requires Memory features enabled in **Settings → Memory**.

## Also called

composer attachments, @ mentions, tool tokens, routing tools

## Related

- [Composer attachments](composer-attachments.md) — files, chats, routing order
- [Composer commands](composer-commands.md) — immediate app actions
- [Composer skills](composer-skills.md) — reasoning frameworks (not routing)
- [Knowledge settings](../features/settings/knowledge.md) — Live Sources and My knowledge
