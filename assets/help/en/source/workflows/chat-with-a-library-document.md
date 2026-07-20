# Chat with a Library document

## Common questions

- How do I ask questions about my uploaded PDFs?
- Do I need `@[tool:library]` every time?
- Why did chat ignore my document?

## What it is

Once documents are ingested, you can ground assistant replies in your files using **`@[tool:library]`**, the **Local Knowledge Base** tools-panel toggle, and/or custom **Library search phrases** with **Enable NLP Auto-Activator**. The model receives retrieved passages as evidence rather than guessing from general knowledge alone.

## Where to find it

Use **Conversations** for chat. Attach routing in the composer input or enable **Local Knowledge Base** / auto-activator in the tools panel. Configure custom phrases in **Settings → Knowledge → Library search phrases**.

## Also called

RAG chat, document Q&A, ask my files, library-grounded chat, search my documents

## How to…

1. Confirm documents in **Library → Main** finished ingesting (**Chunks Indexed** > 0) and search models are ready under **Settings → Knowledge**.
2. Start a new message in **Conversations**.
3. Add **`@[tool:library]`**, enable **Local Knowledge Base** in the tools panel, and/or rely on **Enable NLP Auto-Activator** matching your custom **Library search phrases** (if configured).
4. Ask a specific question referencing the topic; vague prompts retrieve weaker passages.
5. Review cited sources via in-message links or the **Sources** button, which opens an in-app preview of the retrieved passage. Open **Library** separately if you need the full document.

## Related

- [Import documents to Library](import-documents-to-library.md) — get files into Library first
- [Library search returns nothing troubleshooting](../troubleshooting/library-search-returns-nothing.md) — empty retrieval
- [Live sources vs Library search FAQ](../faq/live-sources-vs-library-search.md) — Library vs online sources
- [Composer tools reference](../reference/composer-tools.md) — `@[tool:library]` details
