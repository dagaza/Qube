# Chat with a Library document

## Common questions

- How do I ask questions about my uploaded PDFs?
- Do I need `@[tool:library]` every time?
- Why did chat ignore my document?

## What it is

Once documents are ingested, you can ground assistant replies in your files using **`@[tool:library]`**, the **Local Knowledge Base** tools-panel toggle, and/or custom **Library search phrases** with **Enable NLP Auto-Activator**. The model receives retrieved passages as evidence rather than guessing from general knowledge alone.

## Where to find it

Use **Conversations** for chat. Attach routing in the composer input or enable **Local Knowledge Base** / auto-activator in the tools panel. Configure custom phrases in **Settings → Knowledge → Library search phrases**. Turn sticky per-chat behavior and the scope hint on or off under **Settings → General → Composer**.

## Also called

RAG chat, document Q&A, ask my files, library-grounded chat, search my documents

## How to…

1. Confirm documents in **Library → Main** finished ingesting (**Chunks Indexed** > 0) and search models are ready under **Settings → Knowledge**.
2. Start a new message in **Conversations**.
3. Scope **one file** with **`@[file:…]`** (or upload from Conversations). The **file chip stays** for this chat after you send; each new message **searches that file again** until you remove the chip. Voice follow-ups in the same chat keep the same scope even when the composer is empty.
4. For **all** ingested documents, add **`@[tool:library]`**, enable **Local Knowledge Base** in the tools panel, and/or rely on **Enable NLP Auto-Activator** matching your custom **Library search phrases** (if configured).
5. Ask a specific question referencing the topic; vague prompts retrieve weaker passages.
6. Review cited sources via in-message links or the **Sources** button, which opens an in-app preview of the retrieved passage. Open **Library** separately if you need the full document.

## Do I need to attach the file every turn?

No. Once a **file chip** is on this conversation, follow-up messages (typed or voice) stay scoped to that document until you click **✕** on the chip or start **Deep research** (which clears document scope). Switching away and back to a conversation restores the file chip for that chat.

## Related

- [Import documents to Library](import-documents-to-library.md) — get files into Library first
- [Library search returns nothing troubleshooting](../troubleshooting/library-search-returns-nothing.md) — empty retrieval
- [Live sources vs Library search FAQ](../faq/live-sources-vs-library-search.md) — Library vs online sources
- [Composer tools reference](../reference/composer-tools.md) — `@[tool:library]` details
