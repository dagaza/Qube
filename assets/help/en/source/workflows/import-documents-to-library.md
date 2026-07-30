# Import documents to Library

## Common questions

- How do I add PDFs or notes to my Library?
- Can I ingest a whole folder?
- How long does document ingestion take?
- Why is the Import (**+**) button grayed out?

## What it is

Library holds your personal documents—PDFs, markdown, plain text, and EPUB—chunked and embedded so Qube can search them during chat. Importing is the first step before Library search can use your files.

## Where to find it

Open **Library** from the main navigation. User content lives under **Main**; system help docs live under **Qube** (read-only).

## Also called

add documents, ingest files, upload to library, import PDF, build document collection

## How to…

1. Open **Library** and click the **Main** folder row (or another user folder—not **Qube**).
2. Click **+** (**Ingest New Document**) in the sidebar header.
3. In **Choose indexing mode**, pick **Normal indexing** (free, fast) or **Precision indexing (Pro)** (slower, finer chunking; requires Pro license—button disabled with tooltip if unlicensed). See [Library Pro depth FAQ](../faq/library-pro-depth.md).
4. Select one or more supported files (`.txt`, `.md`, `.pdf`, `.epub`) in the system file picker.
5. If a filename already exists, confirm **Overwrite Files?** when prompted.
5. If **+** is disabled, click **Main** even if you were previewing help docs in **Qube**; see [Library Import button disabled troubleshooting](../troubleshooting/library-import-button-disabled.md).
6. Wait for ingestion to finish—progress appears under the Library header and in notifications. Precision-indexed documents show a **gem badge** in the sidebar.
7. Verify documents appear in the sidebar and the preview shows **Chunks Indexed** with a non-zero count before searching them in chat.
8. Optionally move items into subfolders to keep searches organized.

**Can I ingest a whole folder?** No—use **+** and select one or more files at a time.

## Related

- [Chat with a library document](chat-with-a-library-document.md) — use imported files in chat
- [Prepare search models for library](prepare-search-models-for-library.md) — embeddings required for search
- [Knowledge settings](../features/settings/knowledge.md) — search quality and triggers
- [Library Import button disabled troubleshooting](../troubleshooting/library-import-button-disabled.md) — **+** grayed out on Qube folder
