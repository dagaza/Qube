# Library

## Common questions

- Where do I upload documents?
- What file types can I add to Library?
- How do I search my files?
- What is the **Qube** folder vs **Main** folder?
- Why is **Ingest New Document** (+) grayed out?
- How do I chat with a document I uploaded?
- How do I rename or move a document?
- Why does preview say “Reconstructing document from vector space…”?
- Can I set a background behind the document preview?

## What it is

**Library** stores documents Qube can chunk, embed, and retrieve during chat. Ingested files become searchable when you attach `@[tool:library]`, enable Library retrieval in Conversations, or use automatic Library triggers from **Settings → Knowledge**.

Folders organize documents. **Main** is the default user folder for your uploads. **Qube** is reserved for app-managed content (including this help corpus). User ingest into **Qube** is blocked; selecting a help document for preview does not change the ingest target unless you explicitly click the **Qube** folder row.

The sidebar lists folders and documents. The preview pane shows reconstructed text, metadata, and reading controls. Optional **library wallpaper** can sit behind the preview area—configure it under **Settings → Themes → Wallpapers**. A floating **Chat with document** button appears when a file is open.

Press **?** in the sidebar header for the guided tour (`library`). This page summarizes controls for retrieval; the tour walks the layout spatially.

## Where to find it

Click **Library** in the left navigation (book icon). Press **?** in the document list header for the guided tour.

Built-in help lives under **Library → Qube** (also from **Settings → Help → Open Qube documentation**).

## Also called

document library, knowledge base files, uploaded documents, file manager, ingest documents, my files, knowledge base

## How to…

1. **Ingest a document** — Select **Main** or a user folder (active folder highlight). Click **+** (**Ingest New Document**) and choose files. Supported types in the file picker: **`.txt`**, **`.md`**, **`.pdf`**, **`.epub`**. A progress row appears while indexing runs. If search models are not ready, Qube prompts you to prepare them first (see [Prepare search models workflow](../workflows/prepare-search-models-for-library.md)).
2. **Replace an existing file** — Importing a duplicate filename prompts **Overwrite Files?**; confirming removes the old index before re-ingesting.
3. **Find a file** — Type in **Search titles or indexed text…** to match titles or indexed body text (flat results, up to 200 hits). Clear search to return to folder browse mode.
4. **Preview a document** — Single-click a row. The header shows size and **Chunks Indexed**; body text is reconstructed from the vector index. Use the preview toolbar for font, spacing, and layout.
5. **Chat with a file** — With a document open, click the floating **Chat with document** button (confirm in the dialog). Qube opens **Conversations**, starts a new thread, and prefills an `@[file:…]` attachment—not double-click (double-click on a **folder** row expands or collapses that folder).
6. **Organize folders** — **New folder** in the header; sort with the sort icon (**Sort folders and items** → **By Name** or **By Date**). Folder **⋮**: **Rename Folder** or **Delete Folder** on user folders only (**Main** and **Qube** are system folders).
7. **Manage one document** — Document **⋮** (**Document actions**): **Rename Document**, **Move to folder**, **Delete Document** (removes metadata and index entries).
8. **Browse help** — Open **Qube** for built-in documentation (`reference/`, feature pages, workflows). **Qube** starts **collapsed** so **Main** stays front and center; expand it when you need help articles.

**Not supported in v1:** drag-and-drop reorder of documents; folder ZIP export (that exists in Conversations, not Library).

## Controls

Grouped like the Library layout. Preview readability settings apply to the **current session only** (not saved to Settings).

### Sidebar (folders and documents)

| Control | What it does |
|---------|----------------|
| **?** (guided tour) | Starts the Library tour |
| **+** | **Ingest New Document** — disabled on **Qube** with tooltip explaining reserved folder |
| **New folder** | Create a folder for grouping uploads |
| Sort icon | **Sort folders and items** → **By Name** or **By Date** |
| **Search titles or indexed text…** | Search titles and indexed content |
| Ingest progress row | Spinner / bar while files are indexing |
| Folder row | Click to set active ingest folder; chevron or double-click toggles expand/collapse |
| Folder **⋮** | **Rename Folder**, **Delete Folder** (user folders only) |
| Document row | Single-click to preview; shows filename and size; summary blurb in tooltip when available |
| Document **⋮** | **Rename Document**, **Move to folder**, **Delete Document** |
| **Main** | Default user upload folder |
| **Qube** | App-managed docs; ingest blocked; collapsed by default |

### Preview toolbar (above document body)

| Control | What it does |
|---------|----------------|
| **A−** / **A+** | **Decrease preview font** / **Increase preview font** (Shift+click: larger step) |
| Line spacing icon | Cycles **Compact**, **Comfortable**, **Relaxed** spacing |
| Text alignment icon | Toggles **Left** and **Justified** alignment |
| Reader focus | **Reader focus: dim document header** — dims title/metadata |
| High contrast | **High contrast (document preview)** |
| Layout width icon | **Narrow column** (~800px) vs **Wide column** (~1200px) |

### Preview content

| Area | What it shows |
|------|----------------|
| Document title | Selected filename (wraps long names) |
| Stats line | **Size**, **Chunks Indexed**, optional summary blurb |
| Body | Reconstructed text from the vector store, or placeholder when none selected |
| **Chat with document** (FAB) | Starts a grounded chat in **Conversations** with `@[file:…]` prefilled |

Empty selection placeholder: **Select a document from the left to view its contents.**

## Related

- [Import documents to Library workflow](../workflows/import-documents-to-library.md) — first-time upload steps
- [Chat with a library document workflow](../workflows/chat-with-a-library-document.md) — ask questions about a file
- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — embeddings before search works
- [Library search returns nothing troubleshooting](../troubleshooting/library-search-returns-nothing.md) — empty retrieval
- [Library Import button disabled troubleshooting](../troubleshooting/library-import-button-disabled.md) — **+** grayed out
- [Memory vs Library FAQ](../faq/memory-vs-library.md) — documents vs remembered facts
- [Live sources vs Library search FAQ](../faq/live-sources-vs-library.md) — Library vs online sources
- [Knowledge settings](settings/knowledge.md) — search quality and automatic triggers
- [Conversations](../conversations.md) — composer, `@[tool:library]`, and citations
