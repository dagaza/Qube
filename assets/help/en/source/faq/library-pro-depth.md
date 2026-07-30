# Library Pro depth (precision ingest & retrieval)

## Common questions

- What is Library Pro depth?
- What is precision ingest?
- What is precision retrieval?
- Do I need a Pro license for Library?
- Why is the precision ingest toggle disabled?
- Why is Precision indexing disabled in the import dialog?
- What does the gem badge mean in Library?
- How do I choose Normal vs Precision indexing when importing?
- How do I import a Qube Pro license?
- Do I need to re-import documents after enabling precision ingest?
- What's the difference between precision retrieval and search quality?

## What it is

**Library Pro depth** is an optional **Qube Pro** (or Team) upgrade for users who want maximum Library accuracy on dense documents and large collections.

| Feature | Where | What it does | Cost / tradeoff |
|---------|--------|--------------|-----------------|
| **Precision ingest** | **Library → Import (+)** dialog | **Precision indexing (Pro)** uses embedding-similarity breakpoints during indexing (on top of standard structural chunking) | **Much slower indexing** — often 10–100× more embedding work; precision-indexed docs show a **gem** badge in the sidebar |
| **Default precision on import** | **Settings → Knowledge → Library Pro depth** | Pre-selects **Precision indexing** in the import dialog when licensed (optional; you can still pick **Normal indexing** per upload) | Same indexing cost when you keep that choice |
| **Precision retrieval** | Same Settings card | **Reranks** Library hits with a second bi-encoder pass after hybrid search and MMR | **Extra latency** on each Library query; no re-import needed |

**What stays free:** Standard structural chunking, hybrid vector + text search, MMR diversity, breadcrumb citations, and **Search quality** embedding presets (**Fast**, **Balanced**, **Power**). Pro depth is an optional second layer—not “making Library work.”

**License required:** Precision indexing and the Settings toggles require a valid **Qube Pro** or **Team** license imported under **Settings → Advanced → License**. Without a license:

- **Precision indexing** in the import dialog is **disabled** with a tooltip explaining the Pro requirement.
- Settings toggles show **Pro license required** if you try to enable them.

**Not the same as Search quality** — **Search quality** picks which embedding model indexes your Library. **Precision ingest** changes *how* documents are split before embedding. **Precision retrieval** changes *how* hits are ordered after search—not retrieval profile orchestration or web fetch depth. See [Retrieval profile vs search quality](retrieval-profile-vs-search-quality.md).

## Import flow (precision ingest)

Every Library upload goes through this sequence:

1. Click **+** (**Ingest New Document**) on **Main** or a user folder (not **Qube**).
2. **Choose indexing mode** — **Normal indexing** or **Precision indexing (Pro)**.
3. Pick one or more files in the system file picker (`.txt`, `.md`, `.pdf`, `.epub`).
4. If filenames already exist, confirm **Overwrite Files?** when prompted.
5. Indexing runs; precision-indexed documents show a **gem badge** before the filename in the sidebar. The preview stats line includes **Precision ingest** when that document was indexed in precision mode.

To **change mode** on an existing file, re-import it (overwrite when prompted) and pick the other option.

Documents indexed before this feature was available default to normal indexing (no gem) until re-imported.

## Where to find it

- **Import mode chooser** — **Library → +** after file picker (and optional overwrite prompt)
- **Gem badge** — Library sidebar document list (precision-indexed docs only)
- **Library Pro depth toggles** — **Settings → Knowledge → Library Pro depth**
- **Import license** — **Settings → Advanced → License → Import license file**
- **Browse this guide** — **Library → Qube → faq/library-pro-depth.md**, or ask with **`@[tool:help]`**

## Also called

library pro features, high quality ingest, precision chunking, semantic ingest, library rerank, pro library depth, pro license library, precision indexing, gem badge library, choose indexing mode

## How to…

1. **Import a Pro license** — Obtain a `.qube-license` file from your purchase or administrator. Open **Settings → Advanced → License → Import license file**. Team and Enterprise licenses include Pro Library depth.
2. **Index with precision** — **Library → +**, pick files, choose **Precision indexing (Pro)** in the dialog. Wait for indexing to finish; confirm the **gem** badge appears on the document row.
3. **Optional: default precision on import** — **Settings → Knowledge → Library Pro depth → Default precision ingest on import** pre-selects precision in the import dialog. Read the cost warning before saving.
4. **Enable precision retrieval** — Toggle **Precision retrieval** on the same card. Takes effect on the **next** Library query (`@[tool:library]`, Local Knowledge Base, or automatic triggers)—no re-import needed.
5. **Disable after trial** — Turn Settings toggles off; standard chunking and MMR retrieval resume. Previously indexed precision chunks remain until you re-ingest those files.
6. **Check license status** — If toggles snap off after restart, re-import the license or contact support—cached license may have been removed via **Remove cached license**.

## Related

- [Import documents to Library workflow](../workflows/import-documents-to-library.md) — full upload steps including indexing mode
- [Enable Library Pro depth workflow](../workflows/enable-library-pro-depth.md) — license + Settings setup
- [Knowledge settings](../features/settings/knowledge.md) — all Knowledge controls including Library Pro depth
- [Library feature](../features/library.md) — ingest, preview, gem badge, and chat with documents
- [Advanced settings](../features/settings/advanced.md) — import and remove license
- [Retrieval profile vs search quality](retrieval-profile-vs-search-quality.md) — disambiguate Fast/Balanced naming
- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — Search quality presets (required before any Library search)
