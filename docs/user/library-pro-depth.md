# Library Pro depth (Qube Pro)

Optional **precision ingest** and **precision retrieval** for users with a **Qube Pro** (or Team) license who want maximum Library accuracy on dense documents and large collections.

In-app help (recommended): **Library → Qube → faq/library-pro-depth.md**, or ask with **`@[tool:help]`** in Conversations.

## What stays free

Standard Library features work without a paid license:

- Document import (**PDF**, **EPUB**, **TXT**, **MD**)
- Structural chunking and hybrid search
- MMR diversity and citation breadcrumbs
- **Search quality** presets (**Fast**, **Balanced**, **Power**) for embedding models

Pro depth is an **optional second layer**—not required for Library search to work.

## Pro features

| Feature | Where | What it does | Tradeoff |
|---------|--------|--------------|----------|
| **Precision ingest** | **Library (+)** import dialog | Splits large sections at embedding-similarity breakpoints during indexing | Much slower indexing (often 10–100× more embedding work); precision-indexed docs show a **gem** badge |
| **Default precision on import** | Settings → Knowledge → Library Pro depth | Pre-selects precision in the import dialog (optional) | Same cost as precision ingest when you keep that choice |
| **Precision retrieval** | Same card | Reranks hits with a bi-encoder pass after hybrid search + MMR | Extra latency on each Library query; no re-import needed |

## Import flow

Every upload: **Library → +** → **Choose indexing mode** → system file picker → optional overwrite prompt → indexing. Precision-indexed docs show a **gem** in the sidebar; preview stats include **Precision ingest**.

## Setup

1. **Import a license** — Obtain a `.qube-license` file (Pro, Team, or Enterprise). Open **Settings → License → Import license file**.
2. **Import with precision** — Click **+** in **Library**, pick files, then choose **Precision indexing (Pro)** in the dialog. Without a license, that button is disabled with a tooltip explaining the requirement.
3. **Optional default** — Turn on **Default precision ingest on import** under **Library Pro depth** to pre-select precision each time.
4. **Optional retrieval** — Turn on **Precision retrieval** for reranked query results (no re-import needed).
5. **Search as usual** — Attach **`@[tool:library]`**, enable **Local Knowledge Base**, or use automatic Library triggers.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Toggle disabled or snaps off | Import a valid Pro/Team license under **Settings → License** |
| **Pro license required** dialog | Same—no paid license is cached locally |
| Precision indexing button disabled in import dialog | Import a Pro license; precision ingest requires Pro |
| Precision ingest seems unchanged | Re-import documents and choose **Precision indexing**; mode applies per import job |
| Slow indexing | Expected with precision ingest; choose **Normal indexing** for faster imports |

## Related

- In-app: [Library Pro depth FAQ](../../assets/help/en/faq/library-pro-depth.md)
- In-app: [Enable Library Pro depth workflow](../../assets/help/en/workflows/enable-library-pro-depth.md)
- [How to use Qube — Library](how-to-use.md#library-document-retrieval)
- Engineering design: [library_chunking_retrieval_design.md](../library_chunking_retrieval_design.md) §14
