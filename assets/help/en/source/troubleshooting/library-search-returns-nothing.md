# Library search returns nothing

## Common questions

- Why does `@[tool:library]` find no documents?
- Chat ignores my uploaded PDFs—what is wrong?
- Library search worked before but now returns empty results

## What it is

This issue means retrieval ran but returned **no passages** from your Library, or chat behaved as if no documents matched. Common causes include missing search models, empty collections, overly strict search quality without prepared embeddings, or questions that do not align with ingested text.

## Where to find it

Check **Library → Main** for ingest status (**Chunks Indexed**), **Settings → Knowledge** for search quality and **Prepare search models**, and attach **`@[tool:library]`** or enable **Local Knowledge Base**.

## Also called

empty library search, RAG returns nothing, no document hits, library attachment useless, zero retrieval results

## How to…

1. Confirm documents in **Library → Main** show **Chunks Indexed** with a non-zero count—not still ingesting or failed.
2. Verify search models are ready under **Settings → Knowledge** (see [Prepare search models workflow](../workflows/prepare-search-models-for-library.md)).
3. Lower **Search quality** temporarily to **Fast** to test basic embedding retrieval.
4. Rephrase your question with terms that appear in the document title or body.
5. Explicitly attach **`@[tool:library]`**, enable **Local Knowledge Base**, or match a custom **Library search phrase** if auto-activator is on.
6. Review **Refresh last retrieval trace** under **Settings → Knowledge → Diagnostics** if ingestion succeeded but search still fails.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — triggers and search quality
- [Prepare search models workflow](../workflows/prepare-search-models-for-library.md) — model readiness
- [Search models not ready troubleshooting](search-models-not-ready.md) — models missing
