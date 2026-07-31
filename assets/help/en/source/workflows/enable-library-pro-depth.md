# Enable Library Pro depth

## Common questions

- How do I turn on precision ingest?
- How do I enable precision retrieval?
- Where do I import a Pro license?
- How do I choose Normal vs Precision indexing when importing?

## What it is

This workflow enables **Library Pro depth** — optional **precision ingest** and **precision retrieval** for Qube Pro (or Team) licensees. Standard Library search remains available without a license; this workflow is for users who want maximum indexing and ranking accuracy and accept slower ingest and query latency.

## Where to find it

**Settings → Knowledge → Library Pro depth** (after importing a license under **Settings → License**).

## Also called

enable pro library, turn on precision ingest, turn on precision retrieval, library pro setup, import pro license

## Steps

1. **Obtain a license file** — You need a `.qube-license` issued for **Pro**, **Team**, or **Enterprise** tier.
2. **Import the license** — Open **Settings → License → Import license file** and select the file. Qube validates and caches the license locally.
3. **Open Library Pro depth** — Go to **Settings → Knowledge** and scroll to **Library Pro depth**. Both toggles should be enabled (clickable). If not, confirm the license tier includes Pro capabilities.
4. **Optional: default precision on import** — Turn on **Default precision ingest on import** to pre-select precision in the import dialog. Read the confirmation dialog (indexing can take much longer). Click continue to save the preference.
5. **Import with precision** — Click **+** in **Library**, pick files, then in **Choose indexing mode** select **Precision indexing (Pro)**. Precision-indexed documents show a **gem** badge in the sidebar. To change mode later, re-import (overwrite when prompted).
6. **Optional: precision retrieval** — Turn on **Precision retrieval**. No re-import needed; the next Library query uses the rerank pass.
7. **Verify in chat** — Attach **`@[tool:library]`** or enable **Local Knowledge Base**, ask a question grounded in your Library, and check citations. Use **Sources → INSPECT RETRIEVAL** if you need to debug empty results.

## Tips

- Run **Prepare search models** under **Search quality** before ingesting or searching—Pro depth does not replace embedding model setup.
- Start with **precision retrieval** if you want better ordering without re-indexing everything.
- Use **precision ingest** for dense PDFs, contracts, and long technical papers where section boundaries matter most.

## Related

- [Library Pro depth FAQ](../faq/library-pro-depth.md) — what each toggle does and licensing
- [Knowledge settings](../features/settings/knowledge.md) — Library Pro depth card location
- [Import documents to Library](import-documents-to-library.md) — upload steps and indexing mode chooser
- [License settings](../features/settings/license.md) — license import and removal
