# Delete memory entries

## Common questions

- How do I delete something Qube remembered about me?
- Does deleting a memory stop Qube from saving it again?
- What is the **negative list**?
- How do I bulk-delete memories?
- Is deleting memory the same as clearing chat history?

## What it is

**Long-term memory** holds distilled facts Qube extracted from past conversations — separate from **Conversations** chat history and separate from **Library** documents.

Deleting in **Memory Manager** permanently removes the row from your local memory store **and** records a **negative pattern** so the enrichment pipeline is unlikely to recreate a similar fact from future chats.

## Where to find it

Open **Memory Manager** from the left navigation (memory icon). Use tier, category, **Search memory text…**, and **Flagged only** filters to find the row you want.

Automation toggles (whether Qube captures **new** facts) live in **Settings → Memory** — deleting does not turn enrichment off by itself.

## Also called

forget memory, remove saved memory, clear what qube learned, delete long-term memory, memory negative list, bulk delete memories

## Delete one memory

1. Open **Memory Manager** and locate the card.
2. Click **Delete** on the row.
3. Confirm in the **Delete Memory** dialog — deletion is **permanent**.
4. Qube adds the fact to `~/.qube/memory_negatives.json` (the negative list) so near-duplicate extraction candidates are rejected on later enrichment passes.

To fix wording without deleting, use **Edit** instead — provenance and metadata are kept and the text is re-embedded.

## Bulk delete filtered memories

1. Narrow the list with **All tiers**, **All categories**, **Search memory text…**, or **Flagged only**.
2. Click **Delete all visible** — removes **every row currently shown** after confirmation.
3. Each deleted row is added to the negative list the same way as single deletes.
4. Use **Export visible** first if you want a Markdown backup under `~/.qube/exports/` before bulk deletion.

## What the negative list does

| Aspect | Behaviour |
|--------|-----------|
| **Storage** | `~/.qube/memory_negatives.json` on your machine (not LanceDB) |
| **Trigger** | Every **Delete** in Memory Manager (single or bulk) |
| **Effect** | New extraction candidates very similar to a deleted memory are **skipped** |
| **Limit** | FIFO cap (~500 entries) — oldest patterns rotate out |
| **Not deleted by** | Editing text, flagging for review, or disabling enrichment |

The negative list prevents “I deleted this yesterday but Qube saved it again after a similar chat.” It does **not** erase chat transcripts in **Conversations**.

## Delete memory vs other “forget” actions

| Action | What it removes | Recreated automatically? |
|--------|-----------------|------------------------|
| **Delete** in Memory Manager | One long-term memory row + negative-list entry | Unlikely for near-duplicates |
| **Delete all visible** | All filtered memory rows | Same per row |
| Clear / delete a **Conversations** thread | Chat messages in that session | Memory may still exist if already extracted |
| Disable **Enable Memory Enrichment & Reflection** | Nothing existing — stops **new** capture | Old rows remain until you delete them |
| **Uninstall Qube** (optional data wipe) | Entire `~/.qube` folder if you choose | N/A |

See [Conversations vs memory context](conversations-vs-memory-context.md) and [Memory vs Library](memory-vs-library.md).

## Stop new memories without deleting old ones

Open **Settings → Memory** and turn off **Enable Memory Enrichment & Reflection** to pause automatic extraction. Review or delete existing rows anytime in Memory Manager.

## Related

- [Memory Manager](../features/memory-manager.md) — filters, Edit/Flag/Delete, export
- [Manage long-term memory workflow](../workflows/manage-long-term-memory.md) — review and tune automation
- [Memory settings](../features/settings/memory.md) — enrichment, promotion, consolidation
- [Memory not remembering troubleshooting](../troubleshooting/memory-not-remembering.md) — recall issues (not deletion)
- [Uninstall Qube workflow](../workflows/uninstall-qube.md) — remove all local data
