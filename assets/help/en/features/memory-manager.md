# Memory Manager

## Common questions

- Where do I see what Qube remembered about me?
- How do I delete or edit a memory?
- What are memory tiers (preference, knowledge, episode, context)?
- How is Memory Manager different from chat history?
- What does **Flagged only** do?
- How do I export my memories?

## What it is

**Memory Manager** is the main-window view for browsing and curating **long-term memory** — facts Qube distilled from past conversations and persisted for future turns. It is separate from **Conversations** chat history and separate from **Library** documents.

Use tier and category filters, **Search memory text…**, and **Flagged only** to narrow the list. Each memory appears as a card with tier badges, metadata, and **Edit**, **Flag**, or **Delete** actions. **Export visible** writes the filtered list to Markdown under `~/.qube/exports/`. **Delete all visible** permanently removes matching rows and adds them to the negative list so similar facts are not re-extracted.

Background capture, enrichment, and promotion behaviour are configured in **Settings → Memory** (see [Memory settings](settings/memory.md)).

Press **?** beside the page title for the guided tour (`memory_manager`).

## Where to find it

Click **Memory Manager** in the left navigation (memory icon). Press **?** in the title row for the guided tour.

## Also called

memory browser, saved memories, long-term memory view, memory cards, recall manager, browse memories

## How to…

1. **Review what is stored** — Scroll memory cards grouped by section (**Almost promoted**, **Promotion candidates**, **Flagged for review**, then by category). The subtitle reads **Review what Qube remembers about you.**
2. **Filter by tier** — Open **All tiers** and pick **Preferences**, **Knowledge**, **Episodes**, or **Context** (structural tiers in the memory store).
3. **Filter by category** — Open **All categories** and pick **Preference**, **Identity**, **Project**, **Knowledge**, **Context**, or **Episode** (labels on each card).
4. **Find text** — Type in **Search memory text…** (matches memory body text in the current filter set).
5. **Review flagged entries** — Toggle **Flagged only** to show memories you marked for review.
6. **Edit a memory** — Click **Edit** on a card, update the text in the dialog (**Update the memory text. Provenance and metadata are kept.**), and save. Qube re-embeds the updated text.
7. **Flag or unflag** — Click **Flag** / **Unflag** on a card to mark for the next reflection pass.
8. **Delete one memory** — Click **Delete** and confirm (**Delete Memory**). Deletion is permanent and adds the fact to the negative list.
9. **Export or bulk delete** — **Export visible** saves filtered rows to `~/.qube/exports/memory_YYYYMMDD.md`. **Delete all visible** removes every row currently shown after confirmation.
10. **Refresh the list** — Click the sync icon (**Reload memories from disk**) after external changes or if the list looks stale.

**Presentation profile** (top card) summarises synced presentation preferences (units, locale, display name, verbosity) when available.

**Recurring themes** appears when enough patterns emerge across visible memories.

## Controls

Single main-stage layout (no folder sidebar). Filters apply to the in-memory list loaded from the memory store (up to 2000 rows per refresh).

### Title row

| Control | What it does |
|---------|----------------|
| **Memory Manager** | Page title |
| **?** (guided tour) | Starts the Memory Manager tour |
| Subtitle | **Review what Qube remembers about you.** |
| Sync icon | **Reload memories from disk** — reloads from the store |

### Presentation profile card

| Area | What it shows |
|------|----------------|
| **Presentation profile** | Synced units, locale, display name, verbosity (with provenance), or guidance to set preferences in Settings / chat |

### Filter and bulk actions

| Control | What it does |
|---------|----------------|
| **All tiers** | Filter by tier: **All tiers**, **Preferences**, **Knowledge**, **Episodes**, **Context** |
| **All categories** | Filter by category label on each memory |
| **Flagged only** (toggle) | **Show only memories flagged for review** |
| **Search memory text…** | **Search memory text** — substring match on memory body |
| **Delete all visible** | **Delete all memories currently shown in the list** (confirmed) |
| **Export visible** | **Export visible memories to Markdown under ~/.qube/exports/** |

### Recurring themes card

| Area | What it shows |
|------|----------------|
| **Recurring themes** | Theme labels with counts when patterns are detected (hidden when none) |

### Status banner

Messages such as **Loading memories…**, **No memories yet. Qube will start remembering durable facts as you chat.**, **No memories match the current filter.**, or export confirmation with file path.

### Memory cards (list sections)

Cards may appear under **Almost promoted**, **Promotion candidates**, **⚑ Flagged for review**, or category headers (**preference**, **identity**, **project**, etc.).

| Element | Meaning |
|---------|---------|
| Tier badge | **PREF**, **KNOW**, **EP**, or **CTX** (preference / knowledge / episode / context) |
| Category badge | Uppercase category label from the memory payload |
| Meta line | **subject**, **origin**, **conf**, **decay**, **used** retrieval/citation counts when present |
| **FLAGGED** badge | Marked for review |
| **ACTION** badge | Action-sensitive memory (hover for constraints / expiry) |
| **STAGED** badge | Consolidation hints pending |
| **PROFILE** badge | Presentation preference synced to profile policy |
| Body | Memory text (selectable) |
| Provenance | Quoted source excerpt when available |
| **topics:** | Episode summaries only — topic tags |
| **Edit** | **Edit this memory** |
| **Flag** / **Unflag** | **Flag for review** / **Remove review flag** |
| **Delete** | **Delete this memory** |

Promotion-candidate cards may show a score breakdown in the card tooltip.

## Related

- [Memory settings](settings/memory.md) — enrichment, promotion, consolidation
- [Manage long-term memory workflow](../workflows/manage-long-term-memory.md) — curation workflow
- [Memory vs Library FAQ](../faq/memory-vs-library.md) — memory vs uploaded files
- [Conversations vs memory context FAQ](../faq/conversations-vs-memory-context.md) — chat history vs recall
- [Memory not remembering troubleshooting](../troubleshooting/memory-not-remembering.md) — when recall fails
- [Conversations](../conversations.md) — chat history and `@[tool:memory]`
