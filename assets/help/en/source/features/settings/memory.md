# Memory

## Common questions

- How does long-term memory work in Qube?
- What is memory enrichment?
- What does memory promotion do?
- What does “Highlight memories that keep coming back” do?

## What it is

**Memory** settings govern how Qube captures, refines, and retains facts about you across conversations. Separate from Library documents, Memory stores distilled snippets the assistant can recall later through **Memory Manager** and automatic context injection.

**Memory pipeline** controls three optional background processes:

- **Enable Memory Enrichment & Reflection** — extracts durable facts, session episodic summaries, and periodic LLM audits that flag suspicious memories (uses more RAM)
- **Promote well-used memories to preferences** — occasionally upgrades frequently used facts into long-term preferences (off by default; requires enrichment)
- **Highlight memories that keep coming back** — marks recurring themes in Memory Manager for **your review**; Qube does not merge or delete memories automatically

**Personalization** sets **Default units** (metric/imperial/inferred) for formatted answers.

## Where to find it

Open **Settings → Memory** (settings section `memory`). Press **?** for the guided tour (`settings.memory`). Review stored entries in **Memory Manager** from the main navigation.

## Also called

long-term memory settings, memory enrichment, memory promotion, memory consolidation, MEMORY & PERFORMANCE, saved memories

## How to…

1. **Choose enrichment level** — Enable **Enable Memory Enrichment & Reflection** when you want richer extraction and audits; disable it to reduce background LLM/RAM use (existing memories still recall).
2. **Configure promotion** — Turn on **Promote well-used memories to preferences** only if you want automatic preference upgrades; pick **Promotion preset** (**Conservative**, **Standard**, **Aggressive**).
3. **Review recurring themes** — Leave **Highlight memories that keep coming back** on to surface patterns in Memory Manager, or disable it for fully manual curation.
4. **Set default units** — Under **Personalization**, choose **Default units** for weather and numeric answers.
5. **Audit in Memory Manager** — Open [Memory Manager](../../features/memory-manager.md) to edit, flag, delete, or export entries regardless of automation settings.

## Controls

<!-- include:generated/controls/memory.md -->

## Related

- [Memory Manager feature](../../features/memory-manager.md) — browse stored entries
- [Manage long-term memory workflow](../../workflows/manage-long-term-memory.md) — curate and export memories
- [Memory vs Library FAQ](../../faq/memory-vs-library.md) — how memory differs from documents
- [Memory not remembering troubleshooting](../../troubleshooting/memory-not-remembering.md) — when recall fails
