# Memory not remembering

## Common questions

- Why doesn't Qube recall what I told it yesterday?
- Memory Manager is empty—what enables capture?
- The assistant forgot a fact I saved manually

## What it is

Recall problems usually mean the fact was never stored, memory automation is disabled for **new** capture, the assistant is answering from general knowledge instead of Memory context, or the entry was deleted/edited in Memory Manager. Memory is not the same as chat history in a single conversation thread.

**Existing memories still recall when enrichment is off**—enrichment controls new automatic extraction, not retrieval of rows already saved.

## Where to find it

Review entries in **Memory Manager** and toggles in **Settings → Memory** (enrichment, promotion, highlight recurring themes).

## Also called

forgot my preference, memory not working, long-term memory empty, assistant doesn't remember me

## How to…

1. Open **Memory Manager** and search for the fact—confirm it exists and text is correct.
2. Enable **Enable Memory Enrichment & Reflection** if you expect **new** facts to be captured automatically.
3. Turn on **Promote well-used memories to preferences** only if you want frequently used facts upgraded to preferences (requires enrichment).
4. Ask explicitly in chat (“What do you remember about …?”) after starting a **new** conversation to test recall.
5. Do not disable **Highlight memories that keep coming back** expecting it to stop recall—it only flags recurring rows for review.
6. Do not expect recall from Library documents—import facts to Memory or reference files with **`@[tool:library]`**.

## Related

- [Manage long-term memory workflow](../workflows/manage-long-term-memory.md) — review and edit entries
- [Memory settings](../features/settings/memory.md) — automation toggles
- [Memory vs Library FAQ](../faq/memory-vs-library.md) — different storage roles
