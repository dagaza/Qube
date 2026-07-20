# Conversations vs memory context

## Common questions

- Does Qube remember this chat automatically?
- What's the difference between chat history and Memory Manager?
- Why did the assistant recall something from an old conversation?

## What it is

**Conversations** keeps the **current thread's message history** — what you and the assistant said in that chat session. **Long-term memory** is a separate store of distilled facts Qube may inject into future turns when relevant.

**Existing memories are recalled even when Memory Enrichment is off.** New automatic capture requires **Enable Memory Enrichment & Reflection**; **Promote well-used memories to preferences** optionally upgrades frequently used facts (requires enrichment). **Highlight memories that keep coming back** only adds review nudges in Memory Manager—it does not merge or delete rows automatically.

Chat history is always scoped to the open conversation. Memory entries can appear across sessions when the router finds them relevant.

## Where to find it

- **Chat history** — **Conversations** sidebar and transcript
- **Stored memories** — **Memory Manager** in the main navigation
- **Memory behaviour** — **Settings → Memory**

## Also called

chat history vs memory, conversation context vs long-term recall, session history, does Qube remember me, past chats vs saved facts

## How to…

1. **Re-read an old thread** — Select the conversation in the **Conversations** sidebar; history reloads for that session only.
2. **See what was remembered long-term** — Open **Memory Manager** and search or filter tiers.
3. **Stop automatic capture** — Disable **Enable Memory Enrichment & Reflection** to stop new fact extraction; disable **Promote well-used memories to preferences** to stop tier upgrades; disable **Highlight memories that keep coming back** to stop review nudges only.
4. **Remove a bad memory** — Delete or edit the row in **Memory Manager**; chat history is unaffected.
5. **Ground on a document instead** — Use **Library** and **`@[tool:library]`** when you need file content, not conversational recall.

## Related

- [Conversations](../features/conversations.md) — chat UI and composer
- [Memory Manager](../features/memory-manager.md) — browse and edit memories
- [Memory settings](../features/settings/memory.md) — automation toggles
- [Memory vs Library FAQ](memory-vs-library.md) — files vs remembered facts
