<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->
# Composer attachments (@file, @chat, routing)

## Common questions

- What is the difference between `@[file:…]`, `@[chat:…]`, and `@[tool:…]`?
- Which `@` attachment controls routing when I mix several?
- How do I reference another conversation in chat?

## What attachments are

The `@` palette has five categories: **Files**, **Conversations**, **Tools**, **Skills**, and **Commands**. This page covers the three **routing** attachment kinds that steer retrieval and context for a turn.

Type **`@`** in the composer to browse categories, or keep typing to search everything at once. Pick **1–5** on the category menu for a shortcut; **Enter** or **Tab** selects.

## Files — `@[file:filename.pdf]`

Reference an indexed **Library** document. Forces search scoped to that file only (not your whole Library). Filenames containing **`]`** cannot be inserted from the palette.

Configure Library search in **Settings → Knowledge** (master RAG switch and search models).

## Conversations — `@[chat:session-id::Title]`

Pull another chat's transcript into this turn. The referenced history replaces this conversation's history for that turn only (about **7,000** characters). It does not merge unrelated turns from the current chat.

## Tools — `@[tool:…]`

Route the turn to web discovery, Live Sources, Library search, Memory, Help, deep research, or a custom preset. See [Composer tools](composer-tools.md) for every built-in token.

## Routing rule (order matters)

You can insert several routing tokens, but only the **first** one in your message (left-to-right) controls behaviour — the first among `@[file:…]`, `@[chat:…]`, or `@[tool:…]`. Put the attachment you want **first**.

Example: `@[tool:internet] @[file:doc.pdf]` uses web discovery, not the file.

**Skills** (`@[skill:…]`) and **commands** do not participate in this rule. Skills add prompt guidance; commands run immediately and are not sent to the model.

## Mixing skills with routing attachments

Skills pair well with one routing attachment. Example:

`@[skill:research_synthesis] @[tool:library] Summarize my uploaded notes.`

## When attachments are skipped

- Explicit **“remember …”** turns skip all attachments and all skills (including forced).
- Unknown `@[tool:…]` or `@[skill:…]` IDs are ignored (logged); other tokens may still apply.

## Also called

composer routing, file attachments, conversation references, @ mentions

## Related

- [Composer tools](composer-tools.md) — built-in and preset `@[tool:…]` tokens
- [Composer skills](composer-skills.md) — `@[skill:…]` reasoning frameworks
- [Composer commands](composer-commands.md) — immediate palette actions
- [What do @ mentions do FAQ](../faq/what-do-at-mentions-do.md)
- [Chat with a library document](../workflows/chat-with-a-library-document.md)
