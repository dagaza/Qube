# Sticky library document for one chat

## Common questions

- Do I need to attach my PDF every message?
- Why is the file chip still in the composer after I send?
- Does voice still use my document when the composer looks empty?

## What it is

When you attach **`@[file:…]`** (or upload from Conversations), Qube can keep that **one library file** scoped to the **current conversation** until you remove the chip. Each new message **searches that file again** (chunk retrieval with a filename filter)—the model does not “remember” the PDF from earlier turns.

This is separate from **`@[tool:library]`**, which searches **all** ingested documents.

## Where to find it

**Conversations** composer file chip, **Settings → General → Composer** toggles:

- **Keep attached library document for this chat** (default on)
- **Show “scoped to this document” hint above composer chips** (default on)

## Also called

persistent file attach, document scope per chat, sticky file chip, Cursor-style doc chat

## How to…

1. Attach or upload a file and send a question.
2. For follow-ups, type or use voice **without** re-attaching—the file chip may remain above the composer (or scope applies in the background for voice).
3. Remove scope by clicking **✕** on the file chip.
4. **Deep research** clears document scope for that send path.
5. If the file was deleted from **Library**, Qube clears scope and shows a warning.

## Related

- [Chat with a library document](../workflows/chat-with-a-library-document.md)
- [Add a document from Conversations](../workflows/add-document-from-conversations.md)
- [What do @ mentions do?](what-do-at-mentions-do.md)