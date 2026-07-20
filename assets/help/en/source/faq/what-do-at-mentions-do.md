# What do @ mentions do

## Common questions

- What is the difference between `@` tools, skills, and commands?
- How do I search my Library with `@`?
- Which `@` attachment controls routing?

## What it is

Composer **`@` mentions** attach capabilities to a chat message. **Tools** (for example **`@[tool:library]`**, **`@[tool:internet]`**) route retrieval and behaviour—only the **first routing attachment** (`@[file:…]`, `@[chat:…]`, or `@[tool:…]`) in a message controls routing. Put the attachment you want first. **Skills** add reasoning frameworks to the system prompt without changing routing. **Commands** run immediate app actions (such as resetting tours) and are not sent to the model.

Pick a tool when you need evidence from Library, web discovery, or Live Sources; pick a skill when you want structured analysis on whatever context is already present. **Skills never change the cognitive route** — only tools and the first routing attachment do. Full pathway guide: [Cognitive Router — how routing works](../faq/cognitive-router-how-routing-works.md).

## Where to find it

Insert mentions from the **`@` palette** in Conversations or read the full lists in **Library → Qube** (or **Settings → Help → Open Qube documentation**). Open **Settings → Help → Open @ Composer Guide** for the interactive guide.

## Also called

at mentions, composer attachments, @ tools, @ skills, @ commands, tool tokens

## How to…

1. Type **`@`** in the composer to open the palette.
2. Choose a **tool** when you need Library, internet/web discovery, Live Sources, or another routing target.
3. Add a **skill** alongside a tool when you want a structured reasoning frame (research synthesis, etc.).
4. Use **commands** for app actions—not for model prompts.
5. Consult generated reference pages in **Library → Qube** for the authoritative token list on your version.

## Related

- [Composer attachments reference](../reference/composer-attachments.md) — `@[file:…]`, `@[chat:…]`, routing order
- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — pre/post overrides and route vocabulary
- [Composer tools reference](../reference/composer-tools.md) — every `@[tool:…]` token
- [Composer skills reference](../reference/composer-skills.md) — `@[skill:…]` tokens
- [Composer commands reference](../reference/composer-commands.md) — immediate actions
- [Help settings](../features/settings/help.md) — composer guide entry point
