# Create a knowledge preset

## Common questions

- What is a knowledge preset?
- How do I bundle Live Sources for one project?
- Can a preset search my Library folders?

## What it is

A **knowledge preset** defines a custom composer tool (for example **`@[tool:user:biology]`**) backed by selected **API adapters** or **Web fetch** domains—not Library folders. Use presets when you want repeatable scoped research without retyping adapter ids each session.

Library document search remains separate—use **`@[tool:library]`** or **Local Knowledge Base** for ingested files.

## Where to find it

Create and edit presets in **Settings → Knowledge → My knowledge**. See [Knowledge settings](../features/settings/knowledge.md).

## Also called

research preset, source bundle, scoped knowledge, custom composer tool, saved source profile

## How to…

1. Open **Settings → Knowledge → My knowledge**.
2. Enter a **Preset id** and **Display label**, choose **API adapters (scientific, finance, legal)** or **Web fetch (source profile)**, and list source adapter ids or trusted domains.
3. Click **Save preset**. Use **Explain selected** or **Delete selected** to manage saved rows.
4. Note the preset’s **`@[tool:user:…]`** token for chat.
5. Test with your preset token (or a built-in Live Source tool) to confirm the scope feels right.

**Can a preset mix Library and Live Sources?** No—presets bundle API adapters or web-fetch domains only. Combine **`@[tool:library]`** with a preset tool in the same message when you need both.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — My knowledge UI and Live Sources
- [Export or import knowledge pack](export-or-import-knowledge-pack.md) — share preset bundles
- [Live sources overview reference](../reference/live-sources-overview.md) — adapter catalog
