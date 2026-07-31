# Create a knowledge preset

## Common questions

- What is a knowledge preset?
- How do I bundle Live Sources for one project?
- Can a preset search my Library folders?

## What it is

A **knowledge preset** defines a custom composer tool (for example **`@[tool:user:biology]`**) backed by selected **API adapters**, **Web fetch** domains, and/or **integration capability URNs** — not Library folders. Use presets when you want repeatable scoped research without retyping adapter ids or `@[cap:mcp:…]` tokens each session.

Library document search remains separate—use **`@[tool:library]`** or **Local Knowledge Base** for ingested files.

## Where to find it

Create and edit presets in **Settings → Knowledge → My knowledge**. See [Knowledge settings](../features/settings/knowledge.md).

## Also called

research preset, source bundle, scoped knowledge, custom composer tool, saved source profile

## How to…

1. Open **Settings → Knowledge → My knowledge**.
2. Enter a **Preset id** and **Display label**, choose **API adapters (scientific, finance, legal)** or **Web fetch (source profile)**, and list source adapter ids or trusted domains.
3. Click **Save preset**. Use **Explain selected** or **Delete selected** to manage saved rows.
4. Note the preset’s **`@[tool:user:…]`** token for chat. When the preset bundles MCP capabilities, Qube resolves it to the underlying **`@[cap:mcp:…]`** URNs at invoke time.
5. Test with your preset token (or attach individual **`@[cap:mcp:…]`** capabilities) to confirm the scope feels right.

**Capability bundles from MCP grant review** — When you connect an MCP server, the **Grant review** dialog can save **Suggested presets** to **My knowledge**. Those presets store capability URNs (not raw MCP tool names). Attach **`@[tool:user:…]`** in chat the same way as adapter-only presets.

**Can a preset mix Library and Live Sources?** No—presets bundle API adapters, web-fetch domains, and/or integration capabilities. Combine **`@[tool:library]`** with a preset or **`@[cap:mcp:…]`** token in the same message when you need both.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — My knowledge UI and Live Sources
- [Integrations settings](../features/settings/integrations.md) — MCP capability permissions
- [Connect an MCP server](connect-mcp-server.md) — configure Custom sources MCP connector
- [Export or import knowledge pack](export-or-import-knowledge-pack.md) — share preset bundles
- [Live sources overview reference](../reference/live-sources-overview.md) — adapter catalog
