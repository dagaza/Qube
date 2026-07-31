# Integrations and MCP capabilities

## Common questions

- How do I connect an MCP server to Qube?
- Where do I review permissions for GitHub or filesystem MCP tools?
- What is `@[cap:mcp:…]` in the composer?

## What it is

**Integrations** turn MCP servers into **capabilities** — normalized, permissioned actions you attach in chat with `@[cap:mcp:namespace/action]`. Qube does not expose raw MCP tool names in the main UI; you grant capabilities by tier (read, write, destructive).

MCP servers are configured under **Settings → Knowledge → Custom sources** (connector type **mcp**). After save or test, Qube discovers capabilities and opens a **grant review** dialog. Ongoing permission toggles live under **Settings → Integrations**.

## Where to find it

| Task | Location |
|------|----------|
| Add / edit MCP server | **Settings → Knowledge → Custom sources** |
| Server health summary | **Settings → Integrations → MCP servers** |
| Grant or deny capabilities | **Settings → Integrations → Capability permissions** |
| Attach in chat | Composer `@` → **Integrations** section |

## First connect

When a server connects for the first time, Qube shows a grant review dialog:

- **Read** capabilities are suggested on by default.
- **Write** and **destructive** capabilities stay off until you enable them.
- **Suggested presets** can be saved to My knowledge (bundles capability URNs, not raw tools).

## Drift and re-review

When a server’s capabilities change (new tools, schema updates, tier changes), Qube detects **drift** and prompts you to re-review. Existing grants remain valid only when the capability fingerprint unchanged; escalated tiers require explicit re-consent.

## Knowledge packs

Exporting a knowledge pack includes **integration consent grants** (not secrets). Import merges grants so permission decisions transfer with presets and custom sources.

## Related

- [Export or import a knowledge pack](../workflows/export-or-import-knowledge-pack.md)
- [Composer attachments](../reference/composer-attachments.md)
- [Knowledge settings](knowledge.md)
