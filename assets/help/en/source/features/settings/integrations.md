# Integrations and MCP capabilities

## Common questions

- How do I connect an MCP server to Qube?
- Where do I review permissions for GitHub or filesystem MCP tools?
- What is `@[cap:mcp:…]` in the composer?
- Why is an integration capability locked in the `@` palette?

## What it is

**Integrations** turn MCP servers into **capabilities** — normalized, permissioned actions you attach in chat with **`@[cap:mcp:namespace/action]`**. Qube does not expose raw MCP tool names in the main UI; you grant capabilities by tier (**read**, **write**, **destructive**).

MCP servers are configured under **Settings → Knowledge → Custom sources** (connector type **mcp**). After **Save** or **Test**, Qube discovers capabilities and opens a **grant review** dialog. Ongoing permission toggles live under **Settings → Integrations**.

Integration capabilities are **opt-in** and **attachment-driven** — the model does not gain MCP tools silently on connect. Attach a capability (or a My knowledge preset that bundles capability URNs) for the turn where you want it to run.

## Where to find it

| Task | Location |
|------|----------|
| Add / edit MCP server | **Settings → Knowledge → Custom sources** |
| Server health summary | **Settings → Integrations → MCP servers** |
| Grant or deny capabilities | **Settings → Integrations → Capability permissions** |
| Attach in chat | Composer `@` → **Integrations** section |
| Session integration calls | **Telemetry → Session integrations** (current chat session) |

Open **Settings → Integrations** (settings section `integrations`). Use **Manage in Knowledge → Custom sources** when you need to edit server commands or namespaces.

## First connect

When a server connects for the first time, Qube shows a **grant review** dialog:

- **Read** capabilities are suggested on by default.
- **Write** and **destructive** capabilities stay off until you enable them.
- **Suggested presets** can be saved to **My knowledge** (bundles capability URNs, not raw tools).

## Drift and re-review

When a server's capabilities change (new tools, schema updates, tier changes), Qube detects **drift** and prompts you to **re-review**. Existing grants remain valid only when the capability fingerprint is unchanged; escalated tiers require explicit re-consent.

## Inspect and audit

- Per reply: **Sources → INSPECT RETRIEVAL** shows attachment → MCP invoke → rank → cite steps when an integration capability ran.
- Per session: **Telemetry → Session integrations** lists integration calls for the open conversation (capability group and tier; raw tool names when **Advanced** settings are unlocked).

## Knowledge packs

Exporting a knowledge pack includes **integration consent grants** (not secrets). Import merges grants so permission decisions transfer with presets and custom sources. See [Export or import a knowledge pack](../../workflows/export-or-import-knowledge-pack.md).

## Also called

MCP integrations, MCP capabilities, capability permissions, MCP server registry, INTEGRATIONS settings, cap tokens

## How to…

1. **Connect a server** — Follow [Connect an MCP server](../../workflows/connect-mcp-server.md) (filesystem example).
2. **Review permissions** — Open **Settings → Integrations → Capability permissions** and toggle each capability. Locked rows need grant review or a configured MCP source first.
3. **Attach in chat** — Type `@`, open **Integrations**, or search for a namespace (for example `filesystem`). Insert **`@[cap:mcp:…]`** on the message.
4. **Verify a turn** — Send the message, open **Sources** on the reply, then **INSPECT RETRIEVAL** for capability steps.
5. **Back up grants** — Export a knowledge pack under **Settings → Knowledge → Diagnostics**.

## Related

- [Connect an MCP server workflow](../../workflows/connect-mcp-server.md) — step-by-step filesystem MCP setup
- [Export or import a knowledge pack](../../workflows/export-or-import-knowledge-pack.md) — transfer consent grants
- [Composer attachments](../../reference/composer-attachments.md) — `@[cap:…]` vs `@[tool:…]`
- [Knowledge settings](knowledge.md) — Custom sources MCP connector fields
- [Create a knowledge preset](../../workflows/create-knowledge-preset.md) — presets with capability bundles
- [INSPECT RETRIEVAL FAQ](../../faq/inspect-retrieval.md) — per-reply capability trace
- [Audit session privacy FAQ](../../faq/audit-session-privacy.md) — Telemetry session review
