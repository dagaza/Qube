# Audit session privacy — what left my machine

## Common questions

- How do I see what Qube sent off my computer this session?
- Where is the session egress or privacy summary?
- What is the Web discovery card on Telemetry?
- How do I audit web search without reading JSONL?
- What is coming in Team for privacy reports?

## What you can audit today (free)

Qube does **not** yet ship a full **exportable session egress report** for every adapter domain (planned for **Team** — one-click privacy report). You **can** audit a session with built-in, on-device surfaces:

| Question | Where to look |
|----------|----------------|
| Which **search tier** and **SERP provider** are active? | **Settings → Privacy & data** · **Telemetry → Web discovery** (shortcut on Privacy page) |
| Did **DuckDuckGo budgets** or **backoff** block live search? | **Settings → Knowledge → Web search discovery** (**Live DDG usage**) · **Telemetry → Web discovery** |
| Which **MCP / integration capabilities** ran this chat session? | **Settings → Privacy & data → Open Telemetry → Session integrations** · **Telemetry → Session integrations** (open the conversation first) |
| What did **one reply** retrieve? | **Sources → INSPECT RETRIEVAL** on that assistant message |
| Session-wide **route mix**? | **Telemetry → Router Intelligence** |
| Detailed **JSONL** for a turn? | Enable **Routing debug log** or **Web search log** under **Settings → Privacy & data** (see [Log redaction](log-redaction-sharing-logs.md) before sharing) |

All of the above are **local** — Qube does not upload a privacy dashboard to Qube servers.

## Telemetry → Web discovery (R10)

The **Web discovery** card on **Advanced Telemetry** refreshes about **once per second** while the page is open:

| Metric | Meaning |
|--------|---------|
| **Privacy tier** | Active SERP discovery tier (same as Knowledge settings) |
| **Primary provider** | Current primary SERP route; shows DDG backoff text when paused |
| **DDG burst / session budget** | Live DDG HTTP call counters (cache hits excluded) |
| **Pacing** | Minimum gap between live DDG queries; doubles in conservative mode |
| **System health** | Stable vs budget exhausted vs backoff vs conservative pacing |

For how to read the rest of the dashboard, see [Advanced Telemetry — interpreting](advanced-telemetry-interpreting.md).

## What leaves the device (web)

**Web discovery privacy tiers** control **SERP discovery** (finding links). **Page fetches** after a URL is chosen go **directly from your machine to the destination site** — not through Qube cloud.

See the tier table in [Web discovery privacy tiers](web-discovery-privacy-tiers.md). **Live Sources adapters** (PubMed, APIs you configure, etc.) follow each provider’s endpoint when that adapter runs — INSPECT and routing logs name adapters per turn. **MCP integration capabilities** run locally via configured MCP servers; **Telemetry → Session integrations** summarizes calls for the active chat session.

## Step-by-step session review

1. Open **Settings → Privacy & data → Session audit** — jump to **Telemetry → Web discovery** or **Session integrations**.
2. Open **Telemetry → Web discovery** — confirm tier, budgets, and health.
3. Open **Telemetry → Session integrations** — review MCP/integration calls for the current conversation.
3. Skim **Router Intelligence** for unexpected **WEB** / **HYBRID** volume.
4. For any reply you care about: **Sources → INSPECT RETRIEVAL** — check adapters, integration steps, discovery policy lines, and routing block.
5. Optional: enable **Web search log** recording, reproduce one web turn, **View Web search log** — then disable recording again.
6. Before sharing logs: [Log redaction before sharing logs](log-redaction-sharing-logs.md).

## What Team will add later

**Team SKU** epics (not required for home use):

- **Exportable session egress report** — consolidated adapters/domains and integration calls beyond the in-app **Session integrations** panel
- **One-click privacy report export** — JSON/PDF bundle for IT review
- **Org policy profiles** — enforced privacy tier and redaction defaults

Home users get **Telemetry + INSPECT + Settings** transparency today; Team adds **exportable governance**.

## Related

- [Web discovery privacy tiers](web-discovery-privacy-tiers.md) — tier comparison table
- [INSPECT RETRIEVAL](inspect-retrieval.md) — per-reply forensics
- [Integrations settings](../features/settings/integrations.md) — MCP capabilities and permissions
- [Privacy & data settings](../features/settings/privacy-data.md) — tier, session audit shortcuts, audit logs
- [Advanced Telemetry](../features/telemetry.md) — full dashboard catalog
- [Diagnostic logs](diagnostic-logs-advanced-settings.md) — JSONL detail
- [Delete memory entries](delete-memory-entries.md) — local memory data you control
