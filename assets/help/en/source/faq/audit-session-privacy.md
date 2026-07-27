# Audit session privacy — what left my machine

## Common questions

- How do I see what Qube sent off my computer this session?
- Where is the session egress or privacy summary?
- What is the Web discovery card on Telemetry?
- How do I audit web search without reading JSONL?
- What is coming in Team for privacy reports?

## What you can audit today (free)

Qube does **not** yet ship a full **session egress export** (planned for **Team** — per-domain adapter list and one-click privacy report). You **can** audit a session with built-in, on-device surfaces:

| Question | Where to look |
|----------|----------------|
| Which **search tier** and **SERP provider** are active? | **Settings → Knowledge → Web search discovery** · **Telemetry → Web discovery** |
| Did **DuckDuckGo budgets** or **backoff** block live search? | Same Settings page (**Live DDG usage**) · **Telemetry → Web discovery** |
| What did **one reply** retrieve? | **Sources → INSPECT RETRIEVAL** on that assistant message |
| Session-wide **route mix**? | **Telemetry → Router Intelligence** |
| Detailed **JSONL** for a turn? | Enable **Routing debug log** or **Web search log** under **Settings → Advanced** (see [Log redaction](log-redaction-sharing-logs.md) before sharing) |

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

See the tier table in [Web discovery privacy tiers](web-discovery-privacy-tiers.md). **Live Sources adapters** (PubMed, APIs you configure, etc.) follow each provider’s endpoint when that adapter runs — INSPECT and routing logs name adapters per turn.

## Step-by-step session review

1. Open **Telemetry → Web discovery** — confirm tier, budgets, and health.
2. Skim **Router Intelligence** for unexpected **WEB** / **HYBRID** volume.
3. For any reply you care about: **Sources → INSPECT RETRIEVAL** — check adapters, discovery policy lines, and routing block.
4. Optional: enable **Web search log** recording, reproduce one web turn, **View Web search log** — then disable recording again.
5. Before sharing logs: [Log redaction before sharing logs](log-redaction-sharing-logs.md).

## What Team will add later

**Team SKU** epics (not required for home use):

- **Session egress summary** — consolidated adapters/domains contacted in a session
- **One-click privacy report export** — JSON/PDF bundle for IT review
- **Org policy profiles** — enforced privacy tier and redaction defaults

Home users get **Telemetry + INSPECT + Settings** transparency today; Team adds **exportable governance**.

## Related

- [Web discovery privacy tiers](web-discovery-privacy-tiers.md) — tier comparison table
- [INSPECT RETRIEVAL](inspect-retrieval.md) — per-reply forensics
- [Advanced Telemetry](../features/telemetry.md) — full dashboard catalog
- [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md) — JSONL detail
- [Delete memory entries](delete-memory-entries.md) — local memory data you control
