# Web discovery privacy tiers

## Common questions

- What is the **Privacy tier** for web search?
- What is the difference between **Private search** and **Private + API fallback**?
- Where do `@internet` and Hybrid Internet Mode send my queries?
- How do I use self-hosted **SearXNG** with Qube?
- Does Qube log my search queries on Qube servers?

## What it is

**Web discovery privacy tiers** control which **search providers** Qube may use when discovering URLs for live web turns — `@internet`, `@research`, Hybrid Internet Mode, and related discovery paths.

Tiers affect **SERP discovery** (finding links and snippets). After a URL is chosen, **page fetches go directly from your machine to the destination website** — Qube does not proxy page content through Qube servers.

**Default:** **Private search (recommended)** — DuckDuckGo HTML and Wikipedia only; no commercial search API keys required.

## Where to find it

| Surface | Path |
|---------|------|
| **Settings (persistent)** | **Settings → Knowledge → Web search discovery → Privacy tier** |
| **Conversations (quick switch)** | Tools panel → **Privacy** → tier selector (same setting; label shows the active tier) |

The Knowledge page also shows **What leaves this device** bullets that update with your tier. Review **Live DDG usage** counters before relying on automatic web routing.

## Also called

privacy tier, web search privacy, discovery privacy, private search tier, searxng tier, brave fallback, duckduckgo privacy settings

## The four tiers

| Tier (UI label) | SERP discovery | API fallbacks | Typical use |
|-----------------|----------------|---------------|-------------|
| **Private search (recommended)** | DuckDuckGo HTML + Wikipedia API | **None** — no Brave/Bing/commercial SERP APIs | Default; no third-party search API keys |
| **Private + API fallback** | DuckDuckGo + Wikipedia (primary) | **Brave Search API** when configured and primary path fails | More reliability while keeping private primary path |
| **Maximum reliability** | Same as balanced | Same Brave fallback chain; optional query alternation stays **off** by default | When DDG blocks are frequent and Brave is configured |
| **Self-hosted SearXNG** | Your **SearXNG** instance when URL is configured and reachable | Falls back like balanced when SearXNG is unavailable | Org or power-user self-hosted search |

**Site-biased queries** (for example `@recipe` flows with domain hints) may promote **Brave** to primary when the tier allows API fallback **and** a Brave API key is configured.

## What leaves your device (by tier)

| Data | Private | Balanced / Enhanced | SearXNG |
|------|---------|-------------------|---------|
| Query text to DuckDuckGo | Yes (HTML SERP) | Yes (primary) | When not using SearXNG primary |
| Query text to Wikipedia API | Yes | Yes | Yes (fallback chain) |
| Query text to Brave Search API | **No** | Yes **only as fallback** when configured | Yes when in fallback chain |
| Query text to your SearXNG server | No | No | **Yes** (your instance; upstream engines depend on your config) |
| Page fetch after URL pick | Direct to destination site | Direct to destination site | Direct to destination site |
| Qube cloud logging of queries | **No** | **No** | **No** |

Enable **Web search log** under **Settings → Advanced** only when debugging — it can record query text locally unless you launch with redact flags (see [Diagnostic logs](diagnostic-logs-advanced-settings.md)).

## How to choose a tier

1. Stay on **Private search** unless you need API fallback or SearXNG.
2. Configure **Brave Search API** under **Settings → Knowledge** before selecting **Private + API fallback** or **Maximum reliability**.
3. For **Self-hosted SearXNG**, open **Set up SearXNG…** under **Settings → Knowledge → Web search discovery** to detect a local instance, test JSON search, and apply the tier — or enter the base URL manually.
4. Pair any web tier with sensible **DuckDuckGo pacing** limits on the same page if you hit rate blocks.
5. Read [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) for how **Hybrid Internet Mode** and the **Web** toggle interact with discovery — tiers do not replace routing rules.

## Related

- [Knowledge settings](../features/settings/knowledge.md) — privacy tier, DDG pacing, SearXNG URL, retrieval profile
- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — Web vs Hybrid Internet Mode
- [Live sources vs Library search](live-sources-vs-library-search.md) — web discovery vs Library RAG
- [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md) — web search audit log and redaction
- [Audit session privacy](audit-session-privacy.md) — session privacy review
- [Log redaction before sharing logs](log-redaction-sharing-logs.md) — redaction before sharing logs
