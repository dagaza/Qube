# Privacy & data

## Common questions

- Where do I set the web discovery privacy tier?
- How do I enable Hybrid Internet Mode outside Conversations?
- Which log records web searches or routing queries?
- How do I redact logs before sharing?
- Where do I see integration calls for this session?

## What it is

**Privacy & data** is the home for **web discovery privacy**, **Hybrid Internet Mode**, **session audit shortcuts**, and **audit logs** that may contain queries, prompts, retrieval traces, or SERP snippets stored locally under `~/.qube/logs/`.

| Control | Purpose |
|---------|---------|
| **Session audit** | Jump to Telemetry → Web discovery or Session integrations for the active conversation |
| **Privacy tier** | Same setting as Knowledge → Web search discovery — balances privacy vs optional API fallbacks |
| **What leaves your device** | Plain-language summary of outbound discovery traffic for the current tier |
| **Hybrid Internet Mode** | Same setting as the Conversations tools panel — auto web routing when context warrants it |
| **Audit logs** | LLM, routing, and web search log files with recording, redaction, view, and clear |

Advanced discovery limits, provider setup, DDG usage, and SearXNG configuration remain under **Settings → Knowledge → Web search discovery** (link on this page).

## Where to find it

Open **Settings → Privacy & data** in the **System** sidebar group (settings section `privacy.data`). Press **?** for the guided tour (`settings.privacy_data`).

## Also called

privacy settings, audit logs, sensitive logs, data egress, web discovery privacy, web discovery privacy tier, web discovery privacy tier settings, privacy tier settings, session egress

## How to…

1. **Review session egress** — Use **Open Telemetry → Session integrations** while a conversation is active.
2. **Choose a privacy tier** — Select Private, Balanced, Enhanced, or SearXNG for `@internet` and Hybrid Internet Mode.
3. **Read what leaves your device** — Check the summary card after changing tier.
4. **Enable Hybrid Internet Mode** — Turn on auto web routing when you want Qube to decide per turn (pair with a sensible tier).
5. **Redact before sharing** — Enable redaction toggles on **Web search log** and **Routing debug log**, or use launch env vars for overrides.
6. **Review audit logs briefly** — Enable recording only while reproducing an issue, then **View** or **Clear**.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → Privacy & data**.


### Session audit

- **Open Telemetry → Web discovery**
- **Open Telemetry → Session integrations**

### Web discovery privacy


### LLM debug log

- **Record LLM debug output to this log**
- **View LLM debug log**
- **Clear log**

### Routing debug log

- **Record routing decisions to this log**
- **Hash user queries in this log**
- **View Routing debug log**
- **Clear log**

### Web search log

- **Record web searches to this log**
- **Hash queries and omit snippet bodies in this log**
- **View Web search log**
- **Clear log**

### Web discovery privacy

- **Privacy tier**
- **Open Knowledge → Web search discovery**
- **Hybrid Internet Mode**
- **What leaves your device**

## Related

- [Web discovery privacy tiers FAQ](../../faq/web-discovery-privacy-tiers.md)
- [Log redaction before sharing logs](../../faq/log-redaction-sharing-logs.md)
- [Knowledge settings](knowledge.md) — advanced web discovery controls
- [Diagnostics settings](diagnostics.md)
- [Audit session privacy FAQ](../../faq/audit-session-privacy.md)
