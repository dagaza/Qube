# Log redaction before sharing logs

## Common questions

- How do I share Qube logs without exposing my queries?
- What env vars redact web search or routing logs?
- Which diagnostic logs contain prompt text?
- Should I clear logs before sending a bug report?
- Where is log redaction documented in Settings?

## What it is

Qube diagnostic logs are **local files on your machine**. When you attach excerpts to feedback or post them in a forum, **review or redact first** — several logs can contain **full user queries**, **retrieved context markers**, or **completion traces**.

This page is the **default redaction guide** for **Settings → Advanced → Diagnostic logs**. It complements [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md) (what each log records) and [Web discovery privacy tiers](web-discovery-privacy-tiers.md) (what leaves the device during normal use).

## Where to find controls

| Surface | Path |
|---------|------|
| **Recording toggles** | **Settings → Advanced → Diagnostic logs** |
| **Launch overrides** | Set env vars **before** starting Qube (toggle shows “launch setting”) |
| **Logs folder** | **Open logs folder** on the same page |

## Logs that may contain sensitive text

| Log | Risk | Redaction option |
|-----|------|------------------|
| **Web search log** | Full query text, result URLs, snippets | Launch with `QUBE_WEB_SEARCH_AUDIT_REDACT=1` |
| **Routing debug log** | User query in JSONL per turn | Launch with `QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY=1` |
| **LLM debug log** | Prompt excerpts, completion traces | Disable recording; avoid `QUBE_LOG_RAW_COMPLETION=1` unless needed |
| **Application log** | General runtime; usually less query detail | Lower verbosity; clear before repro |
| **Skills debug log** | Skill scores tied to turn context | Enable only for skill debugging |

Redaction env vars **hash or omit** fields in the **file** — they do not change what Qube sends to search providers during chat.

## Recommended workflow before sharing

1. **Use the smallest log set** — often **Application log** plus one subsystem log.
2. **Enable redaction at launch** when you know you will share routing or web search logs:

   ```bash
   QUBE_WEB_SEARCH_AUDIT_REDACT=1 \
   QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY=1 \
   ./Qube
   ```

3. **Clear** relevant logs (**Clear log** on each row), reproduce **once**, copy only the tail around the failure timestamp.
4. **Skim excerpts** for names, file paths, API keys in URLs, or pasted secrets — remove manually even when redaction is on.
5. Submit via **Settings → Contact & Feedback** with excerpts, not whole multi-megabyte files.

## When redaction is not enough

- **LLM debug log** with `QUBE_LLM_DEBUG=1` may still contain large prompt reconstructions — prefer routing or web search logs with redaction flags for routing/SERP issues.
- **Screenshots** of **INSPECT RETRIEVAL** or **Telemetry** may show query-derived labels — crop or describe in prose instead.
- **Memory** and **Library** content is separate from logs — see [Delete memory entries](delete-memory-entries.md).

## Org / Team note

Enforced org-wide redaction presets are planned for **Team policy profiles** (Phase 3). Today, redaction is **user-controlled** via launch env vars and careful excerpting.

## Related

- [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md) — all five logs and env overrides
- [Audit session privacy](audit-session-privacy.md) — what to check before a session review
- [Web discovery privacy tiers](web-discovery-privacy-tiers.md) — SERP egress by tier
- [Advanced settings](../features/settings/advanced.md) — JSON editor and log controls
- [Contact & Feedback settings](../features/settings/contact-feedback.md) — where to send excerpts
