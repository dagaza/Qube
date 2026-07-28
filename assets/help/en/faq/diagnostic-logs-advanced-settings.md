# Diagnostic logs — Settings → Advanced

## Common questions

- Where does Qube store log files?
- What is the difference between **Advanced Telemetry** and **diagnostic logs**?
- Which log should I enable for routing vs LLM vs web search issues?
- What does the **Application log** contain?
- What does the **LLM debug log** contain?
- Does turning off file logging stop terminal output?
- How do I view or clear logs safely?

## What diagnostic logs are

**Settings → Advanced → Diagnostic logs** exposes **five rotating log files** under your Qube data directory. Each log has:

- A **recording toggle** (when Qube appends new lines to disk)
- **View {log name}** — in-app viewer (last **500 lines**, optional **Live tail** every 2s, **Open externally**)
- **Clear log** — truncates the file and deletes rotated backups (`*.log.1`, …)

Use **Open logs folder** to reveal the directory in your file manager.

These files are **persistent troubleshooting traces**. They complement — but do not replace — the live **[Advanced Telemetry](../features/telemetry.md)** dashboard and per-turn UI affordances (**INSPECT RETRIEVAL**, chat **TTFT** / **TPS** labels).

## Where files live

| Platform | Logs directory |
|----------|----------------|
| Linux / macOS | `~/.qube/logs/` |
| Windows | `%LOCALAPPDATA%\Qube\logs\` (falls back to `~/.qube/logs/` if unset) |

All paths below are relative to that folder.

## Telemetry vs diagnostic logs vs Knowledge diagnostics

| Surface | Where | What it is | Persistence |
|---------|-------|------------|-------------|
| **Advanced Telemetry** | Left nav **Telemetry** | Live CPU/RAM/GPU chart, **Pipeline Latency**, router/sidecar cards | In-memory rolling windows; cleared when Qube exits |
| **Diagnostic log files** | **Settings → Advanced** | Subsystem trace files you enable, view, and clear | Rotating files on disk |
| **Last retrieval trace** | **Settings → Knowledge → Diagnostics** | Summary panel fed from `web_search.log` **`retrieval_trace`** JSONL events | Requires **Web search log** recording (same file) |
| **LLM debug log panel** | **Telemetry** (developer only) | Live tail when launched with `QUBE_LLM_LOG_UI=1` | Same underlying `llm_debug.log` |

For slow replies, start with [Advanced Telemetry — interpreting](../faq/advanced-telemetry-interpreting.md), then enable the relevant log below and reproduce once.

## The five logs (what each records)

Default **recording** states reflect fresh installs (Settings toggle unless a launch env override is set).

| Log | File | Default recording | Logger / purpose |
|-----|------|-------------------|------------------|
| **Application log** | `qube.log` | **On** | General `Qube.*` runtime: boot, workers, voice, model load, ingestion errors (**INFO** in file; use `QUBE_APP_LOG_LEVEL=DEBUG` for verbose capture). **Excludes** the four dedicated debug loggers below so they are not duplicated here. |
| **LLM debug log** | `llm_debug.log` | **On** | `Qube.NativeLLM.Debug` — structured JSON events (discourse, validation, completion traces, router one-liners, etc.). **Heavy native prompt reconstruction** additionally requires `QUBE_LLM_DEBUG=1` at launch (observer-only; does not change inference). The Settings toggle controls **file recording only** — introspection may still run when `QUBE_LLM_DEBUG` is on. |
| **Routing debug log** | `routing_debug.log` | **Off** | `Qube.RoutingDebug` — one compact **JSONL line per chat turn**: route, strategy, intent scores, retrieval outcome blocks, policy trace. Enable recording, send a message, then **View** or grep the file. |
| **Web search log** | `web_search.log` | **Off** | `Qube.WebSearchAudit` — DuckDuckGo SERP audit: trigger reason, query text, result URLs, relevance-gate outcomes, and optional **`retrieval_trace`** events (powers **Knowledge → Diagnostics** refresh). Snippets only — result pages are not fetched. |
| **Skills debug log** | `skills_debug.log` | **Off** | `Qube.SkillsDebug` — per-turn skill activation scores and prompt injection telemetry. Requires **Skills** enabled under **Settings → AI & Models**; with Skills off, no activation telemetry is produced regardless of this toggle. |

Rotating limits (approximate): **Application**, **LLM**, and **Routing** logs — 10 MB × 5 backups; **Web search** and **Skills** — 5 MB × 3 backups.

## Terminal vs file

- **Dedicated debug loggers** (`NativeLLM.Debug`, `RoutingDebug`, `SkillsDebug`, `WebSearchAudit`) are routed to their files and kept **quiet on the terminal**.
- The **Application log** file captures other `Qube.*` modules when recording is on. **Turning file recording off does not change terminal output** — you still see logs in the console when running from a terminal.
- Disabling `QUBE_APP_LOG=0` at launch prevents attaching the application file sink entirely (terminal unchanged).

## Launch environment overrides

When set at process start, these **override** the in-app recording toggles (the toggle is disabled and shows “launch setting”):

| Variable | Affects | Values |
|----------|---------|--------|
| `QUBE_APP_LOG` | Application log file sink | `0` / false = off; default on |
| `QUBE_APP_LOG_LEVEL` | Application log verbosity | `DEBUG`, `INFO`, … (default `INFO`) |
| `QUBE_APP_LOG_FILE` | Application log path | Custom path |
| `QUBE_LLM_DEBUG_LOG` | LLM debug **file** recording | truthy / `0` |
| `QUBE_LLM_DEBUG` | Native LLM introspection (prompt dumps, etc.) | `1` — separate from file toggle |
| `QUBE_LLM_DEBUG_FILE` | Optional extra prompt dump path | File path |
| `QUBE_ROUTING_DEBUG_LOG` | Routing JSONL recording | truthy / `0` |
| `QUBE_ROUTING_DEBUG_LOG_VERBOSE` | Richer routing JSONL | `1` |
| `QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY` | Hash user query in routing log | `1` |
| `QUBE_WEB_SEARCH_AUDIT_LOG` | Web search JSONL recording | truthy / `0` |
| `QUBE_WEB_SEARCH_AUDIT_REDACT` | Hash queries; omit snippet bodies in file | `1` |
| `QUBE_LOG_RAW_COMPLETION` | Raw completion fields in LLM debug log | `1` |
| `QUBE_LLM_LOG_UI` | Show LLM debug tail panel on **Telemetry** page | `1` |

Skills debug recording is controlled only via Settings (no launch override).

## Privacy before sharing

- **Web search log** and **routing debug log** can contain **full query text** unless you launch with redact flags (`QUBE_WEB_SEARCH_AUDIT_REDACT=1`, `QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY=1`).
- **LLM debug log** may contain **prompt excerpts**, retrieved context markers, and completion traces — review before attaching to feedback.
- Use **Clear log** between reproduction runs so excerpts are easy to find; clear again before sharing if old sessions remain.

See [Log redaction before sharing logs](log-redaction-sharing-logs.md) for the full workflow and launch examples.

## Workflow for bug reports

1. Reproduce with the **smallest** log set (often **Application** + one subsystem log).
2. **Clear** relevant logs, reproduce **once**, open **View {log name}** (or **Open logs folder**).
3. Copy the tail around the failure timestamp — not the entire multi-megabyte file.
4. Submit via **Settings → Contact & Feedback** with excerpts (see [Contact & Feedback](../features/settings/contact-feedback.md)).

## Which log should I enable?

| Symptom | Start here |
|---------|------------|
| Crash, worker error, model load failure, voice/STT | **Application log** (`QUBE_APP_LOG_LEVEL=DEBUG` if needed) |
| Wrong prompt, missing context in Internal Engine, token/stop issues | **LLM debug log** + `QUBE_LLM_DEBUG=1` for full native dumps |
| Wrong route (MEMORY/RAG/WEB/HYBRID), retrieval phase oddities | **Routing debug log** — send one chat turn after enabling. Interpret routes with [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) |
| Web search misfires, empty SERP, relevance gate | **Web search log** — trigger a search after enabling |
| Skill not activating or wrong skill injected | **Skills debug log** — confirm Skills enabled, send one turn |
| Library / external knowledge pipeline detail | **Web search log** ( **`retrieval_trace`** lines) + **Knowledge → Diagnostics** refresh |

## Related

- [Advanced settings](../features/settings/advanced.md) — JSON editor and log controls
- [Delete memory entries](delete-memory-entries.md) — local data you control vs logs you may share
- [Web discovery privacy tiers](web-discovery-privacy-tiers.md) — what web discovery sends off-device
- [Log redaction before sharing logs](log-redaction-sharing-logs.md) — redaction env vars and excerpt workflow
- [Audit session privacy](audit-session-privacy.md) — Telemetry and INSPECT session review
- [Advanced Telemetry](../features/telemetry.md) · [Interpreting telemetry](advanced-telemetry-interpreting.md)
- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — route vocabulary before reading JSONL — live dashboard
- [Knowledge settings](../features/settings/knowledge.md) — retrieval trace panel and knowledge pack import/export
- [Contact & Feedback settings](../features/settings/contact-feedback.md) — where to send excerpts
