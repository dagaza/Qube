# Advanced

## Common questions

- Where is the raw JSON settings editor?
- How do I view diagnostic logs?
- Which log file shows routing vs LLM vs web search?
- Is it safe to edit settings JSON directly?
- Where do I import or export a knowledge pack?

## What it is

**Advanced** settings expose power-user tools: a **JSON settings** editor for direct preference edits and **diagnostic logs** for troubleshooting workers, inference, and ingestion. Changes here can affect stability—prefer regular settings pages unless you know the key names.

**Export knowledge pack** / **Import knowledge pack** live under **Settings → Knowledge → Diagnostics**, not on this page.

**Diagnostic logs** are rotating files under `~/.qube/logs/` (Windows: `%LOCALAPPDATA%\Qube\logs\`). They are **not** the same as the live **Telemetry** dashboard — see [Diagnostic logs FAQ](../../faq/diagnostic-logs-advanced-settings.md) for what each file contains, default recording states, privacy, and launch env overrides.

| Log | File | Default recording | Use when |
|-----|------|-------------------|----------|
| Application | `qube.log` | On | General runtime errors, workers, model load |
| LLM debug | `llm_debug.log` | On | Prompt/completion traces (`QUBE_LLM_DEBUG=1` for heavy native dumps) |
| Routing debug | `routing_debug.log` | Off | Per-turn route / retrieval JSONL |
| Web search | `web_search.log` | Off | SERP audit + `retrieval_trace` (Knowledge diagnostics panel) |
| Skills debug | `skills_debug.log` | Off | Skill activation (Skills must be on in AI & Models) |

Each log supports **recording toggle**, **View** (500-line in-app tail, live refresh), and **Clear** (truncates file and rotated backups). Turning off file recording does **not** silence the terminal.

## Where to find it

Open **Settings → Advanced** (settings section `advanced`). Press **?** for the guided tour (`settings.advanced`).

## Also called

JSON settings, advanced preferences, diagnostic logs, debug logs, JSON SETTINGS, power user settings

## How to…

1. **Inspect JSON carefully** — Click **Edit settings.json**, locate the key you need, and validate syntax before saving.
2. **Back up settings manually** — Copy `settings.json` or note values before risky edits (there is no export button on this page).
3. **Open log files on disk** — Use **Open logs folder** under **Diagnostic logs**.
4. **Enable a diagnostic log** — Turn on recording for the subsystem you are debugging (application, LLM, routing, web search, skills).
5. **Reproduce once** — Perform the failing action, then click **View {log name}** and attach excerpts to feedback if needed.
6. **Clear after testing** — Click **Clear log** between runs so new entries are easy to read.

## Controls

<!-- include:generated/controls/advanced.md -->

## Related

- [Diagnostic logs FAQ](../../faq/diagnostic-logs-advanced-settings.md) — educative guide to all five logs
- [Advanced Telemetry](../../features/telemetry.md) · [Interpreting telemetry](../../faq/advanced-telemetry-interpreting.md) — live dashboard (not log files)
- [Contact & Feedback settings](contact-feedback.md) — send logs with bug reports
- [Knowledge settings](knowledge.md) — knowledge pack import/export and retrieval trace under Diagnostics
- [Model won't load troubleshooting](../../troubleshooting/model-wont-load.md) — when logs show VRAM or load errors
- [Search models not ready troubleshooting](../../troubleshooting/search-models-not-ready.md) — embedding worker issues
