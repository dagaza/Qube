# Diagnostics

## Common questions

- Where are application debug logs?
- How do I open the logs folder?
- Which log shows worker or model load errors?
- Where did the LLM / routing logs go?

## What it is

**Diagnostics** settings expose **technical troubleshooting logs** under `~/.qube/logs/`. Privacy-sensitive audit logs (LLM, routing, web search) moved to **Settings → Privacy & data**.

| Log | File | Default recording | Use when |
|-----|------|-------------------|----------|
| Application | `qube.log` | On | General runtime errors, workers, model load |
| Skills debug | `skills_debug.log` | Off | Skill activation (Skills must be on in AI & Models) |

Use **Open logs folder** to reveal the directory in your file manager. Each log supports **recording toggle**, **View**, and **Clear**.

## Where to find it

Open **Settings → Diagnostics** (settings section `diagnostics`). Press **?** for the guided tour (`settings.diagnostics`).

Legacy deep links to **Settings → Advanced → Diagnostic logs** redirect here.

## Also called

diagnostic logs, debug logs, application log, troubleshooting logs

## How to…

1. **Open logs folder** — Reveal `~/.qube/logs/` on disk.
2. **Enable application logging** — Keep **Application log** recording on for support requests.
3. **Debug skills** — Enable **Skills debug log**, send a chat message, then **View Skills debug log**.
4. **Clear between runs** — Click **Clear log** so new entries are easy to read.

## Controls

<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->
Controls listed top-to-bottom for **Settings → Diagnostics**.


### Diagnostic logs

- **Clear log**

### Application log

- **Record application events to this log**
- **View Application log**
- **Clear log**

### Skills debug log

- **Record skill activation to this log**
- **View Skills debug log**
- **Clear log**

## Related

- [Diagnostic logs FAQ](../../faq/diagnostic-logs-advanced-settings.md)
- [Privacy & data settings](privacy-data.md)
- [Contact & Feedback settings](contact-feedback.md) — attach log excerpts to reports
- [Model won't load troubleshooting](../../troubleshooting/model-wont-load.md)
