# Under the hood — explain a chat turn with @diagnostics

## Common questions

- How do I debug one reply with **Under the hood**?
- What is **Under the hood** (`@[tool:diagnostics]`) in the composer?
- What is `@underhood` or `@under` in the `@` palette?
- How is this different from **INSPECT RETRIEVAL**?
- Can I ask **`@[tool:help]`** how this feature works?

## What it does

`@[tool:diagnostics]` (palette label **Under the hood**) builds a **Turn Diagnostics Bundle** from data Qube already recorded for a reply: routing summary, retrieval metadata, sources summary, optional log excerpts, and generation metadata (including how sources were cited when present). Your loaded model then explains what happened in plain language, separating **facts recorded in the bundle** from **hypotheses**.

**INSPECT RETRIEVAL** is the deterministic, tabbed inspector on **Sources**. **Under the hood** is a composer tool that sends that turn’s forensics to the model for a narrated walkthrough—you can also attach it to a **specific** assistant message, not only the latest reply.

## How to use it

1. **Explain the last reply** — Type `@[tool:diagnostics]` or `@[tool:underhood]` in the composer (aliases: `underhood`, `underthehood`, `under`). The palette label is **Under the hood** — multi-word labels cannot be typed with spaces after `@`; pick **Under the hood** from **Tools** after typing `@under`, or insert the chip from the `@` menu.
2. **Explain a specific reply** — Right‑click an assistant message → **Explain under the hood…** (preloads the composer chip for that message).
3. **Ask in `@help`** — Attach **`@[tool:help]`** and ask e.g. “How does Under the hood work?” or “Explain turn diagnostics” — help searches this corpus (including this page).

For richer routing detail before you explain a turn, enable **Routing debug log** under **Settings → Privacy & data**. For more technical bundles (extra log excerpts), enable **Technical detail in turn diagnostics** under **Settings → Privacy & data**.

## Copy or export for support

After an explanation, use the message **⋮** menu to **Copy issue snippet** or **Export diagnostics JSON** when you need a redacted bundle for feedback or bug reports.

## vs INSPECT RETRIEVAL

| | **Under the hood** (`@[tool:diagnostics]`) | **INSPECT RETRIEVAL** |
|---|--------------------------------------------|------------------------|
| Entry | Composer tool or reply context menu | **Sources** on a reply |
| Output | Model narrative from a diagnostics bundle | Fixed UI tabs (Summary, Graph, …) |
| Best for | Narrated orchestration walkthrough in chat | Exact adapter/graph/trace inspection |

## Also called

diagnostics composer, under the hood, underhood, explain turn, turn forensics, orchestration explain, `@diagnostics`

## Related

- [INSPECT RETRIEVAL](inspect-retrieval.md) — deterministic per‑reply retrieval forensics
- [What do @ mentions do](what-do-at-mentions-do.md) — tools vs skills; `@help` for documentation
- [Composer tools](../reference/composer-tools.md) — full `@[tool:…]` list including **Under the hood**
- [Diagnostic logs](diagnostic-logs-advanced-settings.md) — application log files and routing JSONL
- [Diagnostics settings](../features/settings/diagnostics.md) — **Open logs folder**
- [Privacy & data settings](../features/settings/privacy-data.md) — routing debug log and technical detail toggle
- [Conversations](../features/conversations.md) — Sources and reply actions

## Search terms

@help under the hood, diagnostics composer, underhood, explain turn, routing debug, orchestration explain, turn diagnostics bundle