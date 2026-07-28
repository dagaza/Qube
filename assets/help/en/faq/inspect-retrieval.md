# INSPECT RETRIEVAL — per-reply retrieval forensics

## Common questions

- What is **INSPECT RETRIEVAL**?
- Where is the inspect button on an assistant reply?
- What do the Summary, Graph, Compare, and Explain tabs show?
- How do I see why Qube routed a turn to Memory, Library, or web?
- How is INSPECT different from **Telemetry → Router Intelligence**?

## What it is

**INSPECT RETRIEVAL** opens the **Retrieval Inspector** for **one assistant message** — a read-only view of how Qube retrieved evidence for that reply: adapters, preset, pipeline graph, fetch provenance, and (when available) **routing** for the turn.

It complements session-wide **Telemetry → Router Intelligence** and the **Routing debug log** (Settings → Advanced).

## Where to find it

1. In **Conversations**, open an assistant reply that used retrieved sources.
2. Click **Sources** (citation list).
3. Click **INSPECT RETRIEVAL** when the button is shown (requires a stored retrieval bundle for that message).

The inspector also reflects the latest **Knowledge → Diagnostics** retrieval trace when no bundle is stored but a trace exists.

## Also called

retrieval inspector, inspect retrieval button, per-turn retrieval trace, retrieval forensics, sources inspect

## Inspector tabs

| Tab | Shows |
|-----|--------|
| **Summary** | Service, strategy, adapters, coverage, latency, search outcome, **routing (this turn)** when the routing buffer has data |
| **Graph** | Pipeline graph for adapter phases |
| **Compare** | Replay compare when a stored retrieval record and database are available |
| **Explain** | Fetch provenance, discovery policy, preset explain view, or search-outcome detail |

## Routing explainability (Summary)

When Qube still has the latest turn in the in-memory routing buffer, **Summary** includes a **Routing (this turn)** block:

- Initial vs final route (for example RAG → NONE after empty-source downgrade)
- Memory / Library / web hit counts
- Empty-source downgrade notice when retrieval ran but nothing survived relevance gates

For the full JSONL record (intent scores, policy trace, sidecar flags), enable **Routing debug log** under **Settings → Advanced**, send one message, then **View Routing debug log**. See [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md).

## When the button is missing

| Situation | Why |
|-----------|-----|
| Plain chat with no retrieval | No bundle or trace to inspect |
| Sources empty after downgrade | Reply may be ungrounded even if routing attempted retrieval |
| Old session / restart | In-memory routing buffer cleared — use **Routing debug log** if recording was on |

See [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) for route vocabulary and empty-source downgrade.

## Related

- [Cognitive Router — how routing works](cognitive-router-how-routing-works.md) — NONE / MEMORY / RAG / WEB / HYBRID
- [Advanced Telemetry — interpreting](advanced-telemetry-interpreting.md) — session router card vs per-reply INSPECT
- [Retrieval profile vs search quality](retrieval-profile-vs-search-quality.md) — profile names in traces
- [Diagnostic logs — Advanced settings](diagnostic-logs-advanced-settings.md) — routing JSONL and web search audit
- [Conversations](../features/conversations.md) — Sources panel and timing labels
