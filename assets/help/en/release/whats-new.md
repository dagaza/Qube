# What's new in Qube Help (v1.0.12)

## Common questions

- What changed in the help documentation?
- How do I search built-in help from chat?
- Where is the full documentation index?

## What it is

This page summarizes the **v1 English help corpus** shipped with Qube. The corpus lives in **Library → Qube** and powers **`@[tool:help]`** in **Conversations**.

**v1.0.12** adds **[Cognitive Router — how Qube routes a chat turn](../faq/cognitive-router-how-routing-works.md)**: route vocabulary, pre/post overrides, **● HYBRID** vs **HYBRID** route naming, Web vs Hybrid Internet Mode, empty-source downgrade, and where to inspect routing.

## Where to find it

Open **Library → Qube → release/whats-new.md**, or attach **`@[tool:help]`** and ask about help features. The router index is **[Qube Help](../00-index.md)**.

## Also called

help release notes, documentation changelog, in-app help v1, Qube documentation update

## Highlights (v1.0.12 cognitive router)

1. **[Cognitive Router FAQ](../faq/cognitive-router-how-routing-works.md)** — educative routing guide aligned with `docs/cognitive_router.md` and live UI.
2. **[Conversations](../features/conversations.md)** — **Web vs Hybrid Internet Mode** section; HYBRID dot disambiguation.
3. **Cross-links** — Telemetry, Knowledge, `@` mentions, diagnostic logs; four new canonical `@help` answers.

## Highlights (v1.0.11 diagnostic logs)

1. **[Diagnostic logs FAQ](../faq/diagnostic-logs-advanced-settings.md)** — `qube.log`, `llm_debug.log`, routing/web/skills logs, env overrides, bug-report workflow.
2. **[Advanced settings](../features/settings/advanced.md)** — summary table + cross-links; distinguishes live **Telemetry** from persistent log files.
3. **Canonical `@help` answers** — log location, routing debug log, LLM debug log content.

## Highlights (v1.0.10 telemetry docs)

1. **[Advanced Telemetry — interpreting](../faq/advanced-telemetry-interpreting.md)** — what each card measures, rolling windows, and diagnose-slow-reply workflow.
2. **[Advanced Telemetry feature](../features/telemetry.md)** — cross-link to FAQ; clarifies GPU compute vs VRAM and TPS chat-only.
3. **Canonical `@help` answers** — router intelligence, GPU 0%, slow reply diagnosis; richer TTFT answer.

## Highlights (v1.0.9 hardware & external engine)

1. **[Hardware tuning FAQ](../faq/hardware-tuning-internal-engine.md)** — GPU offload layers, CPU thread pool, VRAM caps, unified-memory systems, model reload behaviour.
2. **[Internal engine vs external server](../faq/internal-engine-vs-external-server.md)** — expanded: External context limit is **not** sent to the host; `cache_prompt`, VRAM unload, and tuning split.
3. **Corrected [Generation parameters](../faq/generation-parameters.md)** — External context wording aligned with `llm_worker` payload behaviour.

## Related

- [Qube Help index](../00-index.md)
- [Migration guide](migration-guide.md)
