# ADR 003 — Evidence Convergence Invariant

**Status:** Accepted  
**Date:** 2026-07-10  
**Related:** [External Knowledge Platform Plan](../external_knowledge_platform_plan.md), [Knowledge Platform Evolution Review](../knowledge_platform_evolution_review.md)

## Context

Qube's knowledge platform supports built-in adapters, configured sources, presets, and multiple domain pipelines. Without a single contract at the retrieval boundary, features tend to fork into parallel models (legacy web rows, custom bundles, plugin-specific shapes).

## Decision

**Every retrieval mechanism on the v2 path must produce `EvidenceObject`s inside an `EvidenceBundle` before ranking output is consumed by prompt assembly or LLM interaction.**

### Implications

1. **KnowledgeService.retrieve()** always returns `(EvidenceBundle, relevance_diag, raw_audit)`.
2. **Configured sources** and **built-in adapters** share one adapter ID namespace and converge through `get_search_function()`.
3. **Legacy `run_legacy_web_retrieval()`** is a compatibility path only (`bundle=None`). It is not an extension point.
4. **Inspectability by default:** every v2 turn writes a `RetrievalRecord` in SQLite linking `request_id`, `bundle_id`, and a context fingerprint — independent of verbose audit logging.

## Orchestration vs domain ranking

| Concept | Controls | User-facing |
|---------|----------|-------------|
| **Retrieval Profile** | Fan-out, latency, cache, source ordering | Yes (Fast, Balanced, Thorough, …) |
| **Ranking Profile** | Domain scoring weights | No (internal / Advanced) |
| **Knowledge Preset** | Which adapters compose a composer tool | Yes (My knowledge) |

Presets answer *what sources*; retrieval profiles answer *how hard/fast/local* to search.

## Consequences

- New connectors must emit rows consumable by existing bundle builders.
- Replay, Explain, and Pipeline Graph read from `RetrievalRecord` + `RetrievalTrace` — not parallel telemetry systems.
- Domain pipelines (e.g. scientific) may keep specialized rankers but should compose shared orchestration primitives (`orchestration_kernel.py`).

## Non-goals

- Replacing all domain pipelines with `GenericOrchestrationPipeline` in one step.
- User-defined Python execution or arbitrary scripting.
- Multiple evidence model types (`EnterpriseEvidenceBundle`, etc.).
