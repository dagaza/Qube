# Knowledge Platform Evolution — Design Review

**Status:** Implemented (Phases 0–4 core)  
**Date:** 2026-07-10  
**Related:** [External Knowledge Platform Plan](./external_knowledge_platform_plan.md), [ADR 003 — Evidence convergence](./adr/003-evidence-convergence.md)

## Executive summary

Qube's knowledge spine — `Composer → Knowledge Service → Pipeline → Adapter Runtime → EvidenceBundle` — is sound. The extensibility platform (presets, configured sources, connector types) converges correctly on v2.

This review challenged several proposals and implemented the highest-leverage hardening first:

- Fixed `@[tool:user:*]` worker wiring
- Always-on `RetrievalRecord` persistence
- Retrieval Profiles (orchestration, not ranking)
- Orchestration kernel + trace schema v3 with pipeline stages
- Explain Preset, Retrieval Inspector (Summary / Graph / Compare / Explain)
- Retrieval Replay (current config + compare)

**Rejected as near-term work:** universal pipeline replacement, new connector protocols, workflow editors, user Python plugins.

---

## Architectural principles (documented)

See [ADR 003](./adr/003-evidence-convergence.md):

1. Evidence convergence on v2 paths
2. Orchestration vs domain ranking separation
3. Inspectability by default (`RetrievalRecord`)
4. Presets are source bundles, not pipelines
5. Transparency over integration breadth
6. Local-first boundary for sqlite/filesystem connectors

---

## Component evaluations

### GenericOrchestrationPipeline

**Verdict:** Reference orchestration pattern, not universal engine. Scientific/finance/legal pipelines keep domain rankers.

**Implemented:** `core/knowledge/orchestration_kernel.py` — shared fan-out, budget, stage traces, adapter policy enforcement. `pipeline_generic.py` migrated first.

### Retrieval Profiles

**Verdict:** Warranted. Materialize into `RetrievalBudget` + behavioral knobs via `RetrievalContext.retrieval_profile`.

**Implemented:** Settings → Knowledge → Retrieval profile (Fast / Balanced / Thorough / Evidence-first / Local-first). Applied on generic/preset path; scientific pipeline honors cache + local-first ordering hints.

### Explain Preset

**Verdict:** High-value transparency. Derived view only — no duplicate schema.

**Implemented:** Settings → My knowledge → Explain selected; richer composer tool descriptions; Inspector Explain tab.

### Retrieval Replay

**Verdict:** Strong fit; phased. Full reproducibility bounded by live API drift.

**Implemented:** `core/knowledge/retrieval_replay.py` — replay from `RetrievalRecord`, compare evidence/coverage/latency. Inspector Compare tab. Replay Original labeled best-effort.

### Pipeline Graph Visualization

**Verdict:** Explanation tool only — not a workflow editor.

**Implemented:** Trace schema v3 `pipeline_stages`; `pipeline_graph.py`; Inspector Graph tab.

---

## Implementation map

| Phase | Deliverable | Key files |
|-------|-------------|-----------|
| 0 | Worker fix, RetrievalRecord, ADR | `llm_worker.py`, `database.py`, `retrieval_records.py`, `web_retrieval.py` |
| 1 | Explain, trace panel, Inspector shell | `explain_preset.py`, `retrieval_inspector.py`, `knowledge_presets.py`, `prestige_dialog.py` |
| 2 | Retrieval profiles | `retrieval_profiles.py`, `knowledge.py`, `app_settings.py` |
| 3 | Orchestration kernel, trace v3, graph | `orchestration_kernel.py`, `observability.py`, `pipeline_graph.py` |
| 4 | Replay + compare | `retrieval_replay.py` |

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Pipeline unification regressions | Kernel extraction only; domain rankers preserved |
| Replay false confidence | Best-effort labeling; compare shows drift |
| Profile × preset combinatorics | Profiles affect orchestration only; presets affect adapter set only |

---

## What not to do next

- More connector protocols before strengthening REST/local UX
- Big-bang pipeline collapse
- User-configurable ranking coefficients
- Cloud-dependent monetization that weakens OSS inspectability

---

## Coherence metric

Qube remains a **transparent, inspectable inference platform** when users can answer:

- What sources ran?
- Why were results kept or dropped?
- Can I re-run and compare?

Use **Inspect Retrieval** on any evidence answer, **Explain selected** on presets, and **Settings → Diagnostics** for trace details.
