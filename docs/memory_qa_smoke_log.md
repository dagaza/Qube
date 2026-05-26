# Memory QA — Automated Smoke Log

Proxy coverage for manual plan Sections **1**, **6**, and **E2E-1/E2E-2** plumbing in [`memory_manual_qa.md`](memory_manual_qa.md).

## Latest run

| Field | Value |
|-------|-------|
| **Date** | 2026-05-26 |
| **Command** | `.venv/bin/python -m pytest tests/test_memory_qa_smoke.py -q` |
| **Result** | **8 passed** |

## Test mapping

| Automated test | Manual QA ID | What it validates |
|----------------|--------------|-----------------|
| `test_s1_promotion_default_off` | S1.3 | Promotion opt-in default |
| `test_s1_consolidation_default_on` | S1.6 | Consolidation default on |
| `test_s1_enrichment_default_on` | (baseline) | Enrichment default on |
| `test_s1_promotion_preset_default_standard` | S1.5 | Preset default |
| `test_m6_export_visible_writes_markdown` | M6.5 | Export path + markdown body |
| `test_m6_negative_list_blocks_similar_reinsert` | M6.3, E2E-2 | Negative list reject |
| `test_e2e1_recall_intent_detected_for_stored_fact_query` | E2E-1 (partial) | Recall intent detection |
| `test_e2e1_explicit_remember_detection` | E2E-1 (partial) | Explicit remember parse |

## Still manual-only

- **S1.1, S1.2, S1.4, S1.7** — toggle wiring in live UI
- **M6.1–M6.4** — Edit, flag, delete, bulk delete in Memory Manager
- **E2E-1 full** — learn → recall → edit in UI → recall updated text

Run the automated suite before each memory release candidate; execute the manual sections above in the desktop app for sign-off.
