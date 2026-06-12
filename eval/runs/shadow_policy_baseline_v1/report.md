# Router Evaluation Report

## Accuracy
- Strict accuracy (final route): **48.8%**
- Family accuracy (final route): **70.4%**
- Router accuracy: 10.4%
- Pre-retrieval accuracy: 21.6%
- Downgrade rate: 50.4% (63 cases)

## Retrieval Calibration Summary

- Strict accuracy: **48.8%**
- Family accuracy: **70.4%**
- Over-retrieval rate (CHAT leakage into retrieval): **27.8%** (20/72 CHAT-labeled)
- Under-retrieval rate (missed retrieval): **32.1%** (17/53 retrieval-expected)
- Recall-fusion share of over-retrieval: **20.0%** (4/20)
- Avg chat_score (correct CHAT cases): 0.688
- Avg chat_score (over-retrieval cases): 0.740
- Potential chat guard threshold (median − ε, not enforced): 0.625

### Over-retrieval by category
- adversarial: **28.6%** (2/7)
- ambiguous: **10.0%** (1/10)
- follow_up: **0.0%** (0/7)
- general_knowledge_retrieval_tempting: **40.0%** (10/25)

### CHAT confidence margin (CHAT-labeled prompts)
- 0-0.05: 61 cases
- 0.05-0.10: 8 cases
- 0.10-0.20: 3 cases
- 0.20+: 0 cases

### Retrieval suppression candidates

CHAT-labeled prompts that retrieved anyway (sorted by chat_score desc):

| case_id | chat_score | route | type | hits | recall_fusion |
|---------|------------|-------|------|------|---------------|
| gk_006 | 0.805 | hybrid | hybrid | 2 | False |
| gk_005 | 0.783 | hybrid | rag | 2 | False |
| gk_rt_008 | 0.766 | hybrid | hybrid | 4 | True |
| gk_rt_021 | 0.762 | hybrid | rag | 2 | True |
| gk_rt_004 | 0.757 | hybrid | rag | 4 | False |
| gk_rt_024 | 0.754 | hybrid | rag | 1 | True |
| gk_rt_019 | 0.751 | hybrid | rag | 5 | False |
| gk_007 | 0.749 | hybrid | rag | 1 | False |
| gk_rt_001 | 0.746 | hybrid | rag | 2 | True |
| gk_012 | 0.746 | hybrid | rag | 1 | False |
| gk_rt_013 | 0.745 | hybrid | rag | 3 | False |
| gk_001 | 0.739 | hybrid | rag | 1 | False |
| gk_016 | 0.739 | hybrid | rag | 3 | False |
| gk_rt_025 | 0.729 | hybrid | rag | 4 | False |
| amb_008 | 0.722 | hybrid | rag | 3 | False |
| adv_003 | 0.720 | hybrid | rag | 1 | False |
| gk_rt_016 | 0.708 | hybrid | rag | 3 | False |
| gk_020 | 0.701 | hybrid | rag | 4 | False |
| adv_005 | 0.694 | hybrid | rag | 5 | False |
| gk_rt_010 | 0.688 | hybrid | rag | 2 | False |

## ROUTE PERTURBATION INVARIANCE REPORT

- Cases analyzed: **125**
- Avg route consistency: **0.703**
- Avg retrieval consistency: **0.912**
- Stability: stable=0.0%, moderate=91.2%, highly_unstable=8.8%
- Web trigger stability (avg): 1.000

### By category
- ambiguous: route_cons=0.680 retr_cons=0.944 unstable=100.0%
- follow_up: route_cons=0.778 retr_cons=0.928 unstable=100.0%
- general_knowledge_retrieval_tempting: route_cons=0.597 retr_cons=0.832 unstable=100.0%
- memory_recall: route_cons=0.719 retr_cons=0.927 unstable=100.0%
- rag_retrieval: route_cons=0.737 retr_cons=0.938 unstable=100.0%

### Variance by base route type
- hybrid: avg_consistency=0.681 unstable_rate=100.0% (n=49)
- none: avg_consistency=0.714 unstable_rate=100.0% (n=69)
- rag: avg_consistency=0.752 unstable_rate=100.0% (n=7)

### Instability heatmap (unstable cases)

| chat_score \ margin | 0-0.05 | 0.05-0.10 | 0.10-0.20 | 0.20+ |
|---------------------|--------|-----------|-----------|-------|
| 0.0-0.3 | 0 | 0 | 0 | 0 |
| 0.3-0.5 | 0 | 0 | 0 | 0 |
| 0.5-0.7 | 45 | 12 | 9 | 0 |
| 0.7-1.0 | 39 | 12 | 8 | 0 |

### Top unstable cases

| case_id | category | route pattern | retrieval | route_cons | label |
|---------|----------|---------------|-----------|------------|-------|
| gk_rt_002 | general_knowledge_retrieval_tempting | hybrid ↔ none | 2hits/2miss | 0.50 | highly_unstable |
| gk_rt_004 | general_knowledge_retrieval_tempting | hybrid ↔ none | 2hits/2miss | 0.50 | highly_unstable |
| gk_rt_006 | general_knowledge_retrieval_tempting | hybrid ↔ none | 2hits/2miss | 0.50 | highly_unstable |
| gk_rt_020 | general_knowledge_retrieval_tempting | hybrid ↔ none | 2hits/2miss | 0.50 | highly_unstable |
| gk_rt_025 | general_knowledge_retrieval_tempting | hybrid ↔ none | 2hits/2miss | 0.50 | highly_unstable |
| amb_001 | ambiguous | hybrid ↔ none | 1hits/3miss | 0.50 | highly_unstable |
| amb_004 | ambiguous | hybrid ↔ none | 1hits/3miss | 0.50 | highly_unstable |
| amb_008 | ambiguous | hybrid ↔ none | 3hits/1miss | 0.50 | highly_unstable |
| gk_rt_005 | general_knowledge_retrieval_tempting | hybrid ↔ none | 1hits/3miss | 0.50 | highly_unstable |
| gk_rt_009 | general_knowledge_retrieval_tempting | hybrid ↔ none | 1hits/3miss | 0.50 | highly_unstable |

## ROUTING HYSTERESIS SIMULATION REPORT

- Flip reduction: **37.0%**
- Stability gain: **+0.076**
- Hybrid↔none suppression: **83.1%** (83 → 14)
- Retrieval consistency delta: **+0.000**

### Comparison table

| Metric | Baseline | Hysteresis | Delta |
|--------|----------|------------|-------|
| Route flips | 81 | 51 | ↓30 |
| Hybrid↔none flips | 83 | 14 | ↓69 |
| Retrieval consistency | 0.912 | 0.912 | +0.000 |

### Low-margin band (0–0.05) instability
- Baseline unstable cases: 54
- Hysteresis unstable cases: 8
- Reduction: 46

### By category
- ambiguous: stability_gain=+0.075 hybrid↔none_suppression=100.0%
- follow_up: stability_gain=+0.028 hybrid↔none_suppression=66.7%
- general_knowledge_retrieval_tempting: stability_gain=+0.165 hybrid↔none_suppression=93.8%

## ROUTING CANONICALIZATION LEARNER REPORT

- Clusters analyzed: **125**
- Stable clusters: baseline=68, shadow best=121
- Best threshold set: T_chat=0.80, T_margin_low=0.05, T_sep=0.10
- Cluster instability reduction: **93.0%**
- Variant canonical flip reduction: **-246.9%**
- Canonical agreement gain: **-0.321**
- Retrieval loss: **42.4%**
- Boundary noise share: **2.4%** | Semantic ambiguity: **0.8%**

**Interpretation:** Boundary tuning can stabilize cluster routes, but at significant retrieval cost. Root cause is likely recall-fusion/threshold interaction, not pure semantic ambiguity.

### Instability type breakdown
- boundary_instability: 3
- purely_ambiguous: 1
- recall_fusion_instability: 85
- retrieval_noise_instability: 0
- stable: 36

### Instability reclassification (best shadow thresholds)
- Boundary instability resolved: 100.0% (3/3)
- Retrieval noise unchanged: 100.0%
- Recall-fusion instability share: 68.0%

### Tradeoff curve (top threshold sets)

| T_chat | T_margin | T_sep | cluster Δ | retrieval loss | score |
|--------|----------|-------|-----------|----------------|-------|
| 0.80 | 0.05 | 0.10 | 93.0% | 42.4% | 0.082 |
| 0.80 | 0.08 | 0.10 | 93.0% | 42.4% | 0.082 |
| 0.80 | 0.10 | 0.03 | 93.0% | 42.4% | 0.082 |
| 0.80 | 0.10 | 0.05 | 93.0% | 42.4% | 0.082 |
| 0.80 | 0.10 | 0.07 | 93.0% | 42.4% | 0.082 |
| 0.80 | 0.10 | 0.08 | 93.0% | 42.4% | 0.082 |

- True ambiguity / inconsistent canonical clusters: 11

## RETRIEVAL PROPENSITY MODEL ANALYSIS (SHADOW CONTINUOUS LAYER)

- Avg propensity score: **0.454**
- Recall-fusion flip reduction: **100.0%**
- Hybrid↔none oscillation reduction: **-311.5%**
- Binary→continuous flip reduction: **-300.0%**
- Instability reduction proxy: **96.5%**
- Retrieval loss proxy: **0.0%**
- Best weights: w1=0.30, w2=0.25, w3=0.20, w4=0.15, w5=0.10
- Best thresholds: T_none=0.30, delta=0.08

**Interpretation:** Yes — continuous propensity modeling largely eliminates recall-fusion flip instability while preserving retrieval coverage better than binary gating.

### Comparison vs hysteresis
- Hysteresis flip reduction: 37.0%
- Hysteresis stability gain: +0.076
- Propensity instability reduction delta: +0.889

### Propensity variance by category
- adversarial: 0.0001
- ambiguous: 0.0001
- follow_up: 0.0006
- general_knowledge: 0.0001
- general_knowledge_retrieval_tempting: 0.0001
- memory_recall: 0.0001
- rag_retrieval: 0.0001
- web_retrieval: 0.0002

## CONTINUOUS RECALL-FUSION PILOT REPORT

- Avg propensity score: **0.454**
- Instability reduction: **96.5%**
- Retrieval loss proxy: **0.0%**
- Retrieval continuity: **0.461**
- Hybrid↔none flip reduction: **-311.5%**
- Flip reduction vs canonical: **-300.0%**
- Best thresholds: T_none=0.30, delta=0.08
- Resolves all unstable clusters: **False**

**Interpretation:** Pilot routing reduces instability while preserving retrieval coverage better than canonicalization/hysteresis threshold approaches.

### Comparison vs hysteresis / canonicalization
- Hysteresis stability gain: 0.0763
- Hysteresis hybrid↔none reduction: 83.1%
- Canonicalization instability reduction: 93.0%
- Canonicalization retrieval loss: 42.4%

### Per-category retrieval coverage (pilot vs baseline)
- adversarial: baseline=14.7% pilot=100.0% delta=+85.3%
- ambiguous: baseline=12.2% pilot=100.0% delta=+87.8%
- follow_up: baseline=12.5% pilot=100.0% delta=+87.5%
- general_knowledge: baseline=49.5% pilot=100.0% delta=+50.5%
- general_knowledge_retrieval_tempting: baseline=49.6% pilot=100.0% delta=+50.4%
- memory_recall: baseline=87.0% pilot=100.0% delta=+13.0%
- rag_retrieval: baseline=88.0% pilot=100.0% delta=+12.0%
- web_retrieval: baseline=0.0% pilot=100.0% delta=+100.0%

### Flip patterns
- hybrid↔none: see cluster-level `hybrid_none_flips_*` in `continuous_pilot_routing.json`
- memory↔rag: see cluster-level `memory_rag_flips_*` in export

## CONTINUOUS RECALL-FUSION ARCHITECTURAL VALIDATION REPORT

- Validation passed: **True**
- Avg propensity score: **0.454**
- Instability reduction: **96.5%**
- Retrieval loss proxy: **0.0%**
- Retrieval continuity: **0.000**
- Best thresholds: T_none=0.30, delta=0.08

**Interpretation:** Pilot routing reduces instability while preserving retrieval coverage better than canonicalization/hysteresis threshold approaches.

### Comparison matrix (unstable clusters)

| Method | Unstable clusters |
|--------|-------------------|
| Baseline | 57 |
| Pilot | 2 |
| Hysteresis | 10 |
| Canonical shadow | 4 |

### Instability reduction vs alternatives
- Pilot vs baseline: **96.5%**
- Hysteresis stability gain: 0.0763
- Canonicalization reduction: 0.9298

### Retrieval loss proxy
- Pilot: **0.0%** | Canonicalization: 0.4238

### Flip patterns

| Flip type | Baseline | Pilot | Hysteresis | Canon shadow |
|-----------|----------|-------|------------|--------------|
| hybrid↔none | 78 | 321 | 13 | 244 |
| memory↔rag | 0 | 0 | — | — |

### By corpus category

| Category | Instability Δ | Retrieval loss | hybrid↔none Δ | Agreement |
|----------|---------------|----------------|---------------|-----------|
| adversarial | 100.0% | 0.000 | -25 | 14.7% |
| ambiguous | 100.0% | 0.000 | -34 | 12.2% |
| follow_up | 100.0% | 0.000 | -51 | 12.5% |
| general_knowledge | 100.0% | 0.000 | -27 | 49.5% |
| general_knowledge_retrieval_tempting | 100.0% | 0.000 | -34 | 49.6% |
| memory_recall | 100.0% | 0.000 | 4 | 87.0% |
| rag_retrieval | 66.7% | 0.000 | 0 | 88.0% |
| web_retrieval | 0.0% | 0.000 | -76 | 0.0% |

### Threshold sweep (top candidates)

| T_none | delta | instability Δ | retrieval loss |
|--------|-------|-----------------|----------------|
| 0.30 | 0.08 | 96.5% | 0.0% |
| 0.30 | 0.10 | 96.5% | 0.0% |
| 0.30 | 0.12 | 96.5% | 0.0% |
| 0.35 | 0.08 | 96.5% | 0.0% |
| 0.35 | 0.10 | 96.5% | 0.0% |

### Top unstable clusters (pilot)

| case_id | category | baseline pattern | pilot pattern | retrieval loss |
|---------|----------|------------------|---------------|----------------|
| rag_002 | rag_retrieval | none ↔ rag | hybrid ↔ rag | 0.000 |
| rag_003 | rag_retrieval | none ↔ rag | hybrid ↔ rag | 0.000 |

## SHADOW LLMWORKER RETRIEVAL POLICY ANALYSIS

- Avg propensity score: **0.454**
- Route divergence rate: **53.9%**
- Recall-fusion eliminated rate: **100.0%**
- Hybrid stability gain: **-311.5%**
- Instability reduction: **96.5%**
- Retrieval coverage delta: **+0.000**
- Regression suppressions: 0 | Stability improvements: 0
- Thresholds: T_none=0.30, delta=0.08 | w1=0.30, w2=0.25

**Interpretation:** Shadow policy eliminates recall-fusion cluster instability in offline replay — binary fusion appears redundant for retrieval activation.

### By category

| Category | Divergence | Suppression |
|----------|------------|-------------|
| adversarial | 85.3% | 0.0% |
| ambiguous | 87.8% | 0.0% |
| follow_up | 87.5% | 0.0% |
| general_knowledge | 50.5% | 0.0% |
| general_knowledge_retrieval_tempting | 50.4% | 0.0% |
| memory_recall | 13.0% | 0.0% |
| rag_retrieval | 12.0% | 0.0% |
| web_retrieval | 100.0% | 0.0% |

## Retrieval Hit Rates
- adversarial: 33.3%
- ambiguous: 10.0%
- follow_up: 14.3%
- general_knowledge: 35.0%
- general_knowledge_retrieval_tempting: 40.0%
- memory_recall: 100.0%
- rag_retrieval: 94.4%
- web_retrieval: 0.0%

## Failure Causes

- no_failure: 61
- recall_fusion_upgrade: 27
- router_miss: 20
- downgrade_to_none: 13
- relevance_gate_removed_results: 3
- override_changed_route: 1

## Category Breakdown

| Category | n | Strict | Family | Downgrades |
|----------|---|--------|--------|------------|
| adversarial | 7 | 71.4% | 71.4% | 4 |
| ambiguous | 10 | 90.0% | 90.0% | 9 |
| follow_up | 12 | 58.3% | 66.7% | 6 |
| general_knowledge | 20 | 65.0% | 65.0% | 13 |
| general_knowledge_retrieval_tempting | 25 | 60.0% | 60.0% | 15 |
| memory_recall | 18 | 11.1% | 100.0% | 0 |
| rag_retrieval | 18 | 38.9% | 94.4% | 1 |
| web_retrieval | 15 | 20.0% | 20.0% | 15 |

## Rewrite Impact

_Rewrite analysis not enabled (use `--with-sidecar`)._

## Memory Recall Analysis

- Total memory cases: 18
- With memory hits: 12
- Without memory hits: 6
- Strict success: 2
- Family success: 18

### By memory type (misses)
- episodic: 0 misses / 3 total (100.0% hit rate)
- personal_fact: 2 misses / 4 total (50.0% hit rate)
- preference: 2 misses / 7 total (71.4% hit rate)
- relationship: 2 misses / 4 total (50.0% hit rate)

## Confusion Matrix (strict, expected → final)

- none: hybrid=20, none=52
- memory: hybrid=17, none=1
- hybrid: hybrid=2, none=1
- rag: hybrid=10, none=2, rag=7
- web: none=13

## Interpretation Guide

1. **Over-retrieval rate** — CHAT-labeled prompts that activated retrieval with hits.
2. **Recall-fusion share** — fraction of over-retrieval driven by recall fusion.
3. **Chat score gap** — if over-retrieval cases have low chat_score, a margin guard may help; high chat_score suggests override/fusion not router scoring.
4. **Under-retrieval** — true missed retrieval (not label mismatch).
5. **Suppression candidates** — high chat_score but still retrieved; best targets for guard tuning.