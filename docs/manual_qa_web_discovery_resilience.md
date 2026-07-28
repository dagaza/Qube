# Manual QA — Web discovery privacy & resilience

**Status:** Maintainer checklist (R10)  
**Related:** [web_discovery_privacy_resilience_plan.md](web_discovery_privacy_resilience_plan.md)

Use this matrix before release when discovery policy, pacing, budgets, or telemetry surfaces change. Tests in CI mock providers — this doc covers **manual** verification on a real install.

---

## Prerequisites

- Fresh or reset discovery health (**Settings → Knowledge → Web search discovery → Reset discovery health** when available)
- **Private search (recommended)** tier unless the row says otherwise
- Network available; Brave API key **not** configured for Tier A rows

---

## Matrix

| ID | Tier / profile | Action | Expected |
|----|----------------|--------|----------|
| A1 | Tier A (Private) | `@internet` weather query | DDG primary SERP; no Brave/Bing API calls; INSPECT shows discovery policy |
| A2 | Tier A | Rapid 3× `@internet` within pacing window | Pacing delay felt; **Telemetry → Web discovery** pacing row updates |
| A3 | Tier A | Exhaust burst budget (or lower env limits for test) | Fallback providers; budget counters on Settings + Telemetry; notification if implemented |
| A4 | Tier A | Simulate/mock DDG bot challenge | Backoff countdown; Wikipedia fallback; **Primary provider** shows backoff on Telemetry |
| A5 | Tier A | Trusted Library path during DDG backoff | No live DDG HTTP for trusted retrieval (post-R1) |
| B1 | Tier B + Brave key | DDG failure or backoff | Brave used when tier allows; INSPECT provider line shows Brave |
| C1 | Tier D (SearXNG) | **Set up SearXNG…** wizard: Detect local → Test → Save | URL saved; optional tier switch; provider badge **Configured** |
| D1 | Any | **Telemetry → Web discovery** open 10s | Tier, budgets, pacing refresh without error |
| D2 | Any | **INSPECT RETRIEVAL** on web reply | Summary includes discovery policy lines |
| E1 | Tier A | App restart during DDG backoff | Backoff persists; no duplicate spam notifications |
| E2 | Tier A | Enable **Web search log**, one search, **View** | JSONL lines; redaction flags respected when set at launch |

---

## Telemetry card checks (R10)

While **Advanced Telemetry** is visible:

1. **Privacy tier** matches **Settings → Knowledge**.
2. **DDG burst/session** counters increment only on **live** DDG requests (not cache hits).
3. **System health** shows ⚠️ when backoff, budget exhaustion, or conservative mode is active.
4. Card matches **Active discovery route** summary on Knowledge settings (same policy source).

---

## Log redaction smoke

Launch once with:

```bash
QUBE_WEB_SEARCH_AUDIT_REDACT=1 QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY=1 ./Qube
```

Enable web search + routing logs, send one web turn, confirm query fields are hashed/omitted in files under `~/.qube/logs/`.

---

## Sign-off

| Date | Tester | App version | Tier A pass | Notes |
|------|--------|-------------|-------------|-------|
| | | | ☐ | |

---

*Abbreviated from [web_discovery_privacy_resilience_plan.md §11](web_discovery_privacy_resilience_plan.md). Expand rows when R5 session dedup or new providers ship.*
