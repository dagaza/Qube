# Maintainer triage playbook

Internal guide for managing contributions at scale. Public expectations live in [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

---

## Roles

| Role | GitHub user | Open PRs | Review | Merge to `dev` / `main` |
|------|-------------|----------|--------|-------------------------|
| **Maintainer** | `@dagaza` | Yes | Yes | **Yes (sole merger initially)** |
| **Trusted contributor** | `@keithtmccartney` | Yes | Yes (advisory) | No |
| **Contributor** | Everyone else | Yes | No | No |

### Trusted contributor perks

- PRs from trusted contributors get label `trusted-contributor` and **priority review** when the queue is busy.
- Requested automatically on owned paths via [`.github/CODEOWNERS`](../../.github/CODEOWNERS).
- May label/triage issues when granted **Triage** repository role.
- Does **not** merge until explicitly promoted in Phase 2.

---

## Branch protection checklist (GitHub UI)

Configure once per branch: **Settings → Branches → Add rule** for `dev` and `main`.

| Setting | `dev` | `main` |
|---------|-------|--------|
| Require a pull request before merging | On | On |
| Require status checks to pass | On — **`test`** (CI workflow) | On — **`test`** |
| Require branches to be up to date before merging | Optional | On |
| Restrict who can push | **`dagaza` only** | **`dagaza` only** |
| Allow force pushes | Off | Off |
| Allow deletions | Off | Off |
| Require review from Code Owners | **Off** (Phase 0) | **Off** |

Notes:

- With a single merger, **do not** require approval from another human. CI green + maintainer review at merge time is the gate.
- Required check name in the UI is usually **`test`** (job id from [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)).
- After enabling protection, confirm a test PR shows the check as required before merging.

### Fork PR safety

**Settings → Actions → General → Fork pull request workflows**

- Enable **Require approval for all outside collaborators** (or first-time contributors) so unknown forks cannot run workflows with write tokens until approved.

---

## Daily triage order

Review PRs in this order:

1. CI green + complete **How I tested this** + small size (`size: XS` / `size: S` if labeled)
2. `trusted-contributor`
3. Linked `good first issue`
4. Everything else (FIFO)
5. **Deprioritize:** `needs-tests`, `needs-how-tested`, failing CI, `size: XL` without prior RFC

### Close or request changes when

- CI failing with no progress for 7+ days
- Empty or vague testing section → label `needs-how-tested`, request changes
- Behavior change without tests → label `needs-tests`
- Large feature PR without prior issue/RFC → ask to split or open RFC first
- Duplicate of an open PR

Stale bot ([`.github/workflows/stale.yml`](../../.github/workflows/stale.yml)) closes inactive PRs after 21 + 7 days. Labels `trusted-contributor`, `pinned`, `security`, and `RFC required` are exempt.

---

## Labels

Create in **Issues → Labels** if missing.

### Triage

| Label | Color suggestion | Meaning |
|-------|------------------|---------|
| `triage:needs-info` | `#FBCA04` | Waiting on reporter |
| `triage:needs-repro` | `#FBCA04` | Bug lacks reproduction steps |
| `triage:duplicate` | `#D4C5F9` | Duplicate |
| `triage:wontfix` | `#FFFFFF` | Out of scope |

### Work type

| Label | Meaning |
|-------|---------|
| `good first issue` | Curated starter task |
| `help wanted` | Maintainer wants community help |
| `RFC required` | Feature too large for a drive-by PR |

### PR queue

| Label | Meaning |
|-------|---------|
| `size: XS` | ≤ 50 lines changed |
| `size: S` | 51–200 |
| `size: M` | 201–400 |
| `size: L` | 401–800 |
| `size: XL` | > 800 — scrutinize; RFC likely |
| `trusted-contributor` | Trusted contributor PR — prioritize |
| `needs-tests` | Missing test coverage |
| `needs-how-tested` | PR template testing section incomplete |

### Area (optional)

`area:memory`, `area:routing`, `area:ui`, `area:packaging`, `area:help`, `area:knowledge`

Apply area labels manually during triage until automation is added.

---

## RFC policy (Phase 1)

Require discussion **before** large implementation PRs when any of:

- Estimated diff **> ~400 lines**
- Cross-cutting architecture (routing, memory schema, knowledge platform)
- New user-facing pillar or install/distribution path

Process:

1. Open a GitHub Discussion or issue with label `RFC required`.
2. Maintainer (or trusted contributor) comments **RFC approved** or requests changes to scope.
3. Contributor opens implementation PR linking the RFC.

Draft PRs before RFC approval are welcome for early feedback.

---

## Merge workflow

```
External PR → dev
     │
     ├─ CI red ──────────────────► Request changes / close
     │
     ├─ Template incomplete ─────► needs-how-tested, wait
     │
     ├─ Keith reviewed ──────────► Read comments + diff
     │
     └─ Maintainer merge ────────► dev

Release: maintainer merges dev → main (see docs/releasing.md)
Hotfix: branch from main → PR to main → merge → backport to dev
```

Only `@dagaza` merges until Phase 2 delegates merge-to-`dev` for trusted maintainers.

---

## Promotion path (informal)

| Tier | Criteria |
|------|----------|
| **Contributor** | Any merged PR |
| **Trusted** | Repeated quality PRs; invited by maintainer (Keith: current) |
| **Maintainer** | Sustained ownership of an area; invited by project owner |
| **Merge to `dev` (Phase 2)** | Trusted + explicit delegation from `@dagaza` |

No public reputation scores. Deprioritize repeat low-quality submissions via triage order, not public labels.

---

## Phase roadmap

| Phase | Trigger | Add |
|-------|---------|-----|
| **0** | Now | Branch protection, PR template, CODEOWNERS, stale bot, CI on `dev` |
| **1** | ~10 open PRs or regular external contributors | GitHub Discussions, RFC enforcement, `good first issue` curation |
| **2** | Maintainer vacation or Keith overload | Grant `@keithtmccartney` merge to `dev` only |

---

## Related files

- [`.github/pull_request_template.md`](../../.github/pull_request_template.md)
- [`.github/CODEOWNERS`](../../.github/CODEOWNERS)
- [`.github/workflows/stale.yml`](../../.github/workflows/stale.yml)
- [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
