# Release versioning — quick reference

**Audience:** Maintainers, contributors, and coding agents (Cursor, etc.)  
**Full procedures:** [`releasing.md`](releasing.md) (WinGet, Chocolatey, macOS, Linux, checklists)

This document defines **when to bump semver**, **when to tag**, and **what to do when release CI fails** — without burning patch numbers on every test fix.

---

## Core rule

> **Tags are publication events, not CI retry buttons.**

A git tag `vX.Y.Z` triggers **Build & Release** (`.github/workflows/release.yml`). The **test job runs first**. If it fails, **no installers are published** — but the tag and version bump commits still clutter history if you tag too early.

**Patch numbers (`1.2.1`, `1.2.2`, …) should mean “users can install this build,” not “we fixed another test on main.”**

---

## What each version means

| Artifact | Meaning |
|----------|---------|
| **`main` commits** | Integration; may include unreleased fixes |
| **`CHANGELOG.md` → `[Unreleased]`** | Notes for work not yet tagged |
| **`CHANGELOG.md` → `## [X.Y.Z]`** | Notes for a version you intend to **ship** |
| **Git tag `vX.Y.Z`** | Triggers release CI; should only be pushed when ready to publish |
| **GitHub Release + assets** | **Shipped** — semver is “consumed” for users and package managers |

PR CI (`.github/workflows/ci.yml`) runs the **same test command** as the release workflow:

```bash
pytest tests/ -v --tb=short -m "not packaging"
```

Release CI must be green on **`main`** before you tag, not discovered tag-by-tag.

---

## Recommended workflow (economical tags)

```text
1. Merge features/fixes → main
2. Accumulate user-facing notes under CHANGELOG [Unreleased]
3. Verify main is green (CI + local pytest above)
4. prepare_release.py X.Y.Z  →  version files + CHANGELOG section
5. Commit "Release X.Y.Z" on main
6. git tag vX.Y.Z  &&  git push origin main vX.Y.Z
7. Wait for Build & Release → confirm GitHub Release assets
8. Announce / package-manager updates only after step 7 succeeds
```

**Do not** bump patch and tag again after every failed release attempt. **Fix on `main` first**, then tag once.

---

## If release CI fails after you tagged

### Nothing was published (test job failed)

Typical case: test failure before build/release jobs run.

1. **Do not** immediately cut `X.Y.(Z+1)` unless you already need a new semver for other reasons.
2. Fix tests (or release infra) on **`main`** with normal commits.
3. Keep notes in **`[Unreleased]`** or fold into the pending release section — avoid serial `1.2.1`, `1.2.2`, … entries for unreleased attempts.
4. When **`main` is green**, choose one path:

   | Situation | Action |
   |-----------|--------|
   | Tag never had a GitHub Release / no assets | Delete remote tag `vX.Y.Z` (team agreement), re-run `prepare_release.py` for same version if needed, **re-tag `vX.Y.Z`** once green |
   | Prefer not to delete tags | Use **`vX.Y.Z-rc.N`** for validation (see below), ship **`vX.Y.Z`** only when RC passes |
   | Tag already announced / mirrors picked it up | Ship **`vX.Y.(Z+1)`** with a real fix — do not rewrite history |

### Installers were published (release job succeeded)

- **Never** force-move or delete the tag.
- Ship a **new patch** (`vX.Y.(Z+1)`) with the fix.
- Mark bad release **Pre-release** or yank assets per [`releasing.md`](releasing.md) rollback section.

---

## Release candidates (optional, tag-efficient validation)

Use when you want CI to run the full release pipeline without committing to the final semver:

```text
v1.3.0-rc.1  →  validate (mark GitHub Release pre-release if you publish RC artifacts)
v1.3.0-rc.2  →  if needed
v1.3.0       →  only after RC is green; this is what users install
```

RC tags are **validation-only** in narrative terms; public docs and package managers should point at the final `vX.Y.Z`.

*(Today, only `v*` tags trigger Build & Release. RC tags follow the same workflow unless you add a separate verify workflow later.)*

---

## CHANGELOG discipline

| Do | Don't |
|----|--------|
| Put work-in-progress under **`[Unreleased]`** | Add `## [1.2.1]` for every failed tag attempt |
| Add **`## [X.Y.Z]`** in the same commit you intend to tag | Bump version in `__version__.py` before CI is green |
| Fold multiple pre-release fixes into **one** shipped section | List phantom releases users never downloaded |

---

## Version bump tooling

| Tool | Purpose |
|------|---------|
| `scripts/prepare_release.py X.Y.Z` | Sync `core/__version__.py` + `pyproject.toml`, check CHANGELOG |
| `scripts/set_version.py X.Y.Z` | Version files only (CI uses this from the tag) |
| Git tag `vX.Y.Z` | **Canonical** at publish time; CI derives version from tag |

Run `prepare_release.py` **once**, immediately before the commit + tag you expect to ship.

---

## Guidance for AI agents

When asked to **cut a release**, **fix release CI**, or **bump version**:

1. **Read** [`releasing.md`](releasing.md) and this doc.
2. **If release CI failed on an existing tag:** fix on `main`, run the same pytest as CI, **do not** tag a new patch until tests pass — unless assets were already published.
3. **Do not** run `prepare_release.py` or create git tags until the user explicitly wants to publish **and** `main` is expected to pass CI (or user accepts RC flow).
4. **Do not** add sequential CHANGELOG sections (`1.2.1`, `1.2.2`, …) for unreleased test-fix iterations; use `[Unreleased]` until a successful ship.
5. **Do not** delete or force-push tags unless the user confirms nothing was published and team policy allows it.
6. After a **successful** release only: verify GitHub Release assets, then consider the semver final for users.

---

## Anti-patterns (learned from 1.2.0 → 1.2.2)

```text
❌  Tag v1.2.0 → CI fails → tag v1.2.1 → fails → tag v1.2.2
✅  Merge themes → fix tests on main → CI green → tag v1.2.0 once
```

Failed tags without published artifacts do not help users; they only multiply version numbers and CHANGELOG noise.

---

## See also

- [`releasing.md`](releasing.md) — full release checklist, platforms, rollback
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — branch flow (`dev` → `main`), PR expectations
- [`.github/workflows/release.yml`](../.github/workflows/release.yml) — tag-triggered pipeline
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — PR/main test parity
