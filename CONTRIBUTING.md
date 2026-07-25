# Contributing to Qube

Thank you for helping improve Qube. This project is maintained on GitHub at [dagaza/Qube](https://github.com/dagaza/Qube).

---

## Before you start

- **Bug reports & features:** [GitHub Issues](https://github.com/dagaza/Qube/issues) or [qubeapp.eu](https://www.qubeapp.eu) via **Settings → Contact & Feedback** in the app.
- **Documentation:** User-facing prose lives in [`docs/user/`](docs/user/README.md) and in-app help (`assets/help/en/`). The README is a short storefront — see [`docs/launch_documentation_guidelines.md`](docs/launch_documentation_guidelines.md) before large doc changes.
- **Product priorities vs competitors:** [`docs/competitive_roadmap.md`](docs/competitive_roadmap.md) — parity gaps, moats to deepen, intentional non-goals (companion doc: [`docs/user/competitive-landscape.md`](docs/user/competitive-landscape.md)).

---

## Quick contribution flow

For most changes:

1. Create a branch from `dev`.
2. Make your changes.
3. Run the relevant tests locally (see below), or the full validation suite if your change affects multiple components.
4. Open a pull request **into `dev`** describing what changed and how you tested it.

CI runs the full validation suite on every PR to `dev` or `main`. Maintainers may ask for additional validation if needed.

**Examples:** A typo in user docs needs no help-corpus scripts. A settings change likely needs help regeneration and targeted pytest modules.

---

## Development setup

Requires **Python 3.12+** (CI uses 3.13). Full steps: [`docs/user/install-from-source.md`](docs/user/install-from-source.md).

```bash
git clone https://github.com/dagaza/Qube.git
cd Qube
git checkout dev
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt -r requirements-dev.txt
python main.py
```

Optional GPU build (Linux): `./scripts/install_llama_cpp_gpu.sh`

Local validation details: [`docs/local_validation.md`](docs/local_validation.md).

---

## Running tests

### Full validation (matches CI)

Run this before opening a PR when your change touches code, dependencies, or in-app help:

```bash
python scripts/generate_help_reference.py --check
python scripts/compose_help_corpus.py --check
python scripts/validate_help_manifest.py
python scripts/eval_help_golden.py
python scripts/eval_help_production.py

pytest tests/ -v --tb=short -m "not packaging"
pip-audit -r requirements.txt --ignore-vuln CVE-2025-69872
```

For doc-only changes that do not affect help content, `pytest` alone is usually enough locally — CI will still run the full suite.

Focused smoke (releases):

```bash
pytest tests/test_memory_qa_smoke.py -q
```

---

## Pull requests

### Branching

| Branch | Purpose |
|--------|---------|
| **`dev`** | Integration branch for features and bug fixes — **open PRs here** |
| **`main`** | Release-ready branch; maintainers promote `dev` → `main` when cutting a release |

- **Feature and bug-fix PRs** — target **`dev`**. Branch from an up-to-date `dev`.
- **Release promotion and urgent hotfixes** — target **`main`** (maintainer coordination).

### Checklist

1. **Branch from `dev`** — keep changes focused; one logical change per PR when possible.
2. **Ensure CI passes** — the [CI workflow](.github/workflows/ci.yml) runs on every PR to `dev` or `main`.
3. **Update docs when behavior changes:**
   - **In-app help** — regenerate and validate (see below); bump `corpus_version` when retrieval content changes.
   - **User docs** — [`docs/user/`](docs/user/) for install, requirements, workflows.
   - **CHANGELOG** — add notes under `[Unreleased]` for user-visible changes.
   - **README** — only for positioning, new pillars, or install path changes (not implementation detail).
4. **Describe the PR** — what changed, why, and how you tested it.

Small fixes can be submitted directly. For larger features or behavioral changes, opening a discussion or issue first is encouraged.

Draft pull requests are welcome if you'd like early feedback before a feature is complete.

---

## In-app help corpus

Regenerate the help corpus whenever your changes affect the content or retrieval of in-app help (for example help prose, settings registries, or help composition scripts):

```bash
python scripts/generate_help_reference.py
python scripts/compose_help_corpus.py
python scripts/validate_help_manifest.py
python scripts/eval_help_golden.py
```

Checklist and quarterly review process: [`docs/in_app_help_knowledge_base.md`](docs/in_app_help_knowledge_base.md) and [`docs/releasing.md`](docs/releasing.md) (help section).

---

## Development practices

### Code style

- Match surrounding code — naming, imports, and documentation level in the file you edit.
- Prefer minimal, focused diffs over drive-by refactors.
- Comments only for non-obvious business logic or deep technical constraints.

### Testing and review

- **Add or update tests** when behavior changes — follow existing `tests/test_<module>.py` patterns in the area you touched.
- **No secrets in commits** — API keys, credentials, tokens, or machine-specific paths.
- **Check ADRs** before changing routing, skills, or knowledge architecture: [`docs/adr/`](docs/adr/README.md).
- **User-visible changes** need CHANGELOG notes and doc updates (see Pull requests above).

### Coding agents (Cursor and others)

If you use Cursor or another coding agent, start with the scoped rule files in [`.cursor/rules/`](.cursor/rules/) (`*.mdc`). They capture non-obvious invariants for areas such as the native engine, RAG/memory, UI, audio workers, and diagnostics — constraints that are easy to miss from code alone.

When your PR introduces a new subsystem, changes architectural contracts, or adds "don't do X" rules that agents should respect, update an existing `.mdc` file or add a new one alongside your change. Follow the existing pattern: a short `description`, relevant `globs`, and concise invariant bullets — not a second README.

This is not required for typos or small fixes. It is good practice for behavioral and architectural changes so future contributors (human and agent) stay aligned.

---

## Learn the codebase

- **Architecture:** [`docs/architecture/`](docs/architecture/README.md)
- **Cognitive router:** [`docs/cognitive_router.md`](docs/cognitive_router.md)
- **Logging & diagnostics:** [`docs/logging_and_diagnostics.md`](docs/logging_and_diagnostics.md)

### Capability Plane vs internal `mcp/` package

Qube has two unrelated uses of the name **MCP**:

| Path | Meaning |
|------|---------|
| `core/integrations/providers/mcp/` | **Model Context Protocol** — external tool servers as `CapabilityProvider` peers (Theme C). |
| `mcp/` (e.g. `cognitive_router.py`) | **Internal cognitive routing** — memory/RAG/WEB lanes, unrelated to the protocol. |

**Rules for contributors:**

- External protocol code lives only under `core/integrations/providers/mcp/`.
- Provider-agnostic layers (`core/integrations/`, UI, router, INSPECT) must not `import mcp` or branch on `provider == "mcp"` (principle P6).
- When adding routing or retrieval features, integrate with the existing Capability Plane (`core/integrations/capabilities/`) — do not create a parallel `core/mcp/` subsystem.

See [`docs/mcp_capability_architecture_review.md`](docs/mcp_capability_architecture_review.md) (P1–P8, §12 PR checklist).

---

## Maintainer notes

Release procedures (tagging, WinGet, Chocolatey, macOS, pre-launch documentation pass): [`docs/releasing.md`](docs/releasing.md).

Promoting integration work: merge reviewed PRs into `dev`, then cut releases by merging `dev` into `main` per [`docs/releasing.md`](docs/releasing.md).

Triage, roles, branch protection, and label policy: [`docs/maintainer/triage.md`](docs/maintainer/triage.md).

---

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](LICENSE) as the project.
