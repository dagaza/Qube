# Changelog

All notable changes to Qube are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-20

### Added
- In-app help corpus v1 with `@help` routing and educative documentation.
- Coach guide overlays for all major pages and areas.
- Web page fetching (`web_content_fetch`) and Settings refactor for clearer UX.
- macOS packaging foundation: `.app` bundle, DMG output, and optional signing/notarization pipeline.
- Homebrew Cask distribution scaffolding and automated tap bump workflow.
- Launch-ready README rewrite (user-first storefront; technical depth moved to `docs/`).
- User documentation: [docs/user/](docs/user/README.md) (install, requirements, workflows).
- Architecture docs: [docs/architecture/](docs/architecture/README.md) (extracted from pre-rewrite README).
- Archived legacy README: [docs/archive/readme-pre-launch-rewrite.md](docs/archive/readme-pre-launch-rewrite.md).
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer setup, tests, PR expectations.
- [docs/launch_documentation_guidelines.md](docs/launch_documentation_guidelines.md) — phased doc playbook and pre-launch checklist.
- Release checklist in [docs/releasing.md](docs/releasing.md) now includes documentation pass before public launch.
- Phase 4 (partial): social preview image, GitHub Pages landing, README repositioning (privacy + grounding lead; comparison at bottom).
- [Competitive landscape](docs/user/competitive-landscape.md) — feature matrix, `@`/help, onboarding, runtime/UI RAM, observability, Live Sources, Desktop Companion, **memory**.
- [Competitive roadmap](docs/competitive_roadmap.md) — developer priorities: parity, moats, non-goals.
- [MCP capability integrations plan](docs/mcp_capability_integrations_plan.md) — roadmap for future MCP-based integrations.

### Changed
- Faster theme toggle via lazy-loading main stages and profiling regressions.

### Fixed
- Model Manager page UI bug and silenced GPU layer debug instrumentation.
- Dependency bumps (click, setuptools) to clear `pip-audit` CI findings.

## [1.0.1] - 2026-06-28

### Added
- First-run bootstrap consent dialog (Recommended and Advanced) with disk and memory feasibility checks.
- Selective Hugging Face model downloads on splash to `%LOCALAPPDATA%\Qube\models`.
- Missing-model notifications and Settings download actions for voice, RAG/library, and cognition.
- Advanced **Continue without models** shell install path with confirmation dialog.
- Mock bootstrap downloads only via `--mock-bootstrap-download` (real downloads by default).

### Fixed
- WinGet first-run hang when embedding/GGUF assets were missing (phased splash boot, background downloads).
- Splash progress and bootstrap UX polish; STT load on launch; TTS modularity hardening.

## [1.0.0] - 2026-06-05

### Added
- Initial public release of the Qube PyQt6 desktop assistant.
- Windows release pipeline: PyInstaller, Inno Setup, GitHub Release, WinGet manifest generation.
- Frozen-aware path resolution via `core.paths`.
- PR CI workflow with pytest and `pip-audit`.

[Unreleased]: https://github.com/dagaza/Qube/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/dagaza/Qube/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/dagaza/Qube/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/dagaza/Qube/releases/tag/v1.0.0
