# Changelog

All notable changes to Qube are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/dagaza/Qube/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/dagaza/Qube/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/dagaza/Qube/releases/tag/v1.0.0
