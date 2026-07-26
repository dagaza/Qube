# Changelog

All notable changes to Qube are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Linux CUDA `.deb` release packaging: prune safe bundle bloat, bundle only required NVIDIA wheel libraries, use maximum xz compression, and fail the build early if the `.deb` still exceeds GitHub's 2 GiB asset limit.
- Linux release bundle prune: restrict `strip --strip-debug` to llama.cpp libs only; stripping numpy/scipy OpenBLAS wheels broke CPU/Vulkan smoke tests.
- Linux CUDA `.deb` recompress: fix `ar` repack after xz -9e (do not use empty `mktemp` deb) and skip recompress when fpm output is already under GitHub's 2 GiB cap.

## [1.2.2] - 2026-07-25

### Fixed
- Release CI: allowlist Themes guided tour for lazy-stage footgun audit and register `settings.appearance_themes` in page tour infrastructure tests.
- Release CI: stabilize session-scoped UI tests (theme toggle state, wallpaper defaults, onboarding coach panel sizing) so the full suite passes reliably on Windows runners.

### Added
- [`docs/release_versioning_quick_reference.md`](docs/release_versioning_quick_reference.md) — semver/tag discipline for maintainers and coding agents (fix on `main` before tagging; avoid patch bumps for failed CI only).

## [1.2.1] - 2026-07-25

### Fixed
- Release CI: update Settings/Themes tests for lazy section prefetch, template-based QSS rendering, and deferred Themes preview init.
- Settings → AI & Models: wire toolbar generation spinboxes after the lazy section builds so Max Reply Tokens stays in sync.
- Settings → Themes: add guided tour registration for the new Themes section.

## [1.2.0] - 2026-07-24

### Added
- Customizable color schemes (Themes v1): Settings → Appearance → Themes with built-in and custom schemes, live draft preview, import/export, and follow-system light/dark polarity.
- Surface fills / wallpapers (Themes v2): per-surface wallpaper profiles for the chat transcript and library preview, bundled presets, custom image import, and overlay strength presets (Subtle / Balanced / Vivid).
- Python-first theme system with resolved semantic tokens for shell chrome.
- In-app help for Themes settings and updated Conversations / Library documentation.

### Fixed
- Linux `.deb` packages: use xz compression so CUDA `.deb` artifacts stay under GitHub Releases' 2 GiB per-asset limit.
- Themes preview: fix "More components" toggle and off-screen snapshot rendering blocking radio clicks.

### Changed
- Settings → Appearance opens faster via lazy section construction.

## [1.1.11] - 2026-07-23

### Fixed
- macOS Intel release CI: install `lancedb==0.25.3` and `onnxruntime==1.23.2` on x86_64 runners (current pins have no macOS Intel wheels on PyPI).

## [1.1.10] - 2026-07-23

### Fixed
- Release CI: build Intel macOS DMGs on `macos-15-intel` instead of retired `macos-13` runners (fixes indefinite queue blocking the release job).

## [1.1.9] - 2026-07-23

### Fixed
- Linux CUDA release CI: skip dist/AppImage runtime smoke on driverless GitHub runners and verify bundled CUDA wheel libraries instead (`libcuda.so.1` requires an end-user NVIDIA driver).

## [1.1.8] - 2026-07-23

### Fixed
- Linux CUDA builds: resolve NVIDIA wheel lib paths via namespace `__path__` (not `__file__`) and stage `nvidia-cublas-cu12` alongside `nvidia-cuda-runtime-cu12` for llama-cpp CUDA wheels.

## [1.1.7] - 2026-07-23

### Fixed
- Linux Vulkan builds: restore `spirv-headers` (required by ggml-vulkan CMake) alongside LunarG Vulkan SDK packages.
- Linux CUDA builds: stage `libcudart` into `llama_cpp/lib/` after PyInstaller via `stage_cuda_runtime_libs.py` so smoke tests can load CUDA llama-cpp wheels on CI runners.

## [1.1.6] - 2026-07-23

### Fixed
- Linux Vulkan builds: install matching `vulkan-headers` and `libvulkan-dev` from LunarG (Ubuntu 22.04 stock headers are too old for current llama.cpp ggml-vulkan).
- Linux CUDA builds: copy NVIDIA CUDA runtime `.so` files into `llama_cpp/lib/` so bundled `libllama.so` can load `libcudart.so.12` during smoke tests.

## [1.1.5] - 2026-07-22

### Fixed
- Linux Vulkan builds: install `glslc` from LunarG's apt repo on Ubuntu 22.04 (not in default jammy packages).
- Linux AppImage smoke test: resolve AppImage path to an absolute path so `xvfb-run` can execute it.
- Linux CUDA builds: bundle `nvidia-cuda-runtime-cu12` (and cublas) into the PyInstaller output so smoke tests can import llama-cpp without a host CUDA toolkit.

## [1.1.4] - 2026-07-22

### Fixed
- Linux Vulkan builds: install `glslc` package (Ubuntu 22.04) instead of non-existent `shaderc` apt name.
- Linux PyInstaller bundle: include `mf2py`, `extruct`, and `recipe_scrapers` data files so smoke tests no longer crash on missing `backcompat-rules`.

## [1.1.3] - 2026-07-22

### Fixed
- Linux release CI: install `libegl1` for PyInstaller/PyQt smoke tests (`libEGL.so.1` missing on runners).
- Linux Vulkan builds: install `shaderc` (`glslc`) required by llama-cpp-python CMake.
- Linux CUDA/Vulkan builds on CI: skip GPU runtime llama-cpp import verify on GitHub Actions (no GPU/CUDA on runners).
- Linux `.deb` packages: declare `libegl1` as a runtime dependency.

## [1.1.2] - 2026-07-22

### Fixed
- Linux release CI: `pip_install()` in `install_llama_cpp_variant.sh` now passes the `install` subcommand to pip.
- macOS uninstaller: render user-specific paths with `$HOME` instead of the build runner's home directory.

## [1.1.1] - 2026-07-22

### Fixed
- Linux release CI: install `portaudio19-dev` so PyAudio builds during PyInstaller packaging.
- macOS release CI: install PortAudio via Homebrew before `pip install` (fixes missing `portaudio.h` on arm64 and x86_64 runners).

## [1.1.0] - 2026-07-22

### Added
- In-app help corpus v1 with `@help` routing and educative documentation.
- Coach guide overlays for all major pages and areas.
- Expanded global toolbar panel with updated coach guide, tooltips, and tutorial copy.
- Web page fetching (`web_content_fetch`) and Settings refactor for clearer UX.
- macOS packaging foundation: `.app` bundle, DMG output, and optional signing/notarization pipeline.
- macOS uninstaller with DMG helper, in-app controls, and user documentation.
- Linux release packaging: AppImage and `.deb` artifacts with CPU, Vulkan, and CUDA variants.
- Homebrew Cask distribution scaffolding and automated tap bump workflow.
- Launch-ready README rewrite (user-first storefront; technical depth moved to `docs/`).
- User documentation: [docs/user/](docs/user/README.md) (install, requirements, workflows).
- Architecture docs: [docs/architecture/](docs/architecture/README.md) (extracted from pre-rewrite README).
- Archived legacy README: [docs/archive/readme-pre-launch-rewrite.md](docs/archive/readme-pre-launch-rewrite.md).
- [CONTRIBUTING.md](CONTRIBUTING.md) — developer setup, tests, PR expectations.
- [docs/launch_documentation_guidelines.md](docs/launch_documentation_guidelines.md) — phased doc playbook and pre-launch checklist.
- Release checklist in [docs/releasing.md](docs/releasing.md) now includes documentation pass before public launch.
- Phase 4 (partial): social preview image, GitHub Pages landing, README repositioning (privacy + grounding lead; comparison at bottom).
- [MCP capability integrations plan](docs/mcp_capability_integrations_plan.md) — roadmap for future MCP-based integrations.

### Changed
- Faster theme toggle via lazy-loading main stages and profiling regressions.

### Fixed
- Model Manager page UI bug and silenced GPU layer debug instrumentation.
- Dependency bumps (click, setuptools) to clear `pip-audit` CI findings.
- Six-hour CI hang on TTS voice toggle test; added `pytest-timeout` safety net.
- Tests updated for `user_data_root()` path resolution after Linux packaging work.

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

[Unreleased]: https://github.com/dagaza/Qube/compare/v1.1.7...HEAD
[1.1.7]: https://github.com/dagaza/Qube/compare/v1.1.6...v1.1.7
[1.1.6]: https://github.com/dagaza/Qube/compare/v1.1.5...v1.1.6
[1.1.5]: https://github.com/dagaza/Qube/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/dagaza/Qube/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/dagaza/Qube/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/dagaza/Qube/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/dagaza/Qube/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/dagaza/Qube/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/dagaza/Qube/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/dagaza/Qube/releases/tag/v1.0.0
