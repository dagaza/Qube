# Changelog

All notable changes to Qube are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Early splash (pre-bootstrap):** show a fully opaque branded card with the static Qube logo and a **Loading…** label instead of a black box / frozen circle spinner (timers cannot run while `import main` blocks the GUI thread).
- **Bootstrap splash close (Windows):** closing the startup/bootstrap splash with **X** now aborts phased boot and force-exits the process so Qube cannot linger invisible with no taskbar or tray icon.
- **Bootstrap relaunch after kill (Windows):** single-instance locking now requires an ACK from a live primary; a headless/zombie process (or stale named pipe after Task Manager kill) yields the lock so the next `Qube.exe` launch can show the bootstrap window again.
- **WinGet CUDA validation:** defer `llama_cpp` import until the native or sidecar engine loads a model so post-install validation does not load CUDA DLLs during startup (avoids Microsoft Defender false positives on `dagaza.Qube.CUDA`).
- **Windows uninstall:** stop `Qube.exe` before Inno Setup removal (AppMutex + `taskkill`), and verify uninstall while the app is still running in release CI so `%LOCALAPPDATA%\Programs\Qube\` (including `_internal\`) is fully removed.
- **Windows installer:** `SetupMutex` blocks running multiple Qube setup wizards at once (including CPU/Vulkan/CUDA variants).
- **Windows startup splash:** activate and center early/bootstrap splash on the launch screen so it is not stuck behind other windows during long first-run imports.
- **Bootstrap Whisper download:** stream CTranslate2 weight files directly (same HTTP path as sidecar/Kokoro) instead of `huggingface_hub.snapshot_download`, which could hang at 0% on Windows before any bytes were written.

## [1.3.2] - 2026-08-23

### Fixed
- **Windows bootstrap startup:** early splash, single-instance locking, and streamed Hugging Face download progress for Whisper/Kokoro so first-run bootstrap no longer stalls or opens duplicate sessions.
- **Windows duplicate launch:** reliable second-instance activation via local sockets, including buffered/disconnect edge cases and synchronous handoff before `try_acquire` returns.
- **Release smoke tests:** fix tooltip-controller reentrancy crash on early splash (`RecursionError` / Windows fast-fail) and align Windows dist smoke with Linux bootstrap mocking.
- **Linux AppImage smoke:** tear down the full dist-smoke process tree so a leftover `Qube` process does not block the AppImage liveness check via single-instance locking.
- **Homebrew cask:** fix RuboCop style failures (`depends_on macos`, stanza order, alphabetized `zap trash`, trailing comma).

## [1.3.1] - 2026-08-22

### Fixed
- **Windows Vulkan/CUDA installers:** bundle `vulkan-1.dll` and verify GPU backend DLLs so `llama.dll` loads on clean WinGet validation VMs; defer `llama_cpp` imports so GPU builds launch without crashing when optional runtimes are absent.
- **WinGet Vulkan manifest:** declare dependency on `KhronosGroup.VulkanRT`.
- **Release CI:** smoke-launch all three Windows variants and verify Vulkan/CUDA bundle layouts before publishing.
- **Dependencies:** bump `pypdf` to 6.15.0 for pip-audit (PYSEC-2026-3655/3656).

## [1.3.0] - 2026-08-04

### Added
- **Production license signing key** (`qube-prod-1`) embedded for offline Pro license verification in shipped builds.
- **`tools/generate_qube_signing_key.py`** — maintainer CLI to generate Ed25519 signing keys and register public keys in `assets/licensing/signing_keys.json`.
- **Batch license issuance** — `issue_qube_license.py --count` writes a CSV manifest plus per-customer QUBE1 serial key files.

### Changed
- License signing maintainer docs updated for key generation and batch issuance workflows.
- Licensing CLI tools bootstrap the repo root on `sys.path` when run directly as scripts.

## [1.2.9] - 2026-08-03

### Added
- **Reading font** setting in Settings → Themes: bundled OFL fonts (Inter, Source Sans 3, IBM Plex Sans, Literata) plus a searchable system-font browser for Conversations and Library previews.
- **Themes UX polish**: Pro+ gate for save/import/export theme actions; expanded wallpaper preset catalog; improved swatch borders and preset grid layout.

### Changed
- General settings help, tour, and @mentions cross-links (composer default, Discovery GPU caveats, unit labels).

### Fixed
- System font picker excludes Font Awesome and other icon-font families on Windows.

## [1.2.8] - 2026-08-02

### Added
- **Pro license gating** for custom STT/TTS/embedding model paths, alternate wakeword libraries (and Test Lab), and MCP Filesystem integrations — runtime enforcement, Settings UI, and license sync on import/remove.
- **Deep Research** research profiles, PDF report export, and Pro-gated advanced features.
- **Assistant message export** from Conversations (Markdown/PDF) with related composer and settings updates.

### Fixed
- Settings lazy-load CI hang when wakeword catalog sync runs against mocked audio workers in tests.

## [1.2.7] - 2026-08-02

### Added
- **Conversation turn index** for long chats: compact tick rail beside the transcript column; click or **Ctrl+↑/↓** to jump between user prompts; hover tooltips with turn number and prompt preview; smooth scroll animation.

### Changed
- Routing debug logging enabled by default for easier diagnosis of RAG and memory routing.
- Hardened routing for RAG and memory capability calls.

### Fixed
- Prompt hang when opening or using the prompts UI.
- Transcript turn-index layout: column width and layout-mode (800px / 1200px) behavior preserved alongside the new rail.

## [1.2.6] - 2026-08-01

### Added
- **MCP / capability integration** (Phases 0–4): configured MCP servers, capability registry and invoke path, consent and session egress review, composer capability tokens, and **Settings → Integrations**.
- **Library Pro depth**: structural chunking, per-import precision indexing, precision retrieval, Pro license gating, PDF text normalization, and evaluation tooling.
- **Settings redesign**: split **Privacy & data**, **Diagnostics**, **License**, and slim **Advanced** (JSON editor); sticky section titles; collapsible cards and updated visual hierarchy.
- In-app help updates for the Settings split, MCP workflows, Library Pro depth, and related `@help` canonical answers.

### Changed
- README install guidance: WinGet and Chocolatey GPU commands; macOS Homebrew tap install and upgrade section.

### Fixed
- Settings lazy-load and shared theme-manager subscription cleanup for stable Windows CI.
- UI test reliability for system settings handlers, TTS voice selector sync, and nav sidebar theme toggle.

## [1.2.5] - 2026-07-28

### Added
- Cross-platform in-place update support: Settings → Help release checker against GitHub Releases; Windows Inno upgrade messaging; Linux AppImage install cleanup; CI upgrade smoke tests; user and maintainer update docs ([`docs/user/update-qube.md`](docs/user/update-qube.md), [`docs/app_update_roadmap.md`](docs/app_update_roadmap.md)).
- Windows **Vulkan** and **CUDA** release installers (`Qube-{version}-vulkan-Setup.exe`, `Qube-{version}-cuda-Setup.exe`); WinGet and Chocolatey remain on the CPU `Qube-{version}-Setup.exe`.
- Linux **`.rpm`** (Fedora/RHEL) and portable **`.tar.gz`** artifacts alongside AppImage and `.deb` for each CPU/Vulkan/CUDA variant.
- Unsigned **Homebrew** custom tap (`brew tap dagaza/qube` → `brew install --cask qube`) with automated CI submission to [`dagaza/homebrew-qube`](https://github.com/dagaza/homebrew-qube).

### Changed
- README install guidance: GPU variant table, Linux RPM/tarball formats, and Homebrew tap instructions.

## [1.2.4] - 2026-07-27

### Added
- Commercial platform foundation: capability/licensing infrastructure (offline `.qube-license` verify, pack signing, Settings import).
- Launch-trust feature slices: composer `@` discoverability, INSPECT routing explainability, router regression baseline, trust/privacy help corpus, web discovery telemetry, SearXNG setup wizard, and Linux AppImage install path.
- Settings and UI polish: theme preview cards, settings card layout, and faster Conversations/Library loads.

### Fixed
- Linux CUDA `.deb` release packaging: prune safe bundle bloat, bundle only required NVIDIA wheel libraries, use maximum xz compression, and fail the build early if the `.deb` still exceeds GitHub's 2 GiB asset limit.
- Linux release bundle prune: restrict `strip --strip-debug` to llama.cpp libs only; stripping numpy/scipy OpenBLAS wheels broke CPU/Vulkan smoke tests.
- Linux CUDA `.deb` recompress: fix `ar` repack after xz -9e (do not use empty `mktemp` deb) and skip recompress when fpm output is already under GitHub's 2 GiB cap.
- Theme preview width test for card-constrained layouts on CI runners.
- Bump `cryptography` to 48.0.1 (GHSA-537c-gmf6-5ccf).

## [1.2.3] - 2026-07-26

### Added
- Memory Simple/Advanced mode toggle in Settings.
- Gradient stops for theme wallpaper customization.
- Themes area UX: separate preview cards for Conversations, Library, and Settings-style pages; tooltips and updated coach guide.

### Changed
- Dark mode and Catppuccin Dark are the default on first launch; desktop companion is off by default.
- Updated minimum system requirements documentation.
- Faster Conversations and Library page loads; faster theme switching and preview updates.

### Fixed
- Assistant reply bubbles no longer break when changing themes; transcript bubble rendering and navbar colors corrected.
- Chat wallpaper preview mock in Settings matches applied appearance.
- Theme preview width and Themes section polish.
- Theme persistence: saved light/dark scheme survives app restart when appearance preference was never explicitly set (CI regression fix).

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
