"""Detect lazy main-stage access patterns that eager-build hidden pages.

Lazy stages (Library, Memory, Telemetry, Model Manager, Settings) must be
accessed via private ``_foo_view`` (peek) or ``ensure_foo_view()`` (build).
Public properties and ``getattr(host, "settings_view")`` call ``hasattr`` /
property getters and construct the full page tree unintentionally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_LAZY_VIEW_NAMES = (
    "library_view",
    "memory_manager_view",
    "telemetry_view",
    "model_manager_view",
    "settings_view",
)

# Files where property/getter access is intentional (navigation, tests, property defs).
_AUDIT_ALLOWLIST = frozenset(
    {
        "ui/main_window.py",
        "ui/onboarding/local_llm_setup_tour.py",
        "ui/onboarding/tour_registry.py",
        "ui/onboarding/tour_helpers.py",
        "ui/onboarding/settings_tour_header.py",
        "ui/onboarding/tour_runner.py",
        "ui/onboarding/tours/conversations.py",
        "ui/onboarding/tours/library.py",
        "ui/onboarding/tours/memory_manager.py",
        "ui/onboarding/tours/model_manager.py",
        "ui/onboarding/tours/telemetry.py",
        "ui/onboarding/tours/settings/voice_audio.py",
        "ui/onboarding/tours/settings/_common.py",
        "ui/onboarding/tours/settings/ai_models.py",
        "ui/onboarding/tours/settings/memory.py",
        "ui/onboarding/tours/settings/knowledge.py",
        "ui/onboarding/tours/settings/general.py",
        "ui/onboarding/tours/settings/appearance_themes.py",
        "ui/onboarding/tours/settings/companion_desktop.py",
        "ui/onboarding/tours/settings/notifications.py",
        "ui/onboarding/tours/settings/help.py",
        "ui/onboarding/tours/settings/contact_feedback.py",
        "ui/onboarding/tours/settings/privacy_data.py",
        "ui/onboarding/tours/settings/diagnostics.py",
        "ui/onboarding/tours/settings/license.py",
        "ui/onboarding/tours/settings/integrations.py",
        "ui/onboarding/tours/settings/_diagnostic_logs.py",
        "ui/onboarding/tours/settings/advanced.py",
        "tests/test_ui_settings_open.py",
        "tests/test_composer_commands.py",
        "tests/test_lazy_main_stages.py",
        "core/lazy_main_stage_footguns.py",
        "core/theme_toggle_profile.py",
    }
)

_GETATTR_PATTERN = re.compile(
    r"""getattr\s*\(\s*[^,]+,\s*["'](?P<name>"""
    + "|".join(_LAZY_VIEW_NAMES)
    + r""")["']""",
)

_HASATTR_PATTERN = re.compile(
    r"""hasattr\s*\(\s*[^,]+,\s*["'](?P<name>"""
    + "|".join(_LAZY_VIEW_NAMES)
    + r""")["']""",
)

_DOT_ACCESS_PATTERN = re.compile(
    r"""(?P<prefix>(?:self\.window|window|w|host|main_window))\.(?P<name>"""
    + "|".join(_LAZY_VIEW_NAMES)
    + r""")\b""",
)


@dataclass(frozen=True)
class LazyStageFootgun:
    path: str
    line_no: int
    line: str
    view_name: str
    kind: str


def _should_audit_file(rel_path: str) -> bool:
    if rel_path in _AUDIT_ALLOWLIST:
        return False
    if not rel_path.endswith(".py"):
        return False
    if rel_path.startswith("tests/"):
        return False
    return True


def scan_lazy_stage_footguns(
    root: Path,
    *,
    include_paths: list[Path] | None = None,
) -> list[LazyStageFootgun]:
    """Return suspicious lazy-view access sites under ``root``."""
    findings: list[LazyStageFootgun] = []
    paths: list[Path]
    if include_paths is not None:
        paths = include_paths
    else:
        paths = []
        skip_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "dist",
            "build",
            "__pycache__",
        }
        for path in sorted(root.rglob("*.py")):
            if any(part in skip_dirs for part in path.parts):
                continue
            paths.append(path)

    for path in paths:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not _should_audit_file(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern, kind in (
                (_GETATTR_PATTERN, "getattr"),
                (_HASATTR_PATTERN, "hasattr"),
                (_DOT_ACCESS_PATTERN, "property"),
            ):
                match = pattern.search(line)
                if match is None:
                    continue
                findings.append(
                    LazyStageFootgun(
                        path=rel,
                        line_no=line_no,
                        line=stripped,
                        view_name=match.group("name"),
                        kind=kind,
                    )
                )
    return findings
