#!/usr/bin/env python3
"""Remove extracted handler methods from settings_view.py shell."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui" / "views" / "settings" / "settings_view.py"

SHELL_METHODS = {
    "__init__",
    "select_settings_section",
    "showEvent",
    "resizeEvent",
    "sync_active_native_model_label",
    "refresh_native_local_library",
    "refresh_menu_themes",
    "update_voice_dropdown",
    "_setup_ui",
    "_scroll_to_settings_anchor",
    "_add_settings_group_header",
    "_add_settings_section",
    "_index_section_for_search",
    "_wire_companion_cognition_hint",
    "_sync_internal_engine_subsections",
    "_on_settings_search_changed",
    "_init_settings_layout",
    "_finalize_settings_layout",
    "_build_settings_section_nav_row",
    "_on_settings_section_changed",
    "_update_settings_section_nav_colors",
    "_build_section_header",
    "_build_divider",
    "_apply_settings_menu_button_chevron_state",
    "_make_settings_info_button",
}

MIXIN_IMPORT = """
from ui.views.settings.handlers import (
    AiModelsHandlersMixin,
    CompanionHandlersMixin,
    GenerationMixin,
    MemoryHandlersMixin,
    PersistenceHandlersMixin,
    PrestigeMenuMixin,
    StylingMixin,
    VoiceHandlersMixin,
)
"""


def main() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SettingsView")
    lines = src.splitlines(keepends=True)

    keep_ranges: list[tuple[int, int]] = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in SHELL_METHODS:
                keep_ranges.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.Assign):
            keep_ranges.append((node.lineno, node.end_lineno))

    keep_ranges.sort()
    new_class_body: list[str] = []
    for start, end in keep_ranges:
        new_class_body.extend(lines[start - 1 : end])

    # Find class line
    class_start = cls.lineno - 1
    class_end = cls.end_lineno

    # Insert mixin bases
    class_line = lines[class_start]
    if "HandlersMixin" not in class_line:
        class_line = class_line.replace(
            "class SettingsView(QWidget):",
            "class SettingsView(\n    QWidget,\n    PrestigeMenuMixin,\n    GenerationMixin,\n    StylingMixin,\n    VoiceHandlersMixin,\n    AiModelsHandlersMixin,\n    MemoryHandlersMixin,\n    CompanionHandlersMixin,\n    PersistenceHandlersMixin,\n):",
        )

    before_class = lines[:class_start]
    after_class = lines[class_end:]

    if "from ui.views.settings.handlers import" not in src:
        # insert after sections import block
        insert_at = None
        for i, line in enumerate(before_class):
            if line.startswith("from ui.views.settings.sections import"):
                insert_at = i + 1
                while insert_at < len(before_class) and (
                    before_class[insert_at].startswith("    ")
                    or before_class[insert_at].strip() == ")"
                ):
                    insert_at += 1
                break
        if insert_at is None:
            raise SystemExit("Could not find import insertion point")
        before_class.insert(insert_at, MIXIN_IMPORT)

    out = "".join(before_class) + class_line + "\n" + "".join(new_class_body) + "".join(after_class)
    SOURCE.write_text(out, encoding="utf-8")
    print(f"Shell methods kept: {len(keep_ranges)}")


if __name__ == "__main__":
    main()
