"""Tests for generated help reference markdown."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.composer_attachments import COMPOSER_TOOLS
from core.composer_command_defs import COMPOSER_COMMANDS
from core.help_reference_generator import (
    generate_all_reference_markdown,
    generate_composer_attachments_markdown,
    generate_composer_skills_markdown,
    generate_composer_tools_markdown,
    generate_settings_sections_markdown,
)
from core.skills.registry import iter_skills


class HelpReferenceGeneratorTests(unittest.TestCase):
    def test_generates_all_reference_files(self) -> None:
        docs = generate_all_reference_markdown()
        self.assertEqual(
            set(docs),
            {
                "reference/composer-attachments.md",
                "reference/composer-tools.md",
                "reference/composer-commands.md",
                "reference/composer-skills.md",
                "reference/settings-sections.md",
                "reference/live-sources-overview.md",
            },
        )

    def test_composer_tools_lists_all_builtin_tokens(self) -> None:
        text = generate_composer_tools_markdown()
        self.assertIn("GENERATED FILE", text)
        self.assertIn("@[file:…]", text)  # routing cross-link wording in prose
        for tool in COMPOSER_TOOLS:
            self.assertIn(f"@[tool:{tool['id']}]", text)

    def test_composer_tools_documents_advanced_palette_and_presets(self) -> None:
        text = generate_composer_tools_markdown()
        self.assertIn("Advanced palette tools", text)
        self.assertIn("@[tool:user:…]", text)
        self.assertIn("@[tool:source:", text)
        self.assertIn("first routing attachment", text)

    def test_composer_skills_lists_all_builtin_tokens(self) -> None:
        text = generate_composer_skills_markdown()
        for skill in iter_skills():
            self.assertIn(f"@[skill:{skill.id}]", text)
        self.assertIn("Mutual exclusion", text)
        self.assertIn("Up to **three** skills apply per turn", text)

    def test_composer_attachments_covers_file_chat_routing(self) -> None:
        text = generate_composer_attachments_markdown()
        self.assertIn("@[file:", text)
        self.assertIn("@[chat:", text)
        self.assertIn("left-to-right", text)
        self.assertIn("remember", text)

    def test_composer_tools_lists_library_token(self) -> None:
        text = generate_composer_tools_markdown()
        self.assertIn("@[tool:library]", text)
        self.assertIn("@[tool:help]", text)
        self.assertIn("@[tool:internet]", text)

    def test_composer_commands_match_registry(self) -> None:
        docs = generate_all_reference_markdown()
        commands_md = docs["reference/composer-commands.md"]
        for cmd in COMPOSER_COMMANDS:
            self.assertIn(cmd.id, commands_md)
            self.assertIn(cmd.label, commands_md)

    def test_settings_sections_lists_ai_models(self) -> None:
        text = generate_settings_sections_markdown()
        self.assertIn("AI & Models", text)
        self.assertIn("`ai.models`", text)
        self.assertIn("Settings → Knowledge", text)


if __name__ == "__main__":
    unittest.main()
