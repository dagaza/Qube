"""Tests for discipline pack sync validation."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discipline_pack_sync import (  # noqa: E402
    assert_discipline_packs_synced,
    suggest_pack_notes,
    validate_discipline_packs,
)
from core.knowledge.scientific_discipline_packs import get_discipline_pack  # noqa: E402


class TestDisciplinePackSync(unittest.TestCase):
    def test_packs_validate_clean(self) -> None:
        errors = validate_discipline_packs()
        self.assertEqual(errors, [], msg="\n".join(errors))

    def test_assert_helper_passes(self) -> None:
        assert_discipline_packs_synced()

    def test_suggest_pack_notes_omit_stub_language(self) -> None:
        pack = get_discipline_pack("computer_science")
        self.assertIsNotNone(pack)
        assert pack is not None
        notes = suggest_pack_notes(pack)
        self.assertNotIn("(stub)", notes.lower())
        self.assertIn("acm_dl", notes)
