"""Tests for app startup argument parsing."""

from __future__ import annotations

import unittest

from core.boot_args import parse_boot_args


class ParseBootArgsTests(unittest.TestCase):
    def test_routing_debug_flag_defaults_false(self) -> None:
        args = parse_boot_args([])
        self.assertFalse(args.routing_debug)

    def test_routing_debug_flag_enabled(self) -> None:
        args = parse_boot_args(["--routing-debug"])
        self.assertTrue(args.routing_debug)


if __name__ == "__main__":
    unittest.main()
