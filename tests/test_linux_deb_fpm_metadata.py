"""Metadata for fpm .deb builds (scripts/linux/build_deb.sh)."""

from __future__ import annotations

from core.linux_release_variants import (
    LINUX_RELEASE_VARIANTS,
    deb_conflicts,
    deb_description,
)
from core.uninstall_paths import deb_runtime_dependencies


def test_deb_description_is_multi_word_for_all_variants():
    """build_deb.sh must not split description on whitespace (see build_rpm.sh)."""
    for variant in LINUX_RELEASE_VARIANTS:
        assert " " in deb_description(variant)


def test_deb_fpm_conflict_packages_are_single_tokens():
    for variant in LINUX_RELEASE_VARIANTS:
        names = [n.strip() for n in deb_conflicts(variant).split(",")]
        assert len(names) == 2
        for name in names:
            assert name
            assert " " not in name


def test_deb_fpm_runtime_dependencies_have_no_spaces():
    for variant in LINUX_RELEASE_VARIANTS:
        for dep in deb_runtime_dependencies(variant=variant):
            assert " " not in dep
