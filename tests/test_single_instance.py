"""Tests for duplicate-process protection."""

from __future__ import annotations

import pytest

from core.single_instance import SingleInstanceGuard, build_single_instance_server_name


@pytest.fixture
def unique_server_name(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "qube-test-single-instance-deadbeef"
    monkeypatch.setattr(
        "core.single_instance.build_single_instance_server_name",
        lambda app_id="dagaza.qube": name,
    )
    return name


def test_build_single_instance_server_name_is_user_scoped() -> None:
    name = build_single_instance_server_name()
    assert name.startswith("dagaza.qube-")
    assert len(name.split("-", 1)[1]) == 12


def test_second_guard_exits_when_primary_is_running(qapp_cls, unique_server_name) -> None:
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    primary = SingleInstanceGuard(parent=app)
    assert primary.try_acquire() is True

    duplicate = SingleInstanceGuard(parent=app)
    assert duplicate.try_acquire() is False


def test_activation_handler_runs_for_duplicate_launch(qapp_cls, unique_server_name) -> None:
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    primary = SingleInstanceGuard(parent=app)
    assert primary.try_acquire() is True

    activations: list[str] = []

    def _on_activate() -> None:
        activations.append("focused")

    primary.set_activation_handler(_on_activate)

    duplicate = SingleInstanceGuard(parent=app)
    assert duplicate.try_acquire() is False
    app.processEvents()
    assert activations == ["focused"]
