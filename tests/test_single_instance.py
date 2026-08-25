"""Tests for duplicate-process protection."""

from __future__ import annotations

import uuid

import pytest
from PyQt6.QtNetwork import QLocalServer

from core.single_instance import SingleInstanceGuard, build_single_instance_server_name


@pytest.fixture
def unique_server_name(monkeypatch: pytest.MonkeyPatch) -> str:
    name = f"qube-test-{uuid.uuid4().hex}"
    monkeypatch.setattr(
        "core.single_instance.build_single_instance_server_name",
        lambda app_id="dagaza.qube": name,
    )
    yield name
    QLocalServer.removeServer(name)


def _release_guard(guard: SingleInstanceGuard) -> None:
    guard.release()


def test_build_single_instance_server_name_is_user_scoped() -> None:
    name = build_single_instance_server_name()
    assert name.startswith("dagaza.qube-")
    assert len(name.split("-", 1)[1]) == 12


def test_second_guard_exits_when_primary_is_running(qapp_cls, unique_server_name) -> None:
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    primary = SingleInstanceGuard(parent=None)
    try:
        assert primary.try_acquire() is True
        primary.set_activation_handler(lambda: True)

        duplicate = SingleInstanceGuard(parent=None)
        assert duplicate.try_acquire() is False
    finally:
        _release_guard(primary)
        app.processEvents()


def test_activation_handler_runs_for_duplicate_launch(qapp_cls, unique_server_name) -> None:
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    primary = SingleInstanceGuard(parent=None)
    try:
        assert primary.try_acquire() is True

        activations: list[str] = []

        def _on_activate() -> bool:
            activations.append("focused")
            return True

        primary.set_activation_handler(_on_activate)

        duplicate = SingleInstanceGuard(parent=None)
        assert duplicate.try_acquire() is False
        for _ in range(20):
            if activations:
                break
            app.processEvents()
        assert activations == ["focused"]
    finally:
        _release_guard(primary)
        app.processEvents()


def test_yielding_primary_allows_relaunch_takeover(qapp_cls, unique_server_name) -> None:
    """Headless/zombie primary must not ACK so the next launch becomes primary."""
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    primary = SingleInstanceGuard(parent=None)
    secondary = SingleInstanceGuard(parent=None)
    try:
        assert primary.try_acquire() is True

        def _yield() -> bool:
            primary.release()
            return False

        primary.set_activation_handler(_yield)

        assert secondary.try_acquire() is True
        assert secondary._owns_server  # noqa: SLF001
    finally:
        _release_guard(primary)
        _release_guard(secondary)
        app.processEvents()


def test_stale_socket_without_ack_allows_listen(qapp_cls, unique_server_name) -> None:
    """A listen-only server that never ACKs must not block a real Qube start."""
    del unique_server_name
    app = qapp_cls.instance() or qapp_cls([])
    name = build_single_instance_server_name()
    stale = QLocalServer()
    assert stale.listen(name)
    try:
        # No activation handler / ACK path — connects succeed but never ACK.
        guard = SingleInstanceGuard(parent=None)
        assert guard.try_acquire() is True
        assert guard._owns_server  # noqa: SLF001
        _release_guard(guard)
    finally:
        stale.close()
        QLocalServer.removeServer(name)
        app.processEvents()
