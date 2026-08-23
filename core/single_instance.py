"""Prevent multiple Qube GUI processes; focus the existing window on relaunch."""

from __future__ import annotations

import getpass
import hashlib
import logging
from collections.abc import Callable

from PyQt6.QtCore import QObject
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

logger = logging.getLogger("Qube.SingleInstance")

_ACTIVATE_MESSAGE = b"activate"
_CONNECT_TIMEOUT_MS = 500
_WRITE_TIMEOUT_MS = 1000


def build_single_instance_server_name(app_id: str = "dagaza.qube") -> str:
    """Return a per-user local socket name so different OS users do not collide."""
    user = getpass.getuser() or "default"
    suffix = hashlib.sha256(user.encode("utf-8")).hexdigest()[:12]
    return f"{app_id}-{suffix}"


class SingleInstanceGuard(QObject):
    """Owns the primary-instance local socket server for this process."""

    def __init__(
        self,
        *,
        app_id: str = "dagaza.qube",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._server_name = build_single_instance_server_name(app_id)
        self._server = QLocalServer(self)
        self._activation_handler: Callable[[], None] | None = None
        self._owns_server = False

    @property
    def server_name(self) -> str:
        return self._server_name

    def set_activation_handler(self, handler: Callable[[], None] | None) -> None:
        self._activation_handler = handler

    def try_acquire(self) -> bool:
        """Return True when this process becomes the primary instance."""
        if self._notify_running_instance():
            logger.info("Another Qube instance is already running; exiting duplicate.")
            return False

        QLocalServer.removeServer(self._server_name)
        if not self._server.listen(self._server_name):
            logger.warning(
                "Could not bind single-instance server %s: %s",
                self._server_name,
                self._server.errorString(),
            )
            return True

        self._owns_server = True
        self._server.newConnection.connect(self._on_new_connection)
        logger.debug("Single-instance server listening on %s", self._server_name)
        return True

    def _notify_running_instance(self) -> bool:
        socket = QLocalSocket(self)
        socket.connectToServer(self._server_name)
        if not socket.waitForConnected(_CONNECT_TIMEOUT_MS):
            return False

        socket.write(_ACTIVATE_MESSAGE)
        socket.flush()
        socket.waitForBytesWritten(_WRITE_TIMEOUT_MS)
        socket.disconnectFromServer()
        return True

    def _on_new_connection(self) -> None:
        connection = self._server.nextPendingConnection()
        if connection is None:
            return
        connection.readyRead.connect(lambda: self._handle_activation(connection))

    def _handle_activation(self, connection: QLocalSocket) -> None:
        connection.readAll()
        connection.disconnectFromServer()
        logger.info("Duplicate launch detected; focusing existing Qube window.")
        handler = self._activation_handler
        if handler is not None:
            handler()
