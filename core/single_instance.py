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
_ACK_MESSAGE = b"ok"
_CONNECT_TIMEOUT_MS = 500
_WRITE_TIMEOUT_MS = 1000
_ACK_TIMEOUT_MS = 750
_ACTIVATION_HANDLED_PROP = "_qube_activation_handled"

# Handler returns False to yield the lock (no visible UI / dying primary).
ActivationHandler = Callable[[], bool | None]


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
        self._activation_handler: ActivationHandler | None = None
        self._owns_server = False

    @property
    def server_name(self) -> str:
        return self._server_name

    def set_activation_handler(self, handler: ActivationHandler | None) -> None:
        self._activation_handler = handler

    def release(self) -> None:
        """Drop the single-instance bind so another process can become primary."""
        if self._owns_server:
            try:
                self._server.close()
            except Exception:
                logger.debug("Single-instance server close failed.", exc_info=True)
            self._owns_server = False
        QLocalServer.removeServer(self._server_name)

    def try_acquire(self) -> bool:
        """Return True when this process becomes the primary instance."""
        if self._notify_running_instance():
            logger.info("Another Qube instance is already running; exiting duplicate.")
            return False

        # Free or stale socket (crash / Task Manager kill / yielding zombie).
        QLocalServer.removeServer(self._server_name)
        if not self._server.listen(self._server_name):
            logger.warning(
                "Could not bind single-instance server %s (%s); retrying after cleanup.",
                self._server_name,
                self._server.errorString(),
            )
            QLocalServer.removeServer(self._server_name)
            if not self._server.listen(self._server_name):
                logger.warning(
                    "Could not bind single-instance server %s: %s",
                    self._server_name,
                    self._server.errorString(),
                )
                # Prefer launching over a silent no-op when the lock is broken.
                return True

        self._owns_server = True
        self._server.newConnection.connect(self._on_new_connection)
        logger.debug("Single-instance server listening on %s", self._server_name)
        return True

    def _notify_running_instance(self) -> bool:
        """Return True only when a live primary accepts activation (ACK)."""
        from PyQt6.QtCore import QEventLoop, QTimer

        socket = QLocalSocket(self)
        socket.connectToServer(self._server_name)
        if not socket.waitForConnected(_CONNECT_TIMEOUT_MS):
            return False

        # Do not treat waitForBytesWritten failure as fatal: on some platforms the
        # nested event loop inside that wait already completes the primary ACK
        # handshake, and aborting afterward drops a successful activation.
        socket.write(_ACTIVATE_MESSAGE)
        socket.flush()
        socket.waitForBytesWritten(_WRITE_TIMEOUT_MS)

        # Stale named pipes (common after hard kill on Windows) accept connects
        # but never ACK. Treat that as "no live primary" so we can take over.
        #
        # Use a nested event loop (not only waitForReadyRead) so an in-process
        # primary on the same thread can handle newConnection and write the ACK
        # — required for unit tests and rare same-thread edge cases.
        if socket.bytesAvailable() <= 0:
            loop = QEventLoop(socket)
            socket.readyRead.connect(loop.quit)
            socket.disconnected.connect(loop.quit)
            QTimer.singleShot(_ACK_TIMEOUT_MS, loop.quit)
            loop.exec()

        payload = bytes(socket.readAll())
        if socket.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            socket.disconnectFromServer()
        return _ACK_MESSAGE in payload

    def _on_new_connection(self) -> None:
        connection = self._server.nextPendingConnection()
        if connection is None:
            return
        self._wait_for_activation_payload(connection)
        self._handle_activation(connection)

    def _wait_for_activation_payload(self, connection: QLocalSocket) -> None:
        """Block briefly until the duplicate client sends its activate payload."""
        if connection.bytesAvailable() > 0:
            return
        if connection.state() == QLocalSocket.LocalSocketState.ConnectedState:
            connection.waitForReadyRead(_WRITE_TIMEOUT_MS)
        if (
            connection.bytesAvailable() <= 0
            and connection.state() == QLocalSocket.LocalSocketState.ConnectedState
        ):
            connection.waitForDisconnected(_CONNECT_TIMEOUT_MS)

    def _handle_activation(self, connection: QLocalSocket) -> None:
        if connection.property(_ACTIVATION_HANDLED_PROP):
            return

        connection.setProperty(_ACTIVATION_HANDLED_PROP, True)
        if connection.bytesAvailable() > 0:
            connection.readAll()

        accepted = True
        handler = self._activation_handler
        if handler is not None:
            try:
                result = handler()
                if result is False:
                    accepted = False
            except Exception:
                logger.exception("Activation handler failed.")
                accepted = False

        if accepted:
            logger.info("Duplicate launch detected; focusing existing Qube window.")
            connection.write(_ACK_MESSAGE)
            connection.flush()
            connection.waitForBytesWritten(_WRITE_TIMEOUT_MS)
        else:
            logger.info(
                "Duplicate launch detected; primary has no visible UI and is yielding."
            )

        if connection.state() != QLocalSocket.LocalSocketState.UnconnectedState:
            connection.disconnectFromServer()
