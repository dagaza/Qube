"""Connector type protocol and registry."""

from __future__ import annotations

from typing import Any, Protocol


class ConnectorType(Protocol):
    id: str

    def execute(
        self,
        query: str,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]: ...

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]: ...


_CONNECTORS: dict[str, ConnectorType] = {}


def register_connector(connector: ConnectorType) -> None:
    _CONNECTORS[connector.id] = connector


def get_connector(connector_type: str) -> ConnectorType | None:
    return _CONNECTORS.get((connector_type or "").strip().lower())


def list_connector_types() -> list[str]:
    return sorted(_CONNECTORS.keys())


def _register_builtin_connectors() -> None:
    from core.knowledge.connectors.rest_json import RestJsonConnector
    from core.knowledge.connectors.rss_atom import RssAtomConnector
    from core.knowledge.connectors.sqlite_connector import SqliteConnector
    from core.knowledge.connectors.filesystem_connector import FilesystemConnector
    from core.knowledge.connectors.postgresql_connector import PostgreSQLConnector
    from core.knowledge.connectors.mcp_connector import McpConnector
    from core.knowledge.connectors.graphql_connector import GraphQLConnector

    for connector in (
        RestJsonConnector(),
        RssAtomConnector(),
        SqliteConnector(),
        FilesystemConnector(),
        PostgreSQLConnector(),
        McpConnector(),
        GraphQLConnector(),
    ):
        register_connector(connector)


_register_builtin_connectors()
