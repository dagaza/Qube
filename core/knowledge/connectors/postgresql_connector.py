"""PostgreSQL connector — read-only parameterized queries (enterprise)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Connectors.PostgreSQL")


class PostgreSQLConnector:
    id = "postgresql"
    trust_policy = "enterprise"

    def execute(
        self,
        query: str,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        _ = egress_policy
        sql_template = str(config.get("sql_template") or "").strip()
        title_column = str(config.get("title_column") or "title")
        snippet_column = str(config.get("snippet_column") or "snippet")
        url_column = str(config.get("url_column") or "url")
        adapter_id = str(config.get("adapter_id") or "configured_postgresql")
        dsn = self._resolve_dsn(config, auth)
        if not dsn or ":query" not in sql_template:
            return []

        try:
            import psycopg2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("[PostgreSQL] psycopg2 not installed")
            return []

        rows: list[dict[str, Any]] = []
        try:
            conn = psycopg2.connect(dsn, connect_timeout=int(timeout))
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor() as cur:
                cur.execute(sql_template, {"query": f"%{query}%"})
                fetched = cur.fetchmany(max(1, max_results))
                colnames = [d[0] for d in cur.description or []]
            conn.close()
        except Exception as exc:
            logger.warning("[PostgreSQL] query failed: %s", exc)
            return []

        for record in fetched:
            data = dict(zip(colnames, record))
            title = str(data.get(title_column) or "").strip()
            snippet = str(data.get(snippet_column) or "").strip()
            url = data.get(url_column)
            url = str(url).strip() if url else None
            if not title and not snippet:
                continue
            rows.append(
                {
                    "title": title or snippet[:120],
                    "snippet": snippet[:600],
                    "full_text": None,
                    "url": url,
                    "_adapter": adapter_id,
                    "retrieval_method": "postgresql",
                }
            )
        return rows

    def _resolve_dsn(self, config: dict[str, Any], auth: dict[str, Any] | None) -> str | None:
        from core.knowledge.secret_store import resolve_secret

        ref = ""
        if auth:
            ref = str(auth.get("credential_ref") or "").strip()
        if ref:
            secret = resolve_secret(ref if ref.startswith("configured:") else f"configured:{ref}")
            if secret:
                return secret
        return str(config.get("dsn") or "").strip() or None

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        _ = egress_policy
        dsn = self._resolve_dsn(config, auth)
        if not dsn:
            return False, "Connection string not configured"
        try:
            import psycopg2  # type: ignore[import-untyped]

            conn = psycopg2.connect(dsn, connect_timeout=int(timeout))
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return True, "OK — database reachable (read-only)"
        except ImportError:
            return False, "psycopg2 is not installed"
        except Exception as exc:
            return False, str(exc)
