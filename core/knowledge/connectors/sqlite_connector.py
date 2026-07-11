"""SQLite connector — parameterized read-only queries."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.Knowledge.Connectors.SQLite")


class SqliteConnector:
    id = "sqlite"

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
        _ = auth, egress_policy, timeout
        db_path = Path(str(config.get("database_path") or "")).expanduser()
        sql_template = str(config.get("sql_template") or "").strip()
        title_column = str(config.get("title_column") or "title")
        snippet_column = str(config.get("snippet_column") or "snippet")
        url_column = str(config.get("url_column") or "url")
        adapter_id = str(config.get("adapter_id") or "configured_sqlite")

        if not db_path.is_file() or not sql_template:
            return []
        if ":query" not in sql_template:
            return []

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql_template, {"query": f"%{query}%"})
            fetched = cur.fetchmany(max(1, max_results))
            conn.close()
        except Exception as exc:
            logger.warning("[SQLite] query failed: %s", exc)
            return []

        rows: list[dict[str, Any]] = []
        for record in fetched:
            data = dict(record)
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
                    "retrieval_method": "sqlite",
                }
            )
        return rows

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        _ = auth, egress_policy, timeout
        db_path = Path(str(config.get("database_path") or "")).expanduser()
        if not db_path.is_file():
            return False, f"Database not found: {db_path}"
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("SELECT 1")
            conn.close()
            return True, "OK — database readable"
        except Exception as exc:
            return False, str(exc)
