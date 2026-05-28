import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
import logging

from core.library_folder_policy import (
    FOLDER_KEY_MAIN,
    FOLDER_KEY_QUBE,
    MAIN_FOLDER_DISPLAY_NAME,
    QUBE_FOLDER_DISPLAY_NAME,
    RESERVED_LIBRARY_FOLDER_NAMES,
    is_qube_managed_document_filename,
)

logger = logging.getLogger("Qube.Database")

_MAIN_FOLDER_NAME = MAIN_FOLDER_DISPLAY_NAME
_QUBE_FOLDER_NAME = QUBE_FOLDER_DISPLAY_NAME


class DatabaseManager:
    def __init__(self, db_path: str = "qube_data.db"):
        self.db_path = Path(db_path)
        self.init_db()

    def _get_connection(self):
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row 
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Creates the tables and FTS search index if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 1. Sessions Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 2. Messages Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    )
                """)

                # 3. FTS5 Virtual Table
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
                    USING fts5(content, content='messages', content_rowid='rowid')
                """)

                # 4. Triggers to keep FTS table synced
                cursor.executescript("""
                    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                    END;
                    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
                        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                    END;
                """)

                # 5. Document Library Registry
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_size_kb REAL NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 🔑 6. THE NEW RAG TRIGGERS TABLE
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rag_triggers (
                        id TEXT PRIMARY KEY,
                        phrase TEXT UNIQUE NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 🔑 7. SEED DEFAULT TRIGGERS (If table is empty)
                cursor.execute("SELECT COUNT(*) FROM rag_triggers")
                if cursor.fetchone()[0] == 0:
                    default_triggers = [
                        "search my files",
                        "in my documents",
                        "according to my knowledge base",
                        "based on my files",
                        "check my library"
                    ]
                    for trigger in default_triggers:
                        cursor.execute(
                            "INSERT INTO rag_triggers (id, phrase) VALUES (?, ?)",
                            (str(uuid.uuid4()), trigger)
                        )
                    logger.info("Seeded default RAG triggers into database.")

                # 8. Sidebar folder tables (Conversations + Library)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_folders (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        is_collapsed INTEGER NOT NULL DEFAULT 0,
                        is_system INTEGER NOT NULL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS library_folders (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        is_collapsed INTEGER NOT NULL DEFAULT 0,
                        is_system INTEGER NOT NULL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully.")

                # Message-level RAG / memory citation payloads (JSON list of source dicts)
                try:
                    cursor.execute("ALTER TABLE messages ADD COLUMN sources_json TEXT")
                    logger.info("Added messages.sources_json column.")
                except sqlite3.OperationalError:
                    pass

                for alter_sql in (
                    "ALTER TABLE sessions ADD COLUMN folder_id TEXT REFERENCES conversation_folders(id)",
                    "ALTER TABLE documents ADD COLUMN folder_id TEXT REFERENCES library_folders(id)",
                    "ALTER TABLE documents ADD COLUMN summary_blurb TEXT",
                    "ALTER TABLE library_folders ADD COLUMN folder_key TEXT",
                    "ALTER TABLE library_folders ADD COLUMN allows_user_ingest INTEGER NOT NULL DEFAULT 1",
                ):
                    try:
                        cursor.execute(alter_sql)
                        conn.commit()
                    except sqlite3.OperationalError:
                        pass

                self._ensure_main_folders_and_backfill(conn)
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _ensure_main_folders_and_backfill(self, conn: sqlite3.Connection) -> None:
        """Seed system folders when missing and backfill NULL folder_id on legacy rows."""
        cursor = conn.cursor()
        conv_main = self._ensure_main_folder_row(
            conn, "conversation_folders", _MAIN_FOLDER_NAME
        )
        lib_main, lib_qube = self._ensure_library_system_folders(conn)
        cursor.execute(
            "UPDATE sessions SET folder_id = ? WHERE folder_id IS NULL",
            (conv_main,),
        )
        cursor.execute(
            "UPDATE documents SET folder_id = ? WHERE folder_id IS NULL",
            (lib_main,),
        )
        self._migrate_qube_managed_documents_to_qube_folder(conn, lib_main, lib_qube)
        conn.commit()

    def _ensure_main_folder_row(
        self, conn: sqlite3.Connection, table: str, name: str
    ) -> str:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id FROM {table} WHERE is_system = 1 AND name = ? LIMIT 1",
            (name,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        cursor.execute(f"SELECT id FROM {table} WHERE is_system = 1 LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        if cursor.fetchone()[0] == 0:
            folder_id = str(uuid.uuid4())
            cursor.execute(
                f"""
                INSERT INTO {table}
                    (id, name, sort_order, is_collapsed, is_system)
                VALUES (?, ?, 0, 0, 1)
                """,
                (folder_id, name),
            )
            conn.commit()
            return folder_id
        cursor.execute(
            f"SELECT id FROM {table} ORDER BY sort_order, created_at LIMIT 1"
        )
        fallback = cursor.fetchone()
        return fallback[0] if fallback else str(uuid.uuid4())

    def _ensure_library_system_folders(
        self, conn: sqlite3.Connection
    ) -> tuple[str, str]:
        """Ensure Main + Qube library folders; return (main_id, qube_id)."""
        main_id = self._ensure_library_folder_by_key(
            conn,
            folder_key=FOLDER_KEY_MAIN,
            name=_MAIN_FOLDER_NAME,
            sort_order=0,
            allows_user_ingest=True,
        )
        qube_id = self._ensure_library_folder_by_key(
            conn,
            folder_key=FOLDER_KEY_QUBE,
            name=_QUBE_FOLDER_NAME,
            sort_order=1,
            allows_user_ingest=False,
        )
        return main_id, qube_id

    def _ensure_library_folder_by_key(
        self,
        conn: sqlite3.Connection,
        *,
        folder_key: str,
        name: str,
        sort_order: int,
        allows_user_ingest: bool,
    ) -> str:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM library_folders WHERE folder_key = ? LIMIT 1",
            (folder_key,),
        )
        row = cursor.fetchone()
        if row:
            folder_id = row[0]
            cursor.execute(
                """
                UPDATE library_folders
                SET name = ?, is_system = 1, sort_order = ?, allows_user_ingest = ?
                WHERE id = ?
                """,
                (name, sort_order, 1 if allows_user_ingest else 0, folder_id),
            )
            return folder_id
        cursor.execute(
            "SELECT id FROM library_folders WHERE is_system = 1 AND name = ? LIMIT 1",
            (name,),
        )
        row = cursor.fetchone()
        if row:
            folder_id = row[0]
            cursor.execute(
                """
                UPDATE library_folders
                SET folder_key = ?, is_system = 1, sort_order = ?, allows_user_ingest = ?
                WHERE id = ?
                """,
                (folder_key, sort_order, 1 if allows_user_ingest else 0, folder_id),
            )
            return folder_id
        folder_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT INTO library_folders
                (id, name, sort_order, is_collapsed, is_system, folder_key, allows_user_ingest)
            VALUES (?, ?, ?, 0, 1, ?, ?)
            """,
            (
                folder_id,
                name,
                sort_order,
                folder_key,
                1 if allows_user_ingest else 0,
            ),
        )
        return folder_id

    def _migrate_qube_managed_documents_to_qube_folder(
        self,
        conn: sqlite3.Connection,
        main_id: str,
        qube_id: str,
    ) -> None:
        cursor = conn.execute("SELECT filename, folder_id FROM documents")
        for row in cursor.fetchall():
            if not is_qube_managed_document_filename(row["filename"]):
                continue
            if row["folder_id"] != main_id:
                continue
            conn.execute(
                "UPDATE documents SET folder_id = ? WHERE filename = ?",
                (qube_id, row["filename"]),
            )

    def _get_library_folder_id_by_key(self, folder_key: str) -> str:
        with self._get_connection() as conn:
            folder_id = self._ensure_library_folder_by_key(
                conn,
                folder_key=folder_key,
                name=_MAIN_FOLDER_NAME
                if folder_key == FOLDER_KEY_MAIN
                else _QUBE_FOLDER_NAME,
                sort_order=0 if folder_key == FOLDER_KEY_MAIN else 1,
                allows_user_ingest=folder_key == FOLDER_KEY_MAIN,
            )
            conn.commit()
            return folder_id

    def get_library_folder(self, folder_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, name, sort_order, is_collapsed, is_system, created_at,
                       folder_key, allows_user_ingest
                FROM library_folders WHERE id = ?
                """,
                (folder_id,),
            ).fetchone()
            return self._folder_row_to_dict(row) if row else None

    def library_folder_allows_user_ingest(self, folder_id: str) -> bool:
        folder = self.get_library_folder(folder_id)
        if folder is None:
            return True
        return bool(folder.get("allows_user_ingest", True))

    def _folder_row_to_dict(self, row: sqlite3.Row) -> dict:
        keys = row.keys()
        allows_user_ingest = True
        if "allows_user_ingest" in keys:
            allows_user_ingest = bool(row["allows_user_ingest"])
        elif row["is_system"] and row["name"] == _QUBE_FOLDER_NAME:
            allows_user_ingest = False
        return {
            "id": row["id"],
            "name": row["name"],
            "sort_order": row["sort_order"],
            "is_collapsed": bool(row["is_collapsed"]),
            "is_system": bool(row["is_system"]),
            "created_at": row["created_at"],
            "folder_key": row["folder_key"] if "folder_key" in keys else None,
            "allows_user_ingest": allows_user_ingest,
        }

    def get_main_conversation_folder_id(self) -> str:
        with self._get_connection() as conn:
            return self._ensure_main_folder_row(
                conn, "conversation_folders", _MAIN_FOLDER_NAME
            )

    def get_main_library_folder_id(self) -> str:
        return self._get_library_folder_id_by_key(FOLDER_KEY_MAIN)

    def get_qube_library_folder_id(self) -> str:
        return self._get_library_folder_id_by_key(FOLDER_KEY_QUBE)

    def list_conversation_folders(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, sort_order, is_collapsed, is_system, created_at
                FROM conversation_folders
                ORDER BY sort_order ASC, created_at ASC
                """
            )
            return [self._folder_row_to_dict(row) for row in cursor.fetchall()]

    def list_library_folders(self) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, sort_order, is_collapsed, is_system, created_at,
                       folder_key, allows_user_ingest
                FROM library_folders
                ORDER BY sort_order ASC, created_at ASC
                """
            )
            return [self._folder_row_to_dict(row) for row in cursor.fetchall()]

    def list_user_ingest_library_folders(self) -> list[dict]:
        """Library folders the user may target for manual ingestion or moves."""
        return [
            f
            for f in self.list_library_folders()
            if f.get("allows_user_ingest", True)
        ]

    def create_conversation_folder(self, name: str) -> str | None:
        clean = (name or "").strip()
        if not clean:
            return None
        folder_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM conversation_folders"
            ).fetchone()
            sort_order = int(row[0]) if row else 0
            conn.execute(
                """
                INSERT INTO conversation_folders
                    (id, name, sort_order, is_collapsed, is_system)
                VALUES (?, ?, ?, 0, 0)
                """,
                (folder_id, clean, sort_order),
            )
            conn.commit()
        return folder_id

    def create_library_folder(self, name: str) -> str | None:
        clean = (name or "").strip()
        if not clean:
            return None
        if clean.casefold() in RESERVED_LIBRARY_FOLDER_NAMES:
            return None
        folder_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM library_folders"
            ).fetchone()
            sort_order = int(row[0]) if row else 0
            conn.execute(
                """
                INSERT INTO library_folders
                    (id, name, sort_order, is_collapsed, is_system)
                VALUES (?, ?, ?, 0, 0)
                """,
                (folder_id, clean, sort_order),
            )
            conn.commit()
        return folder_id

    def rename_conversation_folder(self, folder_id: str, name: str) -> bool:
        clean = (name or "").strip()
        if not clean:
            return False
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE conversation_folders SET name = ? WHERE id = ?",
                    (clean, folder_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to rename conversation folder %s: %s", folder_id, e)
            return False

    def rename_library_folder(self, folder_id: str, name: str) -> bool:
        clean = (name or "").strip()
        if not clean:
            return False
        if clean.casefold() in RESERVED_LIBRARY_FOLDER_NAMES:
            return False
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT is_system FROM library_folders WHERE id = ?",
                    (folder_id,),
                ).fetchone()
                if row is None or row["is_system"]:
                    return False
                conn.execute(
                    "UPDATE library_folders SET name = ? WHERE id = ?",
                    (clean, folder_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to rename library folder %s: %s", folder_id, e)
            return False

    def set_conversation_folder_collapsed(self, folder_id: str, collapsed: bool) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE conversation_folders SET is_collapsed = ? WHERE id = ?",
                    (1 if collapsed else 0, folder_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(
                "Failed to set conversation folder collapsed %s: %s", folder_id, e
            )
            return False

    def set_library_folder_collapsed(self, folder_id: str, collapsed: bool) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE library_folders SET is_collapsed = ? WHERE id = ?",
                    (1 if collapsed else 0, folder_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(
                "Failed to set library folder collapsed %s: %s", folder_id, e
            )
            return False

    def delete_conversation_folder(self, folder_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT is_system FROM conversation_folders WHERE id = ?",
                    (folder_id,),
                ).fetchone()
                if row is None:
                    return False
                if row["is_system"]:
                    return False
                conn.execute("DELETE FROM sessions WHERE folder_id = ?", (folder_id,))
                conn.execute(
                    "DELETE FROM conversation_folders WHERE id = ?", (folder_id,)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to delete conversation folder %s: %s", folder_id, e)
            return False

    def delete_library_folder(self, folder_id: str) -> tuple[bool, list[str]]:
        """Delete folder and return (success, list of filenames removed from SQLite)."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT is_system FROM library_folders WHERE id = ?",
                    (folder_id,),
                ).fetchone()
                if row is None:
                    return False, []
                if row["is_system"]:
                    return False, []
                cursor = conn.execute(
                    "SELECT filename FROM documents WHERE folder_id = ?",
                    (folder_id,),
                )
                filenames = [r["filename"] for r in cursor.fetchall()]
                conn.execute("DELETE FROM documents WHERE folder_id = ?", (folder_id,))
                conn.execute("DELETE FROM library_folders WHERE id = ?", (folder_id,))
                conn.commit()
            return True, filenames
        except Exception as e:
            logger.error("Failed to delete library folder %s: %s", folder_id, e)
            return False, []

    def move_session_to_folder(self, session_id: str, folder_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM conversation_folders WHERE id = ?",
                    (folder_id,),
                ).fetchone()
                if not exists:
                    return False
                conn.execute(
                    "UPDATE sessions SET folder_id = ? WHERE id = ?",
                    (folder_id, session_id),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(
                "Failed to move session %s to folder %s: %s", session_id, folder_id, e
            )
            return False

    def move_document_to_folder(self, filename: str, folder_id: str) -> bool:
        if not self.library_folder_allows_user_ingest(folder_id):
            return False
        try:
            with self._get_connection() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM library_folders WHERE id = ?",
                    (folder_id,),
                ).fetchone()
                if not exists:
                    return False
                conn.execute(
                    "UPDATE documents SET folder_id = ? WHERE filename = ?",
                    (folder_id, filename),
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(
                "Failed to move document %s to folder %s: %s", filename, folder_id, e
            )
            return False

    def get_sessions_for_sidebar_by_folder(self) -> tuple[list[dict], dict[str, list[dict]]]:
        with self._get_connection() as conn:
            main_id = self._ensure_main_folder_row(
                conn, "conversation_folders", _MAIN_FOLDER_NAME
            )
            folders = [
                self._folder_row_to_dict(row)
                for row in conn.execute(
                    """
                    SELECT id, name, sort_order, is_collapsed, is_system, created_at
                    FROM conversation_folders
                    ORDER BY sort_order ASC, created_at ASC
                    """
                ).fetchall()
            ]
            grouped: dict[str, list[dict]] = {f["id"]: [] for f in folders}
            cursor = conn.execute(
                """
                SELECT id, title, updated_at, folder_id
                FROM sessions
                ORDER BY updated_at DESC
                """
            )
            for row in cursor.fetchall():
                fid = row["folder_id"] or main_id
                if fid not in grouped:
                    grouped[fid] = []
                grouped[fid].append(dict(row))
            return folders, grouped

    def get_documents_for_sidebar_by_folder(self) -> tuple[list[dict], dict[str, list[dict]]]:
        with self._get_connection() as conn:
            main_id, _qube_id = self._ensure_library_system_folders(conn)
            folders = [
                self._folder_row_to_dict(row)
                for row in conn.execute(
                    """
                    SELECT id, name, sort_order, is_collapsed, is_system, created_at,
                           folder_key, allows_user_ingest
                    FROM library_folders
                    ORDER BY sort_order ASC, created_at ASC
                    """
                ).fetchall()
            ]
            grouped: dict[str, list[dict]] = {f["id"]: [] for f in folders}
            cursor = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            )
            for row in cursor.fetchall():
                doc = dict(row)
                fid = doc.get("folder_id") or main_id
                if fid not in grouped:
                    grouped[fid] = []
                grouped[fid].append(doc)
            return folders, grouped

    def get_session_folder_id(self, session_id: str) -> str | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT folder_id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return row["folder_id"] or self.get_main_conversation_folder_id()

    def get_session(self, session_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at, folder_id
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_sessions_in_folder(self, folder_id: str) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, created_at, updated_at, folder_id
                FROM sessions
                WHERE folder_id = ?
                ORDER BY updated_at DESC
                """,
                (folder_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # ... [Keep your existing methods: cleanup_empty_sessions, get_session_count, etc.] ...
    def cleanup_empty_sessions(self, active_session_id: str = None):
        try:
            with self._get_connection() as conn:
                if active_session_id:
                    conn.execute("""
                        DELETE FROM sessions 
                        WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)
                        AND id != ?
                    """, (active_session_id,))
                else:
                    conn.execute("""
                        DELETE FROM sessions 
                        WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)
                    """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup empty sessions: {e}")

    def get_session_count(self) -> int:
        try:
            with self._get_connection() as conn:
                return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        except Exception:
            return 0

    def get_document_count(self) -> int:
        try:
            with self._get_connection() as conn:
                return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        except Exception:
            return 0

    def create_session(
        self, title: str = "New Chat", folder_id: str | None = None
    ) -> str:
        session_id = str(uuid.uuid4())
        fid = folder_id or self.get_main_conversation_folder_id()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, folder_id) VALUES (?, ?, ?)",
                (session_id, title, fid),
            )
            conn.commit()
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources_json: str | None = None,
    ) -> str:
        """Insert a message and return its generated id.

        The id is used by the memory enrichment pipeline to record the exact
        source message(s) for each extracted fact (``source_message_ids`` on
        the LanceDB payload). Existing callers that ignore the return value
        are unaffected.
        """
        msg_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (id, session_id, role, content, sources_json) VALUES (?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, sources_json),
            )
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()
        return msg_id

    def get_session_history(self, session_id: str) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, role, content, sources_json FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            rows = []
            for row in cursor.fetchall():
                entry = {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                }
                raw = row["sources_json"]
                if raw:
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            entry["sources"] = parsed
                    except json.JSONDecodeError:
                        logger.warning("Bad sources_json for session %s", session_id)
                rows.append(entry)
            return rows

    def search_history(self, query: str) -> list[dict]:
        with self._get_connection() as conn:
            safe_query = f"{query}*"
            cursor = conn.execute(
                """
                SELECT m.session_id, s.title, m.role,
                       snippet(fts, -1, '<b>', '</b>', '...', 10) AS highlight
                FROM messages_fts AS fts
                JOIN messages m ON fts.rowid = m.rowid
                JOIN sessions s ON m.session_id = s.id
                WHERE fts MATCH ?
                ORDER BY m.timestamp DESC
                LIMIT 20
                """,
                (safe_query,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_sessions(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, title, updated_at, folder_id
                FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_sessions_for_sidebar_search(self, query: str, limit: int = 200) -> list[dict]:
        """Sessions whose title or any message body matches query (case-insensitive substring)."""
        q = (query or "").strip()
        if not q:
            return []
        ql = q.lower()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.id, s.title, s.updated_at, s.folder_id, cf.name AS folder_name
                FROM sessions s
                LEFT JOIN conversation_folders cf ON cf.id = s.folder_id
                WHERE instr(lower(s.title), ?) > 0
                UNION
                SELECT s.id, s.title, s.updated_at, s.folder_id, cf.name AS folder_name
                FROM sessions s
                INNER JOIN messages m ON m.session_id = s.id
                LEFT JOIN conversation_folders cf ON cf.id = s.folder_id
                WHERE instr(lower(m.content), ?) > 0
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (ql, ql, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        
    def add_document_metadata(
        self,
        filename: str,
        file_size_kb: float,
        chunk_count: int,
        folder_id: str | None = None,
        summary_blurb: str | None = None,
    ):
        doc_id = str(uuid.uuid4())
        if folder_id:
            fid = folder_id
        elif is_qube_managed_document_filename(filename):
            fid = self.get_qube_library_folder_id()
        else:
            fid = self.get_main_library_folder_id()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, file_size_kb, chunk_count, folder_id, summary_blurb)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, file_size_kb, chunk_count, fid, summary_blurb),
            )
            conn.commit()

    def update_document_blurb(self, filename: str, summary_blurb: str) -> bool:
        blurb = (summary_blurb or "").strip()
        if not filename or not blurb:
            return False
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE documents SET summary_blurb = ? WHERE filename = ?",
                (blurb[:500], filename),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_library_documents(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_library_documents_for_sidebar_search(
        self,
        query: str,
        content_match_filenames: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Documents whose filename matches query and/or appear in content_match_filenames (e.g. RAG chunk hits)."""
        q = (query or "").strip().lower()
        names = list(dict.fromkeys(content_match_filenames or []))[:400]
        if not q and not names:
            return []
        with self._get_connection() as conn:
            if q and names:
                ph = ",".join("?" * len(names))
                cursor = conn.execute(
                    f"""
                    SELECT d.*, lf.name AS folder_name
                    FROM documents d
                    LEFT JOIN library_folders lf ON lf.id = d.folder_id
                    WHERE instr(lower(d.filename), ?) > 0 OR d.filename IN ({ph})
                    ORDER BY d.ingested_at DESC
                    LIMIT ?
                    """,
                    (q, *names, limit),
                )
            elif q:
                cursor = conn.execute(
                    """
                    SELECT d.*, lf.name AS folder_name
                    FROM documents d
                    LEFT JOIN library_folders lf ON lf.id = d.folder_id
                    WHERE instr(lower(d.filename), ?) > 0
                    ORDER BY d.ingested_at DESC
                    LIMIT ?
                    """,
                    (q, limit),
                )
            else:
                ph = ",".join("?" * len(names))
                cursor = conn.execute(
                    f"""
                    SELECT d.*, lf.name AS folder_name
                    FROM documents d
                    LEFT JOIN library_folders lf ON lf.id = d.folder_id
                    WHERE d.filename IN ({ph})
                    ORDER BY d.ingested_at DESC
                    LIMIT ?
                    """,
                    (*names, limit),
                )
            return [dict(row) for row in cursor.fetchall()]

    def delete_document_metadata(self, filename: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()

    def rename_document_metadata(self, old_filename: str, new_filename: str) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE documents SET filename = ? WHERE filename = ?", 
                    (new_filename, old_filename)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to rename document metadata {old_filename}: {e}")
            return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", 
                    (new_title, session_id)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to rename session {session_id}: {e}")
            return False

    def delete_session(self, session_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    # --------------------------------------------------------- #
    #  🔑 NEW RAG TRIGGER METHODS                              #
    # --------------------------------------------------------- #

    def get_rag_triggers(self) -> list[str]:
        """Retrieves all custom RAG trigger phrases."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT phrase FROM rag_triggers ORDER BY created_at ASC")
                return [row["phrase"] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get RAG triggers: {e}")
            return []

    def add_rag_trigger(self, phrase: str) -> bool:
        """Adds a new trigger phrase if it doesn't already exist."""
        try:
            clean_phrase = phrase.strip().lower()
            if not clean_phrase:
                return False
                
            with self._get_connection() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO rag_triggers (id, phrase) VALUES (?, ?)",
                    (str(uuid.uuid4()), clean_phrase)
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to add RAG trigger '{phrase}': {e}")
            return False

    def remove_rag_trigger(self, phrase: str) -> bool:
        """Removes a trigger phrase."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM rag_triggers WHERE phrase = ?", (phrase.strip().lower(),))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to remove RAG trigger '{phrase}': {e}")
            return False