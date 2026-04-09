import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger("Qube.Database")

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
                
                conn.commit()
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

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

    def create_session(self, title: str = "New Chat") -> str:
        session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)", 
                (session_id, title)
            )
            conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        msg_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (msg_id, session_id, role, content)
            )
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,)
            )
            conn.commit()

    def get_session_history(self, session_id: str) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    def search_history(self, query: str) -> list[dict]:
        with self._get_connection() as conn:
            safe_query = f"{query}*" 
            cursor = conn.execute("""
                SELECT m.session_id, s.title, m.role, snippet(messages_fts, -1, '<b>', '</b>', '...', 10) as highlight
                FROM messages_fts fts
                JOIN messages m ON fts.rowid = m.rowid
                JOIN sessions s ON m.session_id = s.id
                WHERE messages_fts MATCH ?
                ORDER BY m.timestamp DESC
                LIMIT 20
            """, (safe_query,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_sessions(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]
        
    def add_document_metadata(self, filename: str, file_size_kb: float, chunk_count: int):
        doc_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, file_size_kb, chunk_count) VALUES (?, ?, ?, ?)",
                (doc_id, filename, file_size_kb, chunk_count)
            )
            conn.commit()

    def get_library_documents(self, limit: int = 20, offset: int = 0) -> list[dict]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
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
                conn.execute(
                    "INSERT OR IGNORE INTO rag_triggers (id, phrase) VALUES (?, ?)",
                    (str(uuid.uuid4()), clean_phrase)
                )
                conn.commit()
                return True
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