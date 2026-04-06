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
        # check_same_thread=False allows our worker threads to safely query the DB
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Returns dict-like rows instead of tuples
        
        # Enforce foreign key constraints (SQLite has them off by default)
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

                # 3. FTS5 Virtual Table for blazing fast text search
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
                    USING fts5(content, content='messages', content_rowid='rowid')
                """)

                # 4. Triggers to keep FTS table synced with the messages table
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
                
                conn.commit()
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def create_session(self, title: str = "New Chat") -> str:
        """Creates a new chat session and returns its UUID."""
        session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)", 
                (session_id, title)
            )
            conn.commit()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Appends a message to a session and updates the session's timestamp."""
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
        """Retrieves all messages for a specific session, formatted for the LLM."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
            return [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]

    def search_history(self, query: str) -> list[dict]:
        """Performs a full-text search across all messages."""
        with self._get_connection() as conn:
            # We use the FTS MATCH operator. 
            # We append a wildcard '*' so partial words match (e.g., 'mark' matches 'marketing')
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

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        """Fetches the list of conversations for the UI Sidebar."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        
    def add_document_metadata(self, filename: str, file_size_kb: float, chunk_count: int):
        """Logs an ingested file so the Library UI can list it instantly."""
        doc_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, file_size_kb, chunk_count) VALUES (?, ?, ?, ?)",
                (doc_id, filename, file_size_kb, chunk_count)
            )
            conn.commit()

    def get_library_documents(self) -> list[dict]:
        """Fetches the registry for the Library Sidebar."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
            return [dict(row) for row in cursor.fetchall()]