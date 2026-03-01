import aiosqlite
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger("hackillinois.database")

# Ensure the DB is created in the same directory as this file, regardless of where the app is launched from
DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")

async def init_db():
    """Initialize the SQLite database schema."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                phone_number TEXT,
                caller_name TEXT,
                address TEXT,
                transcript TEXT,
                actions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        # Safe migration for existing DBs
        try:
            await db.execute("ALTER TABLE sessions ADD COLUMN is_active BOOLEAN DEFAULT 1")
        except Exception:
            pass
        await db.commit()
    logger.info("Database initialized.")

async def update_session_active(session_id: str, is_active: bool):
    """Update the active status of a session."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE sessions SET is_active = ? WHERE session_id = ?", (is_active, session_id))
        await db.commit()
    logger.debug(f"Updated is_active to {is_active} for session {session_id}")

async def create_session(session_id: str, phone_number: str):
    """Create a new session record."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, phone_number, transcript, actions, is_active) VALUES (?, ?, ?, ?, ?)",
            (session_id, phone_number, "[]", "[]", True)
        )
        await db.commit()
    logger.info(f"Session created: {session_id} for {phone_number}")

async def get_session(session_id: str) -> Optional[dict]:
    """Retrieve a session by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

async def update_transcript(session_id: str, transcript: List[dict]):
    """Update the transcript for a session."""
    transcript_json = json.dumps(transcript)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET transcript = ? WHERE session_id = ?",
            (transcript_json, session_id)
        )
        await db.commit()

async def update_extracted_info(session_id: str, caller_name: Optional[str] = None, address: Optional[str] = None, actions: Optional[List[str]] = None):
    """Update extracted information for a session."""
    async with aiosqlite.connect(DB_PATH) as db:
        if caller_name is not None:
            await db.execute("UPDATE sessions SET caller_name = ? WHERE session_id = ?", (caller_name, session_id))
        if address is not None:
            await db.execute("UPDATE sessions SET address = ? WHERE session_id = ?", (address, session_id))
        if actions is not None:
            actions_json = json.dumps(actions)
            await db.execute("UPDATE sessions SET actions = ? WHERE session_id = ?", (actions_json, session_id))
        await db.commit()
    logger.debug(f"Updated info for session {session_id}")

async def get_all_sessions() -> List[dict]:
    """Retrieve all sessions ordered by newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
