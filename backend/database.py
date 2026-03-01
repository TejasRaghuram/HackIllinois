import aiosqlite
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger("hackillinois.database")

# Ensure the DB is created in the same directory as this file
DB_PATH = os.path.join(os.path.dirname(__file__), "calls.db")

async def init_db():
    """Initialize the SQLite database schema."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                phone_number TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                transcript TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Seed the database
        seed_data = [
            ("a1b2c3d4", "+14155551234", 1, "[]", "2026-02-28 10:15:30"),
            ("e5f6g7h8", "+12125559876", 0, '[{"role":"agent","text":"Hello, how can I help?"},{"role":"user","text":"I need to reschedule."}]', "2026-02-27 14:22:05"),
            ("i9j0k1l2", "+18035557654", 1, '[{"role":"agent","text":"Welcome! What can I do for you today?"}]', "2026-02-28 09:45:12")
        ]
        
        await db.executemany("""
            INSERT OR IGNORE INTO calls (call_id, phone_number, is_active, transcript, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, seed_data)
        
        await db.commit()
    logger.info("Database initialized and seeded.")

async def create_call(call_id: str, phone_number: str):
    """Create a new call record."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO calls (call_id, phone_number, is_active, transcript) VALUES (?, ?, 1, '[]')",
            (call_id, phone_number)
        )
        await db.commit()
    logger.info(f"Call created: {call_id} for {phone_number}")

async def set_call_inactive(call_id: str):
    """Mark a call as inactive."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE calls SET is_active = 0 WHERE call_id = ?", (call_id,))
        await db.commit()
    logger.debug(f"Call {call_id} marked inactive.")

async def append_transcript(call_id: str, role: str, text: str):
    """Append a message to the call's transcript."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT transcript FROM calls WHERE call_id = ?", (call_id,)) as cursor:
            row = await cursor.fetchone()
            
        if row:
            transcript_str = row["transcript"]
            try:
                transcript = json.loads(transcript_str) if transcript_str else []
            except json.JSONDecodeError:
                transcript = []
                
            transcript.append({"role": role, "text": text})
            new_transcript_str = json.dumps(transcript)
            
            await db.execute(
                "UPDATE calls SET transcript = ? WHERE call_id = ?",
                (new_transcript_str, call_id)
            )
            await db.commit()

async def get_call(call_id: str) -> Optional[dict]:
    """Retrieve a call by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None

async def get_all_calls() -> List[dict]:
    """Retrieve all calls ordered by newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM calls ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
