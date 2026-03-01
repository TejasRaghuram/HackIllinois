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
        
        # Seed the database with realistic 911 calls
        seed_data = [
            (
                "call_2026_001", 
                "+13125550199", 
                0, 
                json.dumps([
                    {"role": "agent", "text": "911, what is your emergency?"},
                    {"role": "user", "text": "I'm at the corner of Green and Wright. There's a car accident, two vehicles involved."},
                    {"role": "agent", "text": "Are there any injuries?"},
                    {"role": "user", "text": "Yes, one driver looks unconscious and there's smoke coming from the engine."},
                    {"role": "agent", "text": "Stay back from the vehicle. Help is on the way."}
                ]), 
                "2026-03-01 08:30:00"
            ),
            (
                "call_2026_002", 
                "+12175550123", 
                0, 
                json.dumps([
                    {"role": "agent", "text": "911, where is your emergency?"},
                    {"role": "user", "text": "1203 West Nevada Street. I think someone is breaking into my neighbor's house."},
                    {"role": "agent", "text": "Can you see the person?"},
                    {"role": "user", "text": "Yes, a tall male in a dark hoodie is trying to pry open the back window."},
                    {"role": "agent", "text": "Stay on the line with me. Officers are being dispatched to your location."}
                ]), 
                "2026-03-01 09:15:45"
            ),
            (
                "call_2026_003", 
                "+17085550456", 
                1, 
                json.dumps([
                    {"role": "agent", "text": "911, what is the address of the emergency?"},
                    {"role": "user", "text": "Grainger Engineering Library. My friend just collapsed, he's not breathing!"},
                    {"role": "agent", "text": "Is there an AED nearby?"},
                    {"role": "user", "text": "I don't know, I'm looking! Someone else is starting CPR."},
                    {"role": "agent", "text": "Paramedics are en route. I'm going to talk you through CPR instructions."}
                ]), 
                "2026-03-01 10:05:12"
            )
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
