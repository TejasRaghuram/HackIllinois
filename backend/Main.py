from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import asyncio
import logging
import os
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager

import database as db

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from elevenlabs import ElevenLabs
import httpx

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

# --- Logging Setup ---
# Store the last 1000 logs in memory
log_history = deque(maxlen=1000)


class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_history.append(log_entry)


# Configure the logger
logger = logging.getLogger("hackillinois")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

# Memory handler for the HTTP endpoint
memory_handler = MemoryLogHandler()
memory_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(memory_handler)


@app.get("/logs/clear")
async def clear_logs():
    """Clear the log history."""
    log_history.clear()
    # Redirect back to logs page
    return Response(status_code=302, headers={"Location": "/logs"})


@app.get("/logs", response_class=HTMLResponse)
async def view_logs():
    """Simple HTTP endpoint to view the last 1000 logs. Auto-refreshes every 2 seconds."""
    logs_str = "\n".join(log_history)
    html_content = f"""
    <html>
        <head>
            <title>Live Logs</title>
            <meta http-equiv="refresh" content="2">
            <style>
                body {{ background-color: #1e1e1e; color: #00ff00; font-family: monospace; padding: 20px; margin: 0; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; margin-top: 60px; }}
                .header {{ position: fixed; top: 0; left: 0; right: 0; background: #2d2d2d; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #444; }}
                h2 {{ margin: 0; font-size: 18px; }}
                .clear-btn {{ background: #ff4444; color: white; border: none; padding: 8px 16px; cursor: pointer; border-radius: 4px; text-decoration: none; font-weight: bold; font-family: sans-serif; }}
                .clear-btn:hover {{ background: #cc0000; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Backend Logs (Auto-refreshing every 2s)</h2>
                <a href="/logs/clear" class="clear-btn">Clear Logs</a>
            </div>
            <pre>{logs_str if logs_str else "No logs yet..."}</pre>
            <script>
                // Auto-scroll to the bottom
                window.scrollTo(0, document.body.scrollHeight);
            </script>
        </body>
    </html>
    """
    return html_content


@app.get("/sessions", response_class=HTMLResponse)
async def view_sessions():
    """View all stored sessions and their transcripts live."""
    sessions = await db.get_all_sessions()

    html = """
    <html>
        <head>
            <title>Live Sessions</title>
            <meta http-equiv="refresh" content="3">
            <style>
                body { background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; padding: 20px; margin: 0; }
                h1 { color: #569cd6; border-bottom: 1px solid #444; padding-bottom: 10px; }
                .session { border: 1px solid #444; padding: 15px; margin-bottom: 20px; border-radius: 5px; background-color: #252526; }
                .session h2 { margin-top: 0; color: #4ec9b0; }
                pre { background-color: #1e1e1e; padding: 10px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; font-family: monospace; font-size: 14px; }
                .caller { color: #ce9178; }
                .agent { color: #9cdcfe; }
            </style>
        </head>
        <body>
            <h1>Live 911 Sessions Dashboard</h1>
            <p>Auto-refreshing every 3 seconds...</p>
    """

    if not sessions:
        html += "<p>No active sessions found in database.</p>"

    for s in sessions:
        html += f"""
        <div class="session">
            <h2>Session: {s["session_id"]}</h2>
            <p><strong>Phone:</strong> {s["phone_number"]} | <strong>Name:</strong> {s.get("caller_name") or "Pending"} | <strong>Address:</strong> {s.get("address") or "Pending"}</p>
            <p><strong>Actions Taken:</strong> {s.get("actions") or "None"}</p>
            <h3>Live Transcript:</h3>
            <pre>"""

        try:
            transcript = json.loads(s["transcript"]) if s.get("transcript") else []
            for turn in transcript:
                role = turn.get("role", "unknown")
                text = turn.get("text", "")
                css_class = "caller" if role == "caller" else "agent"
                html += f"<span class='{css_class}'><strong>{role.upper()}:</strong> {text}</span>\n"
        except:
            html += f"{s.get('transcript', '')}"

        html += """</pre></div>"""

    html += "</body></html>"
    return html


# --- App Endpoints ---


async def _get_database_json():
    """Helper to fetch and serialize the database for HTTP or WebSocket."""
    sessions = await db.get_all_sessions()
    result = {}
    for s in sessions:
        session_data = dict(s)
        # Force conversion to True/False explicitly using boolean logic against integers
        is_active_val = session_data.get("is_active")
        if is_active_val is None:
            session_data["is_active"] = False
        else:
            session_data["is_active"] = bool(int(is_active_val))

        # Parse JSON strings if possible
        if session_data.get("transcript") and isinstance(
            session_data["transcript"], str
        ):
            try:
                session_data["transcript"] = json.loads(session_data["transcript"])
            except json.JSONDecodeError:
                pass

        if session_data.get("actions") and isinstance(session_data["actions"], str):
            try:
                session_data["actions"] = json.loads(session_data["actions"])
            except json.JSONDecodeError:
                pass

        result[session_data["session_id"]] = session_data

    return result


@app.get("/database")
async def get_database():
    """Return the entire database as JSON with session_id as keys."""
    return await _get_database_json()


@app.websocket("/database-stream")
async def database_stream(websocket: WebSocket):
    """Livestream the entire database as JSON to a connected client."""
    await websocket.accept()
    logger.info("Client connected to /database-stream")
    try:
        while True:
            db_json = await _get_database_json()
            await websocket.send_json(db_json)
            await asyncio.sleep(1)  # Broadcast updates every 1 second
    except WebSocketDisconnect:
        logger.info("Client disconnected from /database-stream")
    except Exception as e:
        logger.error(f"Error in /database-stream: {e}")
        try:
            await websocket.close()
        except:
            pass


@app.get("/hello")
def read_hello():
    logger.info("GET /hello was called")
    return {"message": "hello worlds!!!!!!!!"}


@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    """Handle incoming calls and connect them to our AI stream."""
    logger.info(f"Incoming {request.method} call received at /voice")

    # Extract form data or query params to get caller information
    try:
        if request.method == "POST":
            form_data = await request.form()
            call_sid = form_data.get("CallSid")
            caller_phone = form_data.get("From")
            to_number = form_data.get("To")
        else:
            call_sid = request.query_params.get("CallSid")
            caller_phone = request.query_params.get("From")
            to_number = request.query_params.get("To")

        if not call_sid:
            call_sid = f"unknown_{int(datetime.now().timestamp())}"
        if not caller_phone:
            caller_phone = "unknown"
        if not to_number:
            to_number = "unknown"
    except Exception as e:
        logger.error(f"Error parsing request data: {e}", exc_info=True)
        call_sid = f"unknown_{int(datetime.now().timestamp())}"
        caller_phone = "unknown"
        to_number = "unknown"

    logger.info(f"Session started: {call_sid} from {caller_phone} to {to_number}")
    # Create DB session
    await db.create_session(call_sid, caller_phone)

    try:
        # We also need to extract host for absolute webhook path or rely on relative/pre-configured
        response = elevenlabs_client.conversational_ai.twilio.register_call(
            agent_id=AGENT_ID,
            from_number=caller_phone,
            to_number=to_number,
        )
        logger.debug(f"ElevenLabs register_call response: {response}")
        return Response(content=response, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error calling ElevenLabs register_call: {e}", exc_info=True)
        response = VoiceResponse()
        response.say("Sorry, the dispatch agent is currently unavailable.")
        return Response(content=str(response), media_type="application/xml")


# Initialize ElevenLabs Client
try:
    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
    logger.info(f"Initialized ElevenLabs client for agent {AGENT_ID}")
except Exception as e:
    logger.error(f"Failed to initialize ElevenLabs Client: {e}", exc_info=True)
    elevenlabs_client = None


@app.post("/webhook/elevenlabs")
async def handle_elevenlabs_webhook(request: Request):
    """Handle post-call webhooks from ElevenLabs."""
    try:
        payload = await request.json()
        logger.debug(f"ElevenLabs Webhook Payload: {json.dumps(payload)}")
    except Exception as e:
        logger.error(f"Error parsing webhook payload: {e}", exc_info=True)
        return {"status": "error", "message": "Invalid JSON"}

    conversation_id = payload.get("conversation_id")
    # For inbound calls, ElevenLabs webhook might just have conversation details.
    metadata = payload.get("metadata", {})
    transcript_events = payload.get("transcript", [])
    
    # Check standard properties
    phone_number = payload.get("caller_phone_number") or payload.get("from_number")
    
    # Or from custom data
    if not phone_number:
        custom_data = payload.get("conversation_initiation_client_data", {})
        phone_number = custom_data.get("caller_phone")

    session = None
    if phone_number and phone_number != "unknown":
        session = await db.get_session_by_phone(phone_number)
    
    # If no match by phone, let's grab the most recent active session as a fallback
    if not session:
        sessions = await db.get_all_sessions()
        for s in sessions:
            if s.get("is_active"):
                session = s
                phone_number = s["phone_number"]
                break

    session_id = session["session_id"] if session else f"unknown_{conversation_id}"

    formatted_transcript = []
    for turn in transcript_events:
        role = "caller" if turn.get("role") == "user" else "agent"
        text = turn.get("message", turn.get("text", ""))
        formatted_transcript.append({"role": role, "text": text})

    duration = metadata.get("call_duration_secs", 0)

    logger.info(
        f"Webhook received: conv_id={conversation_id}, phone={phone_number}, "
        f"turns={len(formatted_transcript)}, duration={duration}s"
    )

    if session:
        try:
            await db.update_transcript(session_id, formatted_transcript)
            await db.update_session_active(session_id, False)
            logger.info(f"Saved transcript to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}", exc_info=True)
    else:
        logger.warning(f"Could not find active session for phone {phone_number} to save transcript.")

    return {"status": "success"}
