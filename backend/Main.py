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
import httpx
import websockets

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

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
    sessions = await db.get_all_calls()

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
        status = "Active" if s.get("is_active") else "Ended"
        html += f"""
        <div class="session">
            <h2>Call ID: {s["call_id"]} [{status}]</h2>
            <p><strong>Phone:</strong> {s["phone_number"]} </p>
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
    calls = await db.get_all_calls()
    result = {}
    for c in calls:
        call_data = dict(c)
        is_active_val = call_data.get("is_active")
        if is_active_val is None:
            call_data["is_active"] = False
        else:
            call_data["is_active"] = bool(int(is_active_val))

        if call_data.get("transcript") and isinstance(call_data["transcript"], str):
            try:
                call_data["transcript"] = json.loads(call_data["transcript"])
            except json.JSONDecodeError:
                pass

        result[call_data["call_id"]] = call_data

    return result

@app.get("/database")
async def get_database():
    """Return the entire database as JSON with call_id as keys."""
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

    try:
        if request.method == "POST":
            form_data = await request.form()
            call_sid = form_data.get("CallSid")
            caller_phone = form_data.get("From")
        else:
            call_sid = request.query_params.get("CallSid")
            caller_phone = request.query_params.get("From")

        if not call_sid:
            call_sid = f"unknown_{int(datetime.now().timestamp())}"
        if not caller_phone:
            caller_phone = "unknown"
    except Exception as e:
        logger.error(f"Error parsing request data: {e}", exc_info=True)
        call_sid = f"unknown_{int(datetime.now().timestamp())}"
        caller_phone = "unknown"

    logger.info(f"Call started: {call_sid} from {caller_phone}")
    # Create DB session
    await db.create_call(call_sid, caller_phone)

    # Return TwiML that connects to our WebSocket
    response = VoiceResponse()
    connect = Connect()
    host = request.headers.get("host", "localhost:8000")
    
    # Render Stream URL based on the request host
    stream_url = f"wss://{host}/media-stream"
    stream = Stream(url=stream_url)
    stream.parameter(name="call_id", value=call_sid)
    connect.append(stream)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


async def get_elevenlabs_signed_url() -> str:
    """Helper function to get signed URL for authenticated conversations."""
    async with httpx.AsyncClient() as client:
        url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={AGENT_ID}"
        response = await client.get(url, headers={"xi-api-key": ELEVENLABS_API_KEY})
        response.raise_for_status()
        data = response.json()
        return data["signed_url"]

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """WebSocket for handling Twilio media streams."""
    await websocket.accept()
    logger.info("[Server] Twilio connected to media stream")

    stream_sid = None
    call_id = None
    elevenlabs_ws = None

    try:
        # 1. Get initial message to extract call_id and streamSid
        init_data = await websocket.receive_text()
        init_msg = json.loads(init_data)
        
        if init_msg.get("event") == "start":
            stream_sid = init_msg["start"]["streamSid"]
            call_id = init_msg["start"].get("customParameters", {}).get("call_id")
            if not call_id:
                call_id = init_msg["start"].get("callSid", f"unknown_{int(datetime.now().timestamp())}")
            logger.info(f"[Twilio] Stream started - StreamSid: {stream_sid}, CallId: {call_id}")
        else:
            logger.warning(f"[Twilio] Expected 'start' event, got: {init_msg.get('event')}")
            call_id = f"unknown_{int(datetime.now().timestamp())}"

        # 2. Connect to ElevenLabs
        signed_url = await get_elevenlabs_signed_url()
        elevenlabs_ws = await websockets.connect(signed_url)
        logger.info(f"[ElevenLabs] Connected to Conversational AI for call {call_id}")

        # 3. Create bidirectional forwarding tasks
        async def twilio_to_elevenlabs():
            try:
                # Handle the first message we already popped (if it had media)
                if init_msg.get("event") == "media":
                    audio_payload = init_msg["media"]["payload"]
                    await elevenlabs_ws.send(json.dumps({"user_audio_chunk": audio_payload}))
                
                while True:
                    data = await websocket.receive_text()
                    msg = json.loads(data)
                    
                    if msg.get("event") == "media":
                        audio_payload = msg["media"]["payload"]
                        await elevenlabs_ws.send(json.dumps({
                            "user_audio_chunk": audio_payload
                        }))
                    elif msg.get("event") == "stop":
                        logger.info(f"[Twilio] Stream {stream_sid} ended")
                        break
            except WebSocketDisconnect:
                logger.info("[Twilio] Client disconnected")
            except Exception as e:
                logger.error(f"[Twilio to ElevenLabs] Error: {e}", exc_info=True)

        async def elevenlabs_to_twilio():
            try:
                async for data in elevenlabs_ws:
                    msg = json.loads(data)
                    msg_type = msg.get("type")
                    
                    if msg_type == "audio":
                        # Forward audio back to Twilio
                        chunk = msg.get("audio", {}).get("chunk")
                        if not chunk:
                            chunk = msg.get("audio_event", {}).get("audio_base_64")
                        if chunk and stream_sid:
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": chunk
                                }
                            }))
                            
                    elif msg_type == "interruption":
                        # Clear Twilio's audio queue
                        if stream_sid:
                            await websocket.send_text(json.dumps({
                                "event": "clear",
                                "streamSid": stream_sid
                            }))
                            
                    elif msg_type == "ping":
                        event_id = msg.get("ping_event", {}).get("event_id")
                        if event_id:
                            await elevenlabs_ws.send(json.dumps({
                                "type": "pong",
                                "event_id": event_id
                            }))
                            
                    elif msg_type == "user_transcript":
                        transcript = msg.get("user_transcription_event", {}).get("user_transcript")
                        if transcript and call_id:
                            logger.info(f"[Live Transcript] Caller: {transcript}")
                            await db.append_transcript(call_id, "caller", transcript)
                            
                    elif msg_type == "agent_response":
                        response = msg.get("agent_response_event", {}).get("agent_response")
                        if response and call_id:
                            logger.info(f"[Live Transcript] Agent: {response}")
                            await db.append_transcript(call_id, "agent", response)

            except websockets.exceptions.ConnectionClosed:
                logger.info("[ElevenLabs] WebSocket closed")
            except Exception as e:
                logger.error(f"[ElevenLabs to Twilio] Error: {e}", exc_info=True)

        # Run both tasks concurrently
        await asyncio.gather(
            twilio_to_elevenlabs(),
            elevenlabs_to_twilio(),
            return_exceptions=True
        )

    except Exception as e:
        logger.error(f"Error in media_stream: {e}", exc_info=True)
    finally:
        # Mark call inactive as soon as ANY connection drops
        if call_id:
            await db.set_call_inactive(call_id)
            logger.info(f"Call {call_id} marked as inactive due to disconnect.")

        # Cleanup connections
        if elevenlabs_ws and not elevenlabs_ws.closed:
            try:
                await elevenlabs_ws.close()
            except:
                pass
        try:
            await websocket.close()
        except:
            pass
