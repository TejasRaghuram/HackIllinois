from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import base64
import asyncio
import logging
import os
import websockets
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager

import database as db

import audioop
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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
        else:
            call_sid = request.query_params.get("CallSid")
            caller_phone = request.query_params.get("From")

        if not call_sid:
            call_sid = f"unknown_{int(datetime.now().timestamp())}"
        if not caller_phone:
            caller_phone = "unknown"
    except Exception as e:
        logger.error(f"Error parsing request data: {e}")
        call_sid = f"unknown_{int(datetime.now().timestamp())}"
        caller_phone = "unknown"

    logger.info(f"Session started: {call_sid} from {caller_phone}")
    # Create DB session
    await db.create_session(call_sid, caller_phone)

    response = VoiceResponse()

    connect = Connect()

    host = request.url.netloc
    protocol = "wss" if request.url.scheme == "https" else "ws"

    # Check if headers contain ngrok or DO specific forwarded headers to ensure correct protocol
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        protocol = "wss" if forwarded_proto == "https" else "ws"
        logger.info(f"Using forwarded protocol: {protocol}")

    stream_url = f"{protocol}://{host}/media-stream?session_id={call_sid}&phone_number={caller_phone}"
    logger.info(f"Connecting Twilio stream to: {stream_url}")

    connect.stream(url=stream_url)
    response.append(connect)

    # Add a pause so it doesn't immediately hang up if the stream drops
    response.pause(length=10)
    response.say("Stream disconnected.")

    twiml_str = str(response)
    logger.debug(f"Generated TwiML: {twiml_str}")

    return Response(content=twiml_str, media_type="application/xml")


# Initialize Gemini Client
try:
    # Hardcoded key split into two parts to avoid simple scraping
    _p1 = "AIzaSyBNfb4knutwAHIkcn5i"
    _p2 = "rej7KOPLqs7-HOs"
    gemini_client = genai.Client(api_key=_p1 + _p2)
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}")
    gemini_client = None


class DecisionMakerOutput(BaseModel):
    caller_name: str | None = Field(
        default=None, description="Name of the caller if mentioned."
    )
    address: str | None = Field(
        default=None, description="Location of the emergency if mentioned."
    )
    actions: list[str] = Field(
        default_factory=list,
        description="List of actions taken, such as 'dispatch_police', 'dispatch_fire', 'dispatch_ems', or empty if none.",
    )


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Bidirectional proxy between Twilio and Gemini Live API."""
    session_id = websocket.query_params.get("session_id", "unknown")
    phone_number = websocket.query_params.get("phone_number", "unknown")

    # Try one more fallback if Twilio sent it strangely (e.g., if &amp; broke parsing)
    if session_id == "unknown":
        for key in websocket.query_params.keys():
            if "session_id" in key:
                session_id = websocket.query_params[key]
                break

    await websocket.accept()
    logger.info(
        f"Twilio connected to /media-stream WebSocket for session {session_id} ({phone_number})"
    )

    stream_sid = None
    transcript_buffer = []

    if not gemini_client:
        logger.error("Gemini client not initialized. Closing connection.")
        await websocket.close()
        return

    try:
        # Load prompts dynamically so they can be edited without restarting
        try:
            with open(
                os.path.join(os.path.dirname(__file__), "prompts/system_prompt.txt"),
                "r",
            ) as f:
                system_prompt = f.read()
            with open(
                os.path.join(
                    os.path.dirname(__file__), "prompts/structured_questions.txt"
                ),
                "r",
            ) as f:
                structured_questions = f.read()
            full_system_instruction = f"{system_prompt}\n\n{structured_questions}"
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            full_system_instruction = "You are a 911 dispatch assistant."

        # FIX 1: Use plain dicts for transcription config instead of non-existent class
        config = types.LiveConnectConfig(
            system_instruction=full_system_instruction,
            response_modalities=["AUDIO"],
            input_audio_transcription={},
            output_audio_transcription={},
        )
        async with gemini_client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025", config=config
        ) as session:
            logger.info(f"Connected to Gemini Live API for session {session_id}")

            # FIX 2: Use send_client_content for the initial text prompt
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="A new caller has connected to 911. Please say '911, what is your emergency?' and begin triage."
                        )
                    ],
                ),
                turn_complete=True,
            )

            audio_chunks_sent = 0

            async def receive_from_twilio():
                nonlocal stream_sid, audio_chunks_sent
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)

                        if data["event"] == "start":
                            stream_sid = data["start"]["streamSid"]
                            logger.info(
                                f"Stream started: {stream_sid} for session: {session_id}"
                            )

                        elif data["event"] == "media":
                            # Twilio sends 8kHz µ-law
                            payload = data["media"]["payload"]
                            chunk = base64.b64decode(payload)

                            # Convert 8kHz µ-law to 8kHz PCM
                            pcm_8k = audioop.ulaw2lin(chunk, 2)

                            # Resample 8kHz PCM to 16kHz PCM
                            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)

                            try:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=pcm_16k, mime_type="audio/pcm;rate=16000"
                                    )
                                )
                                audio_chunks_sent += 1
                                # Log every 500 chunks (~25 seconds of audio) to confirm flow
                                if audio_chunks_sent % 500 == 0:
                                    logger.info(
                                        f"Sent {audio_chunks_sent} audio chunks to Gemini [{session_id}]"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error sending audio to Gemini [{session_id}]: {e}"
                                )

                        elif data["event"] == "stop":
                            logger.info(f"Stream stopped by Twilio [{session_id}]")
                            break
                except WebSocketDisconnect:
                    logger.warning(f"Twilio disconnected [{session_id}]")
                except Exception as e:
                    logger.error(f"Error receiving from Twilio [{session_id}]: {e}")

            async def receive_from_gemini():
                try:
                    while True:
                        async for response in session.receive():
                            # Handle model audio/text output
                            if (
                                response.server_content
                                and response.server_content.model_turn
                            ):
                                model_turn = response.server_content.model_turn
                                for part in model_turn.parts:
                                    if part.text:
                                        # Only log if it's not just a thinking/internal thought
                                        if not part.text.startswith("**"):
                                            logger.info(
                                                f"Assistant [{session_id}]: {part.text}"
                                            )
                                            transcript_buffer.append(
                                                {"role": "agent", "text": part.text}
                                            )
                                            # Since DM agent is off, save to DB directly when agent speaks
                                            await db.update_transcript(
                                                session_id, transcript_buffer
                                            )
                                    if part.inline_data and part.inline_data.data:
                                        # Gemini returns 24kHz PCM
                                        pcm_24k = part.inline_data.data

                                        # Resample 24kHz PCM to 8kHz PCM
                                        pcm_8k, _ = audioop.ratecv(
                                            pcm_24k, 2, 1, 24000, 8000, None
                                        )

                                        # Convert 8kHz PCM to 8kHz µ-law
                                        ulaw_8k = audioop.lin2ulaw(pcm_8k, 2)

                                        # Send back to Twilio
                                        payload = base64.b64encode(ulaw_8k).decode(
                                            "utf-8"
                                        )
                                        media_msg = {
                                            "event": "media",
                                            "media": {"payload": payload},
                                        }
                                        if stream_sid:
                                            media_msg["streamSid"] = stream_sid

                                        # logger.debug("Sending audio chunk to Twilio")
                                        await websocket.send_text(json.dumps(media_msg))

                            # Handle interruptions
                            if (
                                response.server_content
                                and response.server_content.interrupted
                            ):
                                logger.info("Model interrupted by user speech")

                            if (
                                response.server_content
                                and response.server_content.turn_complete
                            ):
                                logger.debug("Model turn complete")

                            # Handle input transcription (what the user said)
                            if (
                                response.server_content
                                and hasattr(
                                    response.server_content, "input_transcription"
                                )
                                and response.server_content.input_transcription
                            ):
                                if response.server_content.input_transcription.text:
                                    text = (
                                        response.server_content.input_transcription.text
                                    )
                                    logger.info(f"Caller [{session_id}]: {text}")
                                    transcript_buffer.append(
                                        {"role": "caller", "text": text}
                                    )
                                    # Since DM agent is off, we still want to save transcripts periodically or as they come in.
                                    # We'll just trigger a DB save here for testing so you can see it!
                                    await db.update_transcript(
                                        session_id, transcript_buffer
                                    )

                            # Handle output transcription (what the model said)
                            if (
                                response.server_content
                                and hasattr(
                                    response.server_content, "output_transcription"
                                )
                                and response.server_content.output_transcription
                            ):
                                if response.server_content.output_transcription.text:
                                    # logger.info(f"Assistant transcript: {response.server_content.output_transcription.text}")
                                    pass

                        # logger.debug("session.receive() loop ended, re-entering...")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            async def run_decision_maker():
                """Periodically analyze the transcript and update DB and Gemini."""
                last_transcript_length = 0
                last_actions = []
                while True:
                    await asyncio.sleep(10)
                    if len(transcript_buffer) > last_transcript_length:
                        last_transcript_length = len(transcript_buffer)
                        transcript_text = "\n".join(
                            [
                                f"{msg['role']}: {msg['text']}"
                                for msg in transcript_buffer
                            ]
                        )

                        prompt = f"""
You are a 911 dispatch decision-making agent.
Review the following call transcript and extract the caller's name, address, and any required emergency services to dispatch.
If there's not enough information, leave the fields empty. Do not guess.
Transcript:
{transcript_text}
"""
                        try:
                            start_time = datetime.now()
                            # Use standard gemini-2.5-flash for text reasoning
                            response = await gemini_client.aio.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    response_schema=DecisionMakerOutput,
                                    temperature=0.0,
                                ),
                            )
                            latency = (datetime.now() - start_time).total_seconds()
                            result = json.loads(response.text)
                            logger.info(
                                f"DM Agent [{session_id}] (Latency: {latency:.2f}s): {result}"
                            )

                            # Save to DB
                            await db.update_transcript(session_id, transcript_buffer)
                            await db.update_extracted_info(
                                session_id,
                                caller_name=result.get("caller_name"),
                                address=result.get("address"),
                                actions=result.get("actions"),
                            )

                            current_actions = result.get("actions", [])
                            if current_actions and current_actions != last_actions:
                                action_msg = f"SYSTEM ALERT: The DM agent has taken the following actions: {', '.join(current_actions)}. Subtly inform the caller that help is on the way."
                                logger.info(
                                    f"Injecting to Gemini [{session_id}]: {action_msg}"
                                )
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user", parts=[types.Part(text=action_msg)]
                                    ),
                                    turn_complete=True,
                                )
                                last_actions = current_actions

                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.error(f"Error in DM agent for {session_id}: {e}")

            # Run tasks concurrently and stop when either finishes
            task1 = asyncio.create_task(receive_from_twilio())
            task2 = asyncio.create_task(receive_from_gemini())
            # Task 3 (DM agent) is temporarily commented out for testing
            # task3 = asyncio.create_task(run_decision_maker())

            done, pending = await asyncio.wait(
                [task1, task2],  # , task3],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel any pending tasks to ensure clean up
            for task in pending:
                task.cancel()

    except WebSocketDisconnect:
        logger.warning("Twilio disconnected from /media-stream WebSocket")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
    finally:
        try:
            logger.info(f"Session {session_id} ended. Performing final DB updates...")
            if "transcript_buffer" in locals() and transcript_buffer:
                logger.info(
                    f"Flushing final transcript (length {len(transcript_buffer)}) to DB for {session_id}"
                )
                await db.update_transcript(session_id, transcript_buffer)

            logger.info(f"Setting is_active=False for {session_id}")
            await db.update_session_active(session_id, False)
            logger.info(f"Final DB updates complete for {session_id}")
        except Exception as e:
            logger.error(f"Failed to update session active status or transcript: {e}")
        try:
            await websocket.close()
        except:
            pass


# --- Serve React Frontend ---
# Mount the client/build directory to serve the React application
client_build_dir = os.path.join(os.path.dirname(__file__), "..", "client", "build")
if os.path.exists(client_build_dir):
    app.mount("/", StaticFiles(directory=client_build_dir, html=True), name="static")
else:
    logger.warning(
        f"Client build directory not found at {client_build_dir}. Ensure 'npm run build' has been run."
    )
    
    @app.get("/")
    async def fallback_root():
        return HTMLResponse("<h1>Client Build Not Found</h1><p>Please run npm run build in the client directory.</p>")
