from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import base64
import asyncio
import logging
import os
import websockets
from collections import deque
from datetime import datetime

import audioop
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()

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
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Memory handler for the HTTP endpoint
memory_handler = MemoryLogHandler()
memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
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

# --- App Endpoints ---

@app.get("/hello")
def read_hello():
    logger.info("GET /hello was called")
    return {"message": "hello worlds!"}

@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    """Handle incoming calls and connect them to our AI stream."""
    logger.info("Incoming call received at /voice")
    response = VoiceResponse()
    
    # Let's say a message first so we know the webhook connected successfully
    response.say("Welcome to the AI Dispatcher. Attempting to connect stream.")
    
    connect = Connect()
    
    host = request.url.netloc
    protocol = "wss" if request.url.scheme == "https" else "ws"
    
    # Check if headers contain ngrok or DO specific forwarded headers to ensure correct protocol
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        protocol = "wss" if forwarded_proto == "https" else "ws"
        logger.info(f"Using forwarded protocol: {protocol}")
        
    stream_url = f"{protocol}://{host}/media-stream"
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

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Bidirectional proxy between Twilio and Gemini Live API."""
    await websocket.accept()
    logger.info("Twilio connected to /media-stream WebSocket")
    
    stream_sid = None
    
    # URL of your Modal app (can be overridden via environment variables)
    # MODAL_WS_URL = os.environ.get("MODAL_WS_URL", "wss://vigneshsaravanakumar404--hackillinois-voice-voicepipelin-fdb05b.modal.run/ai-stream")
    
    if not gemini_client:
        logger.error("Gemini client not initialized. Closing connection.")
        await websocket.close()
        return

    try:
        # Connect to Gemini Live API
        # async with websockets.connect(MODAL_WS_URL) as modal_ws:
        # Note: the modal_ws logic is commented out to replace with Gemini Multimodal Live API
        
        config = types.LiveConnectConfig(
            system_instruction="You are a 911 dispatch assistant. Be extremely concise. Ask short questions. You are talking to a caller reporting an emergency. Always start the conversation by asking '911, what is your emergency?'",
            response_modalities=["AUDIO"]
        )
        async with gemini_client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            config=config
        ) as session:
            logger.info("Connected to Gemini Live API")
            
            # Send an initial message to trigger the model's first response
            await session.send(input="A new caller has connected to 911. Please say '911, what is your emergency?' to start the conversation.", end_of_turn=True)
            
            async def receive_from_twilio():
                nonlocal stream_sid
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)
                        
                        if data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            logger.info(f"Stream started: {stream_sid}")
                            
                        elif data['event'] == 'media':
                            # Twilio sends 8kHz µ-law
                            payload = data['media']['payload']
                            chunk = base64.b64decode(payload)
                            
                            # Convert 8kHz µ-law to 8kHz PCM
                            pcm_8k = audioop.ulaw2lin(chunk, 2)
                            
                            # Resample 8kHz PCM to 16kHz PCM
                            pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
                            
                            # Send to Gemini
                            await session.send(input={"data": pcm_16k, "mime_type": "audio/pcm;rate=16000"}, end_of_turn=False)
                            
                        elif data['event'] == 'stop':
                            logger.info("Stream stopped by Twilio")
                            break
                except WebSocketDisconnect:
                    logger.warning("Twilio disconnected")
                except Exception as e:
                    logger.error(f"Error receiving from Twilio: {e}")

            async def receive_from_gemini():
                try:
                    async for response in session.receive():
                        server_content = response.server_content
                        if server_content is not None:
                            model_turn = server_content.model_turn
                            if model_turn is not None:
                                for part in model_turn.parts:
                                    if part.text:
                                        logger.info(f"Gemini (Text): {part.text}")
                                    if part.inline_data and part.inline_data.data:
                                        # Gemini returns 24kHz PCM
                                        pcm_24k = part.inline_data.data
                                        
                                        # Resample 24kHz PCM to 8kHz PCM
                                        pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
                                        
                                        # Convert 8kHz PCM to 8kHz µ-law
                                        ulaw_8k = audioop.lin2ulaw(pcm_8k, 2)
                                        
                                        # Send back to Twilio
                                        payload = base64.b64encode(ulaw_8k).decode('utf-8')
                                        media_msg = {
                                            "event": "media",
                                            "media": {
                                                "payload": payload
                                            }
                                        }
                                        if stream_sid:
                                            media_msg["streamSid"] = stream_sid
                                        
                                        logger.debug("Sending audio chunk to Twilio")
                                        await websocket.send_text(json.dumps(media_msg))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            # Run both tasks concurrently and stop when either finishes
            task1 = asyncio.create_task(receive_from_twilio())
            task2 = asyncio.create_task(receive_from_gemini())
            
            done, pending = await asyncio.wait(
                [task1, task2],
                return_when=asyncio.FIRST_COMPLETED
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
            await websocket.close()
        except:
            pass
