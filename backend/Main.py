from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import base64
import asyncio
import logging
from datetime import datetime

app = FastAPI()

# --- Logging Setup ---
# We keep a list of active log watcher WebSockets
log_watchers = set()

class WebsocketLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        # We need to send this to all connected log watchers
        # Since emit is synchronous and websockets are async, we use the event loop
        if log_watchers:
            try:
                loop = asyncio.get_running_loop()
                for watcher in list(log_watchers):
                    loop.create_task(self.send_log(watcher, log_entry))
            except RuntimeError:
                pass # No running event loop
                
    async def send_log(self, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception:
            log_watchers.discard(websocket)

# Configure the logger
logger = logging.getLogger("hackillinois")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# WebSocket handler
ws_handler = WebsocketLogHandler()
ws_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ws_handler)

@app.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    """Live streaming endpoint for logs."""
    await websocket.accept()
    log_watchers.add(websocket)
    try:
        await websocket.send_text("Connected to live logs stream.")
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_watchers.discard(websocket)
    except Exception as e:
        log_watchers.discard(websocket)

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
    
    connect = Connect()
    
    host = request.url.netloc
    # Force 'wss' if required, but let's log what we are generating
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
    
    # Optional: Add a pause to prevent immediate hangup if the stream connection takes a moment
    # response.pause(length=40)
    
    twiml_str = str(response)
    logger.debug(f"Generated TwiML: {twiml_str}")
    
    return Response(content=twiml_str, media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle the real-time audio stream from Twilio and echo it back after 3 seconds."""
    await websocket.accept()
    logger.info("Twilio connected to /media-stream WebSocket")
    
    stream_sid = None
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                logger.info(f"Stream started: {stream_sid}")
                
            elif data['event'] == 'media':
                payload = data['media']['payload']
                
                # Test: Echo back the audio after a small delay
                async def echo_after_delay(p, sid):
                    await asyncio.sleep(3)
                    echo_message = {
                        "event": "media",
                        "streamSid": sid,
                        "media": {
                            "payload": p
                        }
                    }
                    try:
                        await websocket.send_text(json.dumps(echo_message))
                        logger.debug(f"Echoed audio chunk for stream {sid}")
                    except Exception as e:
                        logger.error(f"Error sending echo: {e}")

                # Start the echo task
                asyncio.create_task(echo_after_delay(payload, stream_sid))
                
            elif data['event'] == 'stop':
                logger.info("Stream stopped by Twilio")
                break
                
    except WebSocketDisconnect:
        logger.warning("Twilio disconnected from /media-stream WebSocket")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except:
            pass
