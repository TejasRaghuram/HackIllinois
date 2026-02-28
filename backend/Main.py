from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import base64
import asyncio
import logging
from collections import deque
from datetime import datetime

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
