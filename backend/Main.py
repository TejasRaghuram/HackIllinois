from fastapi import FastAPI, Request, Response, WebSocket
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
import json
import base64
import asyncio

app = FastAPI()

@app.get("/hello")
def read_hello():
    return {"message": "hello worlds!"}

@app.api_route("/voice", methods=["GET", "POST"])
async def voice(request: Request):
    """Handle incoming calls and connect them to our AI stream."""
    response = VoiceResponse()
    
    # Twilio sends a POST request by default. We need to respond with TwiML.
    # We'll use the <Connect><Stream> verb to start a bi-directional WebSocket.
    connect = Connect()
    
    # Determine the host and protocol for the WebSocket
    host = request.url.netloc
    protocol = "wss" if request.url.scheme == "https" else "ws"
    
    # The stream URL points to our /media-stream endpoint
    stream_url = f"{protocol}://{host}/media-stream"
    
    connect.stream(url=stream_url)
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle the real-time audio stream from Twilio and echo it back after 3 seconds."""
    await websocket.accept()
    print("Twilio connected to media stream")
    
    stream_sid = None
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                print(f"Stream started: {stream_sid}")
                
            elif data['event'] == 'media':
                # This is the raw audio (Mu-law 8000Hz)
                payload = data['media']['payload']
                
                # TEST: Echo back the audio after 3 seconds
                # Note: In a real app, we'd process this with STT/LLM/TTS
                async def echo_after_delay(p, sid):
                    await asyncio.sleep(3)
                    # Send the audio back to Twilio
                    echo_message = {
                        "event": "media",
                        "streamSid": sid,
                        "media": {
                            "payload": p
                        }
                    }
                    try:
                        await websocket.send_text(json.dumps(echo_message))
                    except:
                        pass # Connection might be closed

                # Start the echo task in the background
                asyncio.create_task(echo_after_delay(payload, stream_sid))
                
            elif data['event'] == 'stop':
                print("Stream stopped")
                break
                
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass
