import asyncio
import base64
import json
from fastapi import WebSocket
from elevenlabs.conversational_ai.conversation import AudioInterface
from starlette.websockets import WebSocketDisconnect, WebSocketState


class TwilioAudioInterface(AudioInterface):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.input_callback = None
        self.stream_sid = None
        self.loop = asyncio.get_event_loop()

    def start(self, input_callback):
        self.input_callback = input_callback

    def stop(self):
        self.input_callback = None
        self.stream_sid = None

    def output(self, audio: bytes):
        asyncio.run_coroutine_threadsafe(self._send_audio(audio), self.loop)

    def interrupt(self):
        asyncio.run_coroutine_threadsafe(self._send_clear(), self.loop)

    async def _send_audio(self, audio: bytes):
        if self.stream_sid:
            payload = base64.b64encode(audio).decode("utf-8")
            message = {
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }
            try:
                if self.websocket.application_state == WebSocketState.CONNECTED:
                    await self.websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def _send_clear(self):
        if self.stream_sid:
            message = {"event": "clear", "streamSid": self.stream_sid}
            try:
                if self.websocket.application_state == WebSocketState.CONNECTED:
                    await self.websocket.send_text(json.dumps(message))
            except (WebSocketDisconnect, RuntimeError):
                pass

    async def handle_twilio_message(self, data: dict):
        event_type = data.get("event")
        if event_type == "start":
            self.stream_sid = data["start"]["streamSid"]
        elif event_type == "media" and self.input_callback:
            audio_data = base64.b64decode(data["media"]["payload"])
            self.input_callback(audio_data)
