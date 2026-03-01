import asyncio
import base64
import json
import logging
from fastapi import WebSocket
from elevenlabs.conversational_ai.conversation import AudioInterface
from starlette.websockets import WebSocketDisconnect, WebSocketState

logger = logging.getLogger("hackillinois")


class TwilioAudioInterface(AudioInterface):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.input_callback = None
        self.stream_sid = None
        self.loop = asyncio.get_event_loop()
        self._output_count = 0
        self._input_count = 0

    def start(self, input_callback):
        self.input_callback = input_callback
        logger.info("[AudioInterface] start() called — ready to stream")

    def stop(self):
        logger.info(f"[AudioInterface] stop() called — sent {self._output_count} output chunks, received {self._input_count} input chunks")
        self.input_callback = None
        self.stream_sid = None

    def output(self, audio: bytes):
        self._output_count += 1
        if self._output_count <= 3:
            logger.info(f"[AudioInterface] output() chunk #{self._output_count}, size={len(audio)} bytes, first_bytes={audio[:8].hex()}")
        elif self._output_count % 50 == 0:
            logger.debug(f"[AudioInterface] output() chunk #{self._output_count} (still flowing)")
        asyncio.run_coroutine_threadsafe(self._send_audio(audio), self.loop)

    def interrupt(self):
        logger.info("[AudioInterface] interrupt() called")
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
        else:
            if self._output_count <= 3:
                logger.warning("[AudioInterface] output() called but stream_sid is None — audio dropped!")

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
            logger.info(f"[AudioInterface] stream_sid set: {self.stream_sid}")
        elif event_type == "media" and self.input_callback:
            audio_data = base64.b64decode(data["media"]["payload"])
            self._input_count += 1
            if self._input_count <= 3:
                logger.info(f"[AudioInterface] input chunk #{self._input_count}, size={len(audio_data)} bytes, first_bytes={audio_data[:8].hex()}")
            self.input_callback(audio_data)
