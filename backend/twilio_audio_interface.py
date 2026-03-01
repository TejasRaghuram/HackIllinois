import asyncio
import audioop
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
        self._ulaw_to_pcm_state = None

    def start(self, input_callback):
        self.input_callback = input_callback
        logger.info("[AudioInterface] start() called — ready to stream")

    def stop(self):
        logger.info(f"[AudioInterface] stop() — sent {self._output_count} output chunks, received {self._input_count} input chunks")
        self.input_callback = None
        self.stream_sid = None

    def output(self, audio: bytes):
        """Receive PCM 16-bit 16kHz from ElevenLabs, convert to mulaw 8kHz for Twilio."""
        self._output_count += 1
        try:
            # Downsample PCM from 16kHz to 8kHz
            audio_8k, _ = audioop.ratecv(audio, 2, 1, 16000, 8000, None)
            # Convert PCM to mulaw
            audio_ulaw = audioop.lin2ulaw(audio_8k, 2)
        except audioop.error:
            audio_ulaw = audio

        if self._output_count <= 3:
            logger.info(f"[AudioInterface] output #{self._output_count}: PCM {len(audio)}B -> ulaw {len(audio_ulaw)}B")

        asyncio.run_coroutine_threadsafe(self._send_audio(audio_ulaw), self.loop)

    def interrupt(self):
        logger.info("[AudioInterface] interrupt()")
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
            logger.info(f"[AudioInterface] stream_sid set: {self.stream_sid}")
        elif event_type == "media" and self.input_callback:
            raw_mulaw = base64.b64decode(data["media"]["payload"])
            self._input_count += 1

            try:
                # Convert mulaw to PCM 16-bit
                audio_pcm = audioop.ulaw2lin(raw_mulaw, 2)
                # Upsample from 8kHz to 16kHz
                audio_16k, self._ulaw_to_pcm_state = audioop.ratecv(
                    audio_pcm, 2, 1, 8000, 16000, self._ulaw_to_pcm_state
                )
            except audioop.error:
                audio_16k = raw_mulaw

            if self._input_count <= 3:
                logger.info(f"[AudioInterface] input #{self._input_count}: ulaw {len(raw_mulaw)}B -> PCM {len(audio_16k)}B")

            self.input_callback(audio_16k)
