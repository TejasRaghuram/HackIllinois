import os
import json
import base64
import asyncio
import modal

app = modal.App("hackillinois-voice")

voice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget", "git", "curl")
    .pip_install(
        "faster-whisper>=1.1.0",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "numpy<2.0",
        "fastapi==0.115.5",
        "uvicorn[standard]",
        "httpx",
        "piper-tts",
    )
    .run_commands(
        "mkdir -p /models/piper",
        'wget -q -O /models/piper/en_US-lessac-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true"',
        'wget -q -O /models/piper/en_US-lessac-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"',
    )
)

WHISPER_MODEL = "tiny.en"
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

@app.cls(
    image=voice_image,
    gpu="T4",
    min_containers=1,
)
@modal.concurrent(max_inputs=10)
class VoicePipeline:

    @modal.enter()
    def setup_models(self):
        """Load all models once when the container starts."""
        import torch
        from faster_whisper import WhisperModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        # --- 1. VAD (Silero) ---
        print("Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.get_speech_timestamps = self.vad_utils[0]

        # --- 2. STT (Faster-Whisper) ---
        print(f"Loading Faster-Whisper ({WHISPER_MODEL})...")
        self.stt_model = WhisperModel(
            WHISPER_MODEL, device=self.device, compute_type="float16"
        )

        # --- 3. TTS (Piper) ---
        print("Loading Piper TTS...")
        from piper import PiperVoice

        self.tts_voice = PiperVoice.load("/models/piper/en_US-lessac-medium.onnx")

        print("All models loaded. Pipeline ready.")

    @modal.asgi_app()
    def asgi_app(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        import httpx

        web_app = FastAPI()

        try:
            llm_function = modal.Function.from_name("hackillinois-llm", "serve")
            llm_url = llm_function.web_url
            print(f"Discovered LLM Server URL: {llm_url}")
        except Exception as e:
            print(f"Warning: Could not discover LLM server URL: {e}")
            llm_url = "PLACEHOLDER_URL_REPLACE_ME"

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        @web_app.websocket("/ai-stream")
        async def ai_stream(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket client connected.")

            import audioop
            import numpy as np
            import re

            chat_history = [
                {
                    "role": "system",
                    "content": (
                        "You are a 911 dispatch assistant. Be extremely concise. "
                        "Ask short questions. You are talking to a caller reporting "
                        "an emergency."
                    ),
                }
            ]

            audio_buffer = bytearray()
            is_speaking = False
            silence_chunks = 0

            async def generate_response(user_text: str):
                """Call the external vLLM API, then TTS the response."""
                chat_history.append({"role": "user", "content": user_text})

                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{llm_url}/v1/chat/completions",
                            json={
                                "model": LLM_MODEL,
                                "messages": chat_history,
                                "max_tokens": 150,
                                "temperature": 0.7,
                                "stream": False,
                            },
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        result = response.json()
                        full_response = result["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    print(f"Error calling LLM at {llm_url}: {e}")
                    full_response = "I am sorry, I am experiencing technical difficulties."

                print(f"LLM: {full_response}")
                chat_history.append(
                    {"role": "assistant", "content": full_response}
                )

                # Sentence-split and synthesize each chunk
                sentences = re.split(r"(?<=[.!?])\s+", full_response)
                for sentence in sentences:
                    if sentence.strip():
                        await synthesize_and_send(sentence.strip(), websocket)

            async def synthesize_and_send(text: str, ws: WebSocket):
                """Generate TTS audio and send as 8kHz mu-law for Twilio."""
                import audioop as _audioop

                for pcm_chunk in self.tts_voice.synthesize_stream_raw(text):
                    converted, _ = _audioop.ratecv(
                        pcm_chunk, 2, 1, 22050, 8000, None
                    )
                    mulaw = _audioop.lin2ulaw(converted, 2)
                    await ws.send_text(
                        json.dumps(
                            {
                                "event": "media",
                                "media": {
                                    "payload": base64.b64encode(mulaw).decode()
                                },
                            }
                        )
                    )

            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    if data["event"] == "start":
                        print("Stream started")

                    elif data["event"] == "media":
                        payload = data["media"]["payload"]
                        chunk = base64.b64decode(payload)
                        pcm_data = audioop.ulaw2lin(chunk, 2)
                        audio_buffer.extend(pcm_data)

                        if len(audio_buffer) >= 2048:
                            import torch
                            import torchaudio

                            np_array = (
                                np.frombuffer(audio_buffer, dtype=np.int16)
                                .astype(np.float32)
                                / 32768.0
                            )
                            tensor_chunk = torch.from_numpy(np_array)
                            tensor_16k = torchaudio.transforms.Resample(
                                8000, 16000
                            )(tensor_chunk)
                            speech_prob = self.vad_model(
                                tensor_16k, 16000
                            ).item()

                            if speech_prob > 0.5:
                                is_speaking = True
                                silence_chunks = 0
                            elif is_speaking:
                                silence_chunks += 1

                            if is_speaking and silence_chunks > 5:
                                print("Speech ended — transcribing...")
                                is_speaking = False
                                silence_chunks = 0

                                full_np = (
                                    np.frombuffer(audio_buffer, dtype=np.int16)
                                    .astype(np.float32)
                                    / 32768.0
                                )
                                full_16k = torchaudio.transforms.Resample(
                                    8000, 16000
                                )(torch.from_numpy(full_np))

                                segments, _ = self.stt_model.transcribe(
                                    full_16k.numpy(), beam_size=1
                                )
                                transcript = " ".join(
                                    s.text for s in segments
                                ).strip()

                                print(f"Caller: {transcript}")
                                audio_buffer.clear()

                                if transcript:
                                    asyncio.create_task(
                                        generate_response(transcript)
                                    )

                    elif data["event"] == "stop":
                        print("Stream stopped")
                        break

            except WebSocketDisconnect:
                print("Client disconnected.")
            except Exception as e:
                import traceback

                print(f"Error: {e}")
                traceback.print_exc()

        return web_app
