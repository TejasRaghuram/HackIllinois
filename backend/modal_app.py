"""
Voice Pipeline on Modal — Fixed & Modernized
=============================================
Fixes the vLLM tokenizer crash (all_special_tokens_extended) by:
  1. Using a modern vLLM version with Modal's CUDA image
  2. Serving vLLM as a subprocess (Modal's recommended pattern)
  3. Pinning compatible dependency versions

Architecture: Single-container approach for hackathon simplicity.
For production, split STT/LLM/TTS into separate Modal services.
"""

import os
import io
import asyncio
import json
import base64
import modal

# ---------------------------------------------------------------------------
# Modal App
# ---------------------------------------------------------------------------
app = modal.App("hackillinois-voice-pipeline")

# Use Modal's CUDA image (recommended for GPU workloads) instead of debian_slim
# This avoids CUDA driver mismatches and provides proper GPU support.
voice_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "wget", "git", "curl")
    .pip_install(
        # Modern vLLM — fixes the tokenizer crash
        "vllm>=0.8.0",
        # Pin transformers to avoid TokenizersBackend issues
        "transformers>=4.45,<4.48",
        # STT
        "faster-whisper>=1.1.0",
        # Audio processing
        "torch>=2.4.0",
        "torchaudio>=2.4.0",
        "numpy<2.0",
        # Web framework
        "fastapi",
        "websockets",
        "uvicorn",
    )
    # Download a small Piper TTS model for voice synthesis
    .run_commands(
        "mkdir -p /models/piper",
        'wget -q -O /models/piper/en_US-lessac-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true"',
        'wget -q -O /models/piper/en_US-lessac-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"',
        "pip install piper-tts",
    )
)

# Cache model weights so they persist across container restarts
model_volume = modal.Volume.from_name("voice-pipeline-models", create_if_missing=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Pick a small, fast model. Phi-3.5-mini or Qwen2.5-3B-Instruct both work.
# If Phi-3.5-mini keeps crashing for you, swap to Qwen — it's well-tested with vLLM.
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
WHISPER_MODEL = "tiny.en"  # "small.en" for better accuracy, "tiny.en" for speed


@app.cls(
    image=voice_image,
    gpu="A100",  # A100 is cheaper than H200 and plenty for a 3B model
    volumes={"/root/.cache": model_volume},
    container_idle_timeout=300,
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
        print(f"Using device: {self.device}")

        # --- 1. VAD (Voice Activity Detection) ---
        print("Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,  # Suppress the trust warning
        )
        self.get_speech_timestamps = self.vad_utils[0]

        # --- 2. STT (Faster-Whisper) ---
        print(f"Loading Faster-Whisper ({WHISPER_MODEL})...")
        self.stt_model = WhisperModel(
            WHISPER_MODEL, device=self.device, compute_type="float16"
        )

        # --- 3. LLM via OpenAI-compatible API ---
        # Instead of using AsyncLLMEngine directly (which causes tokenizer crashes),
        # we launch vLLM as a subprocess serving an OpenAI-compatible API.
        # This is Modal's recommended pattern and is much more stable.
        print(f"Starting vLLM server for {LLM_MODEL}...")
        import subprocess

        self.vllm_process = subprocess.Popen(
            [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model", LLM_MODEL,
                "--host", "127.0.0.1",
                "--port", "8000",
                "--max-model-len", "2048",
                "--gpu-memory-utilization", "0.5",
                "--enforce-eager",  # Faster startup, good for low-latency
                "--dtype", "bfloat16",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for vLLM server to be ready
        import time
        import urllib.request
        for i in range(120):  # Wait up to 2 minutes
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health")
                print("vLLM server is ready!")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("vLLM server failed to start in time")

        # --- 4. TTS (Piper) ---
        print("Loading Piper TTS...")
        from piper import PiperVoice
        self.tts_voice = PiperVoice.load("/models/piper/en_US-lessac-medium.onnx")

        print("All models loaded successfully!")

    @modal.exit()
    def cleanup(self):
        """Clean up the vLLM subprocess on container exit."""
        if hasattr(self, "vllm_process"):
            self.vllm_process.terminate()
            self.vllm_process.wait()

    @modal.asgi_app()
    def asgi_app(self):
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        web_app = FastAPI()

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        @web_app.websocket("/ai-stream")
        async def ai_stream(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connected.")

            import audioop
            import numpy as np
            import uuid
            import re
            import httpx

            # System prompt
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
            SAMPLE_RATE = 8000

            async def generate_response(user_text: str):
                """Call the local vLLM OpenAI-compatible API and stream TTS back."""
                chat_history.append({"role": "user", "content": user_text})

                # Call vLLM's OpenAI-compatible endpoint
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://127.0.0.1:8000/v1/chat/completions",
                        json={
                            "model": LLM_MODEL,
                            "messages": chat_history,
                            "max_tokens": 150,
                            "temperature": 0.7,
                            "stream": False,  # Set True for streaming if needed
                        },
                        timeout=30.0,
                    )
                    result = response.json()
                    full_response = result["choices"][0]["message"]["content"].strip()

                print(f"LLM response: {full_response}")
                chat_history.append({"role": "assistant", "content": full_response})

                # Split into sentences and synthesize each
                sentences = re.split(r"(?<=[.!?])\s+", full_response)
                for sentence in sentences:
                    if sentence.strip():
                        await synthesize_and_send(sentence.strip(), websocket)

            async def synthesize_and_send(text: str, ws: WebSocket):
                """Generate TTS audio and send as 8kHz mu-law."""
                import audioop

                audio_stream = self.tts_voice.synthesize_stream_raw(text)

                for pcm_chunk in audio_stream:
                    # Resample from 22050Hz to 8000Hz
                    converted_audio, _ = audioop.ratecv(
                        pcm_chunk, 2, 1, 22050, 8000, None
                    )
                    # Convert PCM to mu-law
                    mulaw_audio = audioop.lin2ulaw(converted_audio, 2)

                    await ws.send_text(
                        json.dumps(
                            {
                                "event": "media",
                                "media": {
                                    "payload": base64.b64encode(mulaw_audio).decode(
                                        "utf-8"
                                    )
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

                        # Twilio: 8kHz mu-law -> PCM 16-bit
                        pcm_data = audioop.ulaw2lin(chunk, 2)
                        audio_buffer.extend(pcm_data)

                        if len(audio_buffer) >= 2048:
                            np_array = (
                                np.frombuffer(audio_buffer, dtype=np.int16).astype(
                                    np.float32
                                )
                                / 32768.0
                            )
                            import torch
                            import torchaudio

                            tensor_chunk = torch.from_numpy(np_array)

                            # Resample to 16kHz for Silero VAD
                            tensor_chunk_16k = torchaudio.transforms.Resample(
                                8000, 16000
                            )(tensor_chunk)
                            speech_prob = self.vad_model(
                                tensor_chunk_16k, 16000
                            ).item()

                            if speech_prob > 0.5:
                                is_speaking = True
                                silence_chunks = 0
                            else:
                                if is_speaking:
                                    silence_chunks += 1

                            # ~1 second of silence = end of speech
                            if is_speaking and silence_chunks > 5:
                                print("Speech ended. Transcribing...")
                                is_speaking = False
                                silence_chunks = 0

                                # Transcribe with Whisper
                                full_np = (
                                    np.frombuffer(
                                        audio_buffer, dtype=np.int16
                                    ).astype(np.float32)
                                    / 32768.0
                                )
                                full_tensor = torch.from_numpy(full_np)
                                full_16k = torchaudio.transforms.Resample(
                                    8000, 16000
                                )(full_tensor)

                                segments, _ = self.stt_model.transcribe(
                                    full_16k.numpy(), beam_size=1
                                )
                                transcript = " ".join(
                                    [seg.text for seg in segments]
                                ).strip()

                                print(f"Caller said: {transcript}")
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