"""
Voice Pipeline on Modal — Verified Against Official Modal Examples
===================================================================
Verified against:
  - modal-labs/quillman (Moshi voice chatbot)
  - modal-projects/open-source-av-ragbot (low-latency voice RAGbot)
  - modal-labs/modal-examples vllm_inference.py

Key patterns from Modal's production code:
  1. nvidia/cuda base image with cudnn (not debian_slim)
  2. vLLM as subprocess on a SEPARATE port from the ASGI app
  3. Modal Volumes for weight caching + compilation artifacts
  4. HF_HUB_ENABLE_HF_TRANSFER=1 for fast model downloads
  5. Don't pipe subprocess stderr to PIPE (causes deadlock)

Architecture: Single-container for hackathon. Production = split services.
"""

import os
import json
import base64
import asyncio
import modal

# ---------------------------------------------------------------------------
# Modal App & Volumes
# ---------------------------------------------------------------------------
app = modal.App("hackillinois-voice-pipeline")

# Two volumes, matching Modal's own vLLM example pattern:
#   - One for HuggingFace model weights
#   - One for vLLM compilation/CUDA graph cache
weights_volume = modal.Volume.from_name("voice-pipeline-weights", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("voice-pipeline-vllm-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container Image
# ---------------------------------------------------------------------------
# Modal's ragbot uses nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 with Python 3.12
# Their vLLM inference example also uses an nvidia CUDA base.
# CRITICAL: Do NOT install torch separately — vLLM bundles its own compatible torch.
voice_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "wget", "git", "curl")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Modal pattern: fast HF downloads
    .pip_install(
        # vLLM — let it pull its own torch + transformers (no manual pins!)
        "vllm==0.8.5.post1",
        # STT
        "faster-whisper>=1.1.0",
        # Audio
        "torchaudio",
        "numpy<2.0",
        # Web framework + HTTP client
        "fastapi==0.115.5",
        "uvicorn[standard]",
        "httpx",
        # Fast HF downloads
        "hf_transfer",
        "huggingface_hub[hf-xet]",
    )
    # Download Piper TTS model at image build time
    .run_commands(
        "mkdir -p /models/piper",
        'wget -q -O /models/piper/en_US-lessac-medium.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true"',
        'wget -q -O /models/piper/en_US-lessac-medium.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"',
        "pip install piper-tts",
    )
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Qwen2.5-3B-Instruct is well-tested with vLLM and fast enough for voice.
# Modal's ragbot uses Qwen3-4B-Instruct for their voice bot.
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
WHISPER_MODEL = "tiny.en"

# CRITICAL: vLLM subprocess must use a DIFFERENT port than Modal's ASGI server.
# @modal.asgi_app() uses Modal's internal port (typically 8000).
# We put vLLM on 8001 to avoid conflicts.
VLLM_PORT = 8001


@app.cls(
    image=voice_image,
    gpu="A100",  # A100 is plenty for a 3B model; cheaper than H200
    volumes={
        "/root/.cache/huggingface": weights_volume,    # HF model weights
        "/root/.cache/vllm": vllm_cache_volume,        # vLLM compilation cache
    },
    scaledown_window=300,  # Keep container warm for 5 min (Modal's pattern name)
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

        # --- 3. LLM (vLLM as subprocess — Modal's recommended pattern) ---
        # Modal's own vllm_inference.py launches vLLM via subprocess.Popen
        # with @modal.web_server. We do the same but on a local-only port
        # since we already have an ASGI app serving the WebSocket.
        print(f"Starting vLLM server for {LLM_MODEL} on port {VLLM_PORT}...")
        import subprocess
        import sys

        self.vllm_process = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", LLM_MODEL,
                "--host", "127.0.0.1",
                "--port", str(VLLM_PORT),
                "--max-model-len", "2048",
                "--gpu-memory-utilization", "0.5",
                "--enforce-eager",
                "--dtype", "bfloat16",
            ],
            # CRITICAL: Do NOT use stdout=PIPE, stderr=PIPE together.
            # The pipe buffer fills up and causes a deadlock.
            # Modal's examples use subprocess.DEVNULL or just let it inherit.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Poll the health endpoint until vLLM is ready
        import time
        import urllib.request

        health_url = f"http://127.0.0.1:{VLLM_PORT}/health"
        for attempt in range(180):  # Up to 3 minutes for model download + load
            try:
                urllib.request.urlopen(health_url)
                print("vLLM server is ready!")
                break
            except Exception:
                # Check if the process died
                if self.vllm_process.poll() is not None:
                    raise RuntimeError(
                        f"vLLM process exited with code {self.vllm_process.returncode}"
                    )
                time.sleep(1)
        else:
            self.vllm_process.terminate()
            raise RuntimeError("vLLM server failed to start within 3 minutes")

        # --- 4. TTS (Piper) ---
        print("Loading Piper TTS...")
        from piper import PiperVoice

        self.tts_voice = PiperVoice.load("/models/piper/en_US-lessac-medium.onnx")

        print("All models loaded. Pipeline ready.")

    @modal.exit()
    def cleanup(self):
        """Terminate vLLM subprocess on container shutdown."""
        if hasattr(self, "vllm_process") and self.vllm_process.poll() is None:
            self.vllm_process.terminate()
            self.vllm_process.wait(timeout=10)

    # Modal's QuiLLMan and ragbot both use @modal.asgi_app() to serve FastAPI
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
            print("WebSocket client connected.")

            import audioop
            import numpy as np
            import re
            import httpx

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
                """Call the local vLLM OpenAI-compatible API, then TTS the response."""
                chat_history.append({"role": "user", "content": user_text})

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
                        json={
                            "model": LLM_MODEL,
                            "messages": chat_history,
                            "max_tokens": 150,
                            "temperature": 0.7,
                            "stream": False,
                        },
                        timeout=30.0,
                    )
                    result = response.json()
                    full_response = (
                        result["choices"][0]["message"]["content"].strip()
                    )

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