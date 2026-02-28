import os
import io
import asyncio
import json
import base64
import wave
import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Define the Modal application
app = modal.App("hackillinois-voice-pipeline")

# Create a container image with all necessary dependencies
# vllm requires a specific modal CUDA image to build properly
voice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "wget", "git")
    .pip_install(
        "fastapi",
        "websockets",
        "uvicorn",
        "faster-whisper",
        "vllm==0.6.3.post1",
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "numpy<2.0",
    )
    # Download a small piper-tts model for voice synthesis
    .run_commands([
        "mkdir -p /models/piper",
        "wget -O /models/piper/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true",
        "wget -O /models/piper/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true",
        "pip install piper-tts"
    ])
)

@app.cls(image=voice_image, gpu="H200", min_containers=1)
@modal.concurrent(max_inputs=20)
class VoicePipeline:
    @modal.enter()
    def setup_models(self):
        print("Initializing models...")
        import torch
        from faster_whisper import WhisperModel
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from piper import PiperVoice

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load VAD (Voice Activity Detection)
        print("Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = self.vad_utils[0]

        # 2. Load STT (Faster-Whisper)
        print("Loading Faster-Whisper...")
        self.stt_model = WhisperModel("tiny.en", device=self.device, compute_type="float16")

        # 3. Load LLM (vLLM)
        # We use a very small, fast model to keep latency low. You can switch to meta-llama/Meta-Llama-3-8B-Instruct
        print("Loading vLLM Engine...")
        engine_args = AsyncEngineArgs(
            model="microsoft/Phi-3.5-mini-instruct",
            gpu_memory_utilization=0.5,
            max_model_len=2048,
            enforce_eager=True, # Good for low-latency streaming
            tokenizer_mode="slow"
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 4. Load TTS (Piper)
        print("Loading Piper TTS...")
        self.tts_voice = PiperVoice.load("/models/piper/en_US-lessac-medium.onnx")
        
        print("All models loaded successfully.")

    @modal.asgi_app()
    def asgi_app(self):
        web_app = FastAPI()

        @web_app.websocket("/ai-stream")
        async def ai_stream(websocket: WebSocket):
            await websocket.accept()
            print("WebSocket connected from DigitalOcean Proxy.")
            import audioop
            import numpy as np
            import uuid
            import re
            
            # System prompt for the triage/dispatch agent
            chat_history = [
                {"role": "system", "content": "You are a 911 dispatch assistant. Be extremely concise. Ask short questions. You are talking to a caller reporting an emergency."}
            ]
            
            audio_buffer = bytearray()
            is_speaking = False
            silence_chunks = 0
            
            # Twilio audio is 8000Hz, mu-law encoded
            SAMPLE_RATE = 8000
            
            async def generate_response(user_text: str):
                """Passes text to LLM, streams words, and synthesized speech back to the websocket."""
                from vllm import SamplingParams
                chat_history.append({"role": "user", "content": user_text})
                
                # Format prompt based on the chosen model (vLLM handles this if chat template is supported, but manual formatting is safer for simple models)
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) + "\nassistant: "
                
                request_id = str(uuid.uuid4())
                sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
                
                results_generator = self.llm_engine.generate(prompt, sampling_params, request_id)
                
                full_response = ""
                current_sentence = ""
                
                # Stream the LLM response
                async for request_output in results_generator:
                    text = request_output.outputs[0].text
                    new_text = text[len(full_response):]
                    full_response = text
                    current_sentence += new_text
                    
                    # Synthesize audio when we have a full sentence
                    if re.search(r'[.!?]\s', current_sentence) or request_output.finished:
                        if current_sentence.strip():
                            print(f"Synthesizing: {current_sentence.strip()}")
                            await synthesize_and_send(current_sentence.strip(), websocket)
                            current_sentence = ""
                
                chat_history.append({"role": "assistant", "content": full_response.strip()})

            async def synthesize_and_send(text: str, ws: WebSocket):
                """Generate TTS audio and send to the websocket encoded as 8kHz mu-law."""
                import audioop
                # Generate PCM 16-bit audio at 22050Hz (Piper default)
                audio_stream = self.tts_voice.synthesize_stream_raw(text)
                
                # Read chunks, resample to 8000Hz, encode to mu-law, and send
                for pcm_chunk in audio_stream:
                    # pcm_chunk is bytes of 16-bit PCM at 22050Hz
                    # We need to resample from 22050Hz to 8000Hz
                    converted_audio, _ = audioop.ratecv(pcm_chunk, 2, 1, 22050, 8000, None)
                    # Convert PCM to mu-law
                    mulaw_audio = audioop.lin2ulaw(converted_audio, 2)
                    
                    # Send payload
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(mulaw_audio).decode("utf-8")
                        }
                    }))

            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    if data["event"] == "start":
                        print("Stream started")
                        
                    elif data["event"] == "media":
                        payload = data["media"]["payload"]
                        chunk = base64.b64decode(payload)
                        
                        # Twilio sends 8kHz mu-law audio. Convert to PCM 16-bit for VAD/STT
                        pcm_data = audioop.ulaw2lin(chunk, 2)
                        audio_buffer.extend(pcm_data)
                        
                        # Need at least 512 samples for VAD at 8000Hz (~64ms)
                        # Twilio typically sends 20ms chunks (160 samples)
                        if len(audio_buffer) >= 2048: # Wait for a larger buffer to process
                            # Convert to numpy float32 array in range [-1.0, 1.0]
                            np_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            import torch
                            import torchaudio
                            tensor_chunk = torch.from_numpy(np_array)
                            
                            # Simple Voice Activity Detection check
                            # VAD model works best on 16000Hz, but we can try 8000Hz.
                            # For better accuracy, we should resample to 16000Hz for Silero VAD.
                            tensor_chunk_16k = torchaudio.transforms.Resample(8000, 16000)(tensor_chunk)
                            speech_prob = self.vad_model(tensor_chunk_16k, 16000).item()
                            
                            if speech_prob > 0.5:
                                is_speaking = True
                                silence_chunks = 0
                            else:
                                if is_speaking:
                                    silence_chunks += 1
                                    
                            # If caller stops speaking for ~1 second (e.g., 5 chunks of silence)
                            if is_speaking and silence_chunks > 5:
                                print("Speech ended. Processing chunk...")
                                is_speaking = False
                                silence_chunks = 0
                                
                                # Convert the entire accumulated buffer to 16000Hz float32 for Whisper
                                full_np_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                                full_tensor = torch.from_numpy(full_np_array)
                                full_tensor_16k = torchaudio.transforms.Resample(8000, 16000)(full_tensor)
                                
                                segments, _ = self.stt_model.transcribe(full_tensor_16k.numpy(), beam_size=1)
                                transcript = " ".join([seg.text for seg in segments]).strip()
                                
                                print(f"Caller said: {transcript}")
                                
                                # Clear buffer
                                audio_buffer.clear()
                                
                                if transcript:
                                    # Forward transcript to LLM and send TTS back
                                    asyncio.create_task(generate_response(transcript))
                                    
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
