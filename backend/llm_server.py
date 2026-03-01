import modal
import subprocess
import sys

app = modal.App("hackillinois-llm")

llm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm==0.8.5.post1", "hf_transfer", "transformers>=4.51.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONIOENCODING": "utf8"})
)

weights_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("llm-vllm-cache", create_if_missing=True)

@app.function(
    image=llm_image,
    gpu="A100",
    volumes={
        "/root/.cache/huggingface": weights_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    min_containers=1,
    scaledown_window=300,
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "Qwen/Qwen2.5-3B-Instruct",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
        "--dtype", "bfloat16"
    ]
    subprocess.Popen(cmd).wait()
