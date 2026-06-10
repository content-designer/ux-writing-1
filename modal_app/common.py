"""Shared Modal app, image, volume, secret, and base-weight caching.

All GPU work (train, generate, serve, merge) runs on Modal. Pure-Python orchestration
and scoring run locally with the `uxft` package — the container image deliberately does
NOT depend on `uxft`, so it stays a clean ML-only image.

Setup (one time):
    pip install modal
    python3 -m modal setup                                   # browser auth
    modal secret create hf-token HF_TOKEN=$(cat ~/.cache/huggingface/token)

Cache the base weights into the Volume (avoids re-downloading ~56 GB each run):
    modal run modal_app/common.py::download_weights
"""

import modal

BASE_MODEL = "Qwen/Qwen3.6-27B"
GPU = "A100-80GB"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"

# ML-only image. If bitsandbytes 4-bit has CUDA issues on the GPU host, swap
# debian_slim() for: modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.57.0",  # arch is qwen3_5; pip resolves the newest. arch_check confirms load.
        "trl>=0.21.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "accelerate>=1.0.0",
        "datasets>=2.18.0",
        "trackio",
        "pillow",
        "torchvision",
        "huggingface_hub>=0.34.0",
        "hf_transfer",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Make the local project packages importable INSIDE the container. Modal 1.x requires
# this explicitly — automounting won't put `modal_app`/`uxft` on the container's path,
# which is why `from modal_app.common import ...` fails without it.
image = image.add_local_python_source("modal_app", "uxft")

app = modal.App("ux-writing-finetune", image=image)
weights_vol = modal.Volume.from_name("qwen36-27b-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(volumes={MODEL_CACHE: weights_vol}, secrets=[hf_secret], timeout=60 * 60)
def download_weights(base_model: str = BASE_MODEL):
    """Snapshot the base model into the Volume so later runs read it locally."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(base_model, cache_dir=HF_CACHE)
    weights_vol.commit()
    print(f"cached {base_model} -> {path}")
    return path


# NOTE: no @app.local_entrypoint here. The shared `app` is imported by every runner
# (arch_check/train/eval_*), and Modal requires local-entrypoint names to be unique across
# the app — a `main` here would collide with each runner's `main`. Invoke this directly:
#   python3 -m modal run modal_app/common.py::download_weights
