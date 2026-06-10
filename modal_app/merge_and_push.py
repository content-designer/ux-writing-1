"""Merge a LoRA adapter into the base and push a STANDALONE merged model on Modal.

A merged model serves without PEFT (vLLM / Inference Endpoints). bf16, NOT quantized —
quantizing before a merge corrupts the weights, so we load the clean base on an A100-80GB
(27.8B bf16 fits 80 GB) and merge_and_unload.

    modal run --detach modal_app/merge_and_push.py --src text      # -> gr33r/ux-writing-1
    modal run --detach modal_app/merge_and_push.py --src combined  # -> gr33r/ux-writing-2.0-combined-qwen36-merged

Mirrors scripts/merge_vision_adapter.py (load base bf16 -> PeftModel -> merge_and_unload ->
save model+processor -> push private), ported to the established Modal patterns (common.py).
"""

import modal

# SELF-CONTAINED on purpose: app, image, volume, and secret are defined inline (no
# cross-file imports) so the Modal container never hits a ModuleNotFoundError when it
# re-imports this module to run the function. Mirrors modal_app/train.py exactly.
BASE_MODEL = "Qwen/Qwen3.6-27B"
GPU = "A100-80GB"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.57.0",
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

app = modal.App("ux-writing-finetune", image=image)
weights_vol = modal.Volume.from_name("qwen36-27b-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

# src -> (adapter repo, merged output repo). The text adapter is the released flagship
# family (renamed from ux-writing-2.0-rewrite-qwen36): merged model = gr33r/ux-writing-1.
MERGE_TARGETS = {
    "text": ("gr33r/ux-writing-1-lora", "gr33r/ux-writing-1"),
    "combined": ("gr33r/ux-writing-2.0-combined-qwen36", "gr33r/ux-writing-2.0-combined-qwen36-merged"),
}


@app.function(
    gpu="A100-80GB",  # 27.8B bf16 clean merge fits 80 GB
    volumes={MODEL_CACHE: weights_vol},
    secrets=[hf_secret],
    timeout=45 * 60,
)
def merge(src: str = "text", base_model: str = BASE_MODEL):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert src in ("text", "combined"), src
    adapter_repo, out_repo = MERGE_TARGETS[src]

    print(f"[1/4] load base {base_model} (bf16, clean) + adapter {adapter_repo}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    model = PeftModel.from_pretrained(model, adapter_repo)

    print("[2/4] merge_and_unload")
    model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out = "/tmp/merged"
    print(f"[3/4] save -> {out}")
    model.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)

    # Upload the saved folder directly (transformers 5.x push_to_hub dropped the
    # safe_serialization kwarg, and upload_folder avoids re-serializing 56 GB).
    print(f"[4/4] upload {out} -> {out_repo} (private)")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(out_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=out_repo)
    print(f"done -> https://huggingface.co/{out_repo}")


@app.local_entrypoint()
def main(src: str = "text"):
    # .spawn() + `modal run --detach`: fire-and-forget so the merge survives a client
    # disconnect (a .remote() call gets canceled if the local handle dies).
    call = merge.spawn(src)
    print(f"spawned merge src={src}; fc={call.object_id}")
