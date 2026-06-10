"""In-process rewrite generation on Modal (no HTTP server — robust, one container).

Generate predictions for the rewrite test set with one of three model configs, then
SCORE LOCALLY with the repo's source-of-truth heuristics (uxft.eval.score_rewrite via
eval/score_v3_eval.py) — the container stays a clean ML-only image.

    # prompt-only BASELINE (base, no adapter):
    modal run modal_app/eval_rewrite.py --out base_preds.jsonl

    # 1a text adapter (base + LoRA):
    modal run modal_app/eval_rewrite.py \
        --model-repo gr33r/ux-writing-2.0-rewrite-qwen36 --is-adapter --out text_preds.jsonl

    # 1b combined adapter (base + LoRA):
    modal run modal_app/eval_rewrite.py \
        --model-repo gr33r/ux-writing-2.0-combined-qwen36 --is-adapter --out combined_preds.jsonl

    # a merged standalone (no PEFT) is just is_adapter=False with the merged repo:
    modal run modal_app/eval_rewrite.py --model-repo gr33r/ux-writing-2.0-rewrite-qwen36-merged

Then score locally (predictions carry raw assistant JSON {rewrite,reason,risk}, parsed by
the scorer):
    python3 eval/score_v3_eval.py            # or uxft.eval.score_rewrite directly

The rewrite JSON contract (system+user prompt, assistant {rewrite,reason,risk}) mirrors
uxft/benchmark_model.py; the load/generate plumbing follows the Modal patterns in common.py.
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

DATASET_REPO = "gr33r/ux-writing-sft"
MAX_NEW_TOKENS = 256


@app.function(
    gpu=GPU,
    volumes={MODEL_CACHE: weights_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def generate_rewrites(
    model_repo: str = BASE_MODEL,
    is_adapter: bool = False,
    base_model: str = BASE_MODEL,
    limit: int | None = None,
    tag: str = "model",
    max_new_tokens: int = MAX_NEW_TOKENS,
    enable_thinking: bool = True,
):
    import json
    import os
    import tempfile

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # --- model: merged/standalone (is_adapter=False) or base + LoRA (is_adapter=True) ---
    src = base_model if is_adapter else model_repo
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, cache_dir=HF_CACHE)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        src,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    if is_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_repo)
    model = model.eval()

    # --- data: rewrite rows from the test split ---
    data = load_dataset(DATASET_REPO, split="test")
    data = data.filter(lambda r: r["task"] == "rewrite")
    if limit:
        data = data.select(range(min(limit, len(data))))
    print(json.dumps({"model_repo": model_repo, "is_adapter": is_adapter, "rows": len(data)}))

    preds = []
    for row in data:
        msgs = json.loads(row["messages_json"])
        # prompt = system+user; drop the gold assistant turn.
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        gold = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

        enc = processor.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,  # Qwen3 thinking toggle; False = direct answer
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        pred = processor.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        preds.append({
            "id": row["id"],
            "prediction": pred,  # raw string; scored locally
            "gold": gold,
            "metadata": json.loads(row["metadata_json"]),
        })
        print(f"{row['id']:<40} {pred[:80]}")

    # Persist to the Hub so results survive even if the local client disconnects.
    from huggingface_hub import HfApi

    tmp = os.path.join(tempfile.gettempdir(), f"{tag}.jsonl")
    with open(tmp, "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    HfApi().upload_file(
        path_or_fileobj=tmp,
        path_in_repo=f"eval_preds/{tag}.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"uploaded -> hf://datasets/{DATASET_REPO}/eval_preds/{tag}.jsonl")
    return preds


@app.local_entrypoint()
def main(
    model_repo: str = BASE_MODEL,
    is_adapter: bool = False,
    tag: str = "model",
    limit: int = 0,
    max_new_tokens: int = MAX_NEW_TOKENS,
    enable_thinking: bool = True,
):
    # .spawn() + `modal run --detach`: fire-and-forget so the run survives a client
    # disconnect. Predictions are persisted to the Hub (eval_preds/<tag>.jsonl); fetch +
    # score locally with eval/score_rewrite_preds.py.
    call = generate_rewrites.spawn(
        model_repo=model_repo, is_adapter=is_adapter, tag=tag,
        limit=limit or None, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking,
    )
    print(f"spawned eval gen tag={tag}; fc={call.object_id}; "
          f"results -> hf://datasets/{DATASET_REPO}/eval_preds/{tag}.jsonl")
