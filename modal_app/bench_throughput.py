"""Measure batched review throughput for ux-writing-1 on one A100 — honest $/1K strings.

Loads the merged model bf16, reviews ~200 held-out rewrite prompts with batched greedy
generation (direct mode, the production scan configuration), and reports strings/hour
and $/1K strings at the A100-80GB list price. Results persist to the dataset repo so
they survive client disconnects.

    modal run --detach modal_app/bench_throughput.py
"""

import modal

MERGED_REPO = "gr33r/ux-writing-1"
DATASET_REPO = "gr33r/ux-writing-sft"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"
A100_USD_PER_HOUR = 2.50  # Modal list price, June 2026
BATCH_SIZE = 16
MAX_NEW_TOKENS = 192
N_STRINGS = 192  # 12 batches of 16

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.57.0", "accelerate>=1.0.0",
        "datasets>=2.18.0", "huggingface_hub>=0.34.0", "hf_transfer",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("uxw1-bench", image=image)
weights_vol = modal.Volume.from_name("qwen36-27b-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(gpu="A100-80GB", volumes={MODEL_CACHE: weights_vol}, secrets=[hf_secret],
              timeout=90 * 60)
def bench():
    import json
    import time

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MERGED_REPO, trust_remote_code=True, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation
    model = AutoModelForImageTextToText.from_pretrained(
        MERGED_REPO, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE,
    ).eval()

    # ~200 realistic prompts: the rewrite test split, cycled to N_STRINGS
    data = load_dataset(DATASET_REPO, split="test").filter(lambda r: r["task"] == "rewrite")
    rows = [json.loads(r["messages_json"]) for r in data]
    prompts = []
    while len(prompts) < N_STRINGS:
        prompts.extend(rows)
    prompts = prompts[:N_STRINGS]

    texts = [
        tok.apply_chat_template([m for m in msgs if m["role"] != "assistant"],
                                add_generation_prompt=True, tokenize=False,
                                enable_thinking=False)
        for msgs in prompts
    ]

    # warmup (excluded from timing): one batch
    warm = tok(texts[:BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)

    valid = 0
    out_tokens = 0
    t0 = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        out_tokens += int((new != tok.pad_token_id).sum())
        for row in tok.batch_decode(new, skip_special_tokens=True):
            tail = row.rsplit("</think>", 1)[-1]
            s, e = tail.find("{"), tail.rfind("}")
            if s != -1 and e > s:
                try:
                    if isinstance(json.loads(tail[s:e + 1]).get("rewrite"), str):
                        valid += 1
                except json.JSONDecodeError:
                    pass
        print(f"batch {i // BATCH_SIZE + 1}/{len(texts) // BATCH_SIZE} done")
    elapsed = time.time() - t0

    per_hour = len(texts) / elapsed * 3600
    usd_per_1k = 1000 / per_hour * A100_USD_PER_HOUR
    result = {
        "model": MERGED_REPO, "gpu": "A100-80GB", "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS, "mode": "direct (enable_thinking=False), greedy",
        "strings": len(texts), "elapsed_s": round(elapsed, 1),
        "strings_per_hour": round(per_hour),
        "usd_per_1k_strings": round(usd_per_1k, 2),
        "valid_json_rate": round(valid / len(texts), 4),
        "avg_output_tokens": round(out_tokens / len(texts), 1),
        "gpu_usd_per_hour": A100_USD_PER_HOUR,
    }
    print(json.dumps(result, indent=2))
    HfApi().upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(),
                        path_in_repo="eval_preds/throughput.json",
                        repo_id=DATASET_REPO, repo_type="dataset")
    return result


@app.local_entrypoint()
def main():
    call = bench.spawn()
    print(f"spawned throughput bench; fc={call.object_id}")
