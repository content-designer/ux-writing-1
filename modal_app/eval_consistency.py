"""Vision consistency eval on Modal — adapter-vs-base on the SAME real screenshots.

Port of scripts/eval_consistency_adapter.py, SELF-CONTAINED (no cross-file imports). Same prompt,
same screens; the only difference is whether the LoRA adapter is enabled (model.disable_adapter()
gives the prompt-only base on the identical screen). The container returns RAW adapter/base
outputs; the deterministic consistency_postfilter is applied LOCALLY by the scorer.

    modal run modal_app/eval_consistency.py \
        --adapter-repo gr33r/ux-writing-2.0-combined-qwen36 --out consistency_preds.jsonl

Then score locally (postfilter + provenance-sliced recall):
    python3 eval/score_real_bug_eval.py consistency_preds.jsonl

The serving contract is load-bearing and mirrored exactly from serve_consistency.py /
eval_consistency_adapter.py: issues-only SYSTEM prompt, MAX_LONG_SIDE=1792. The deterministic
postfilter runs locally (eval/score_real_bug_eval.py), not in-container.
"""

import os

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
MAX_LONG_SIDE = 1792  # load-bearing: dense real screens lose small-text detection below this
MAX_NEW_TOKENS = 2048

# Issues-only system prompt. Must match serve_consistency.SYSTEM_PROMPT /
# eval_consistency_adapter.V14_SYSTEM — changing the framing breaks the adapter.
V14_SYSTEM = (
    "You are a senior UX writer doing a SCREEN-LEVEL CONSISTENCY review of one screen.\n"
    "Your job is NOT to restyle individual strings. Find problems that exist only in the "
    "RELATIONSHIPS between strings on the screen:\n"
    "1. Duplication — two strings that say the same thing, where one should add new value "
    "(e.g., an empty-state body that just repeats the page subtitle).\n"
    "2. Terminology/brand inconsistency — the same concept named differently (e.g., 'Team' "
    "vs 'Teams'), or a brand spelled wrong/inconsistently (e.g., 'Github' vs 'GitHub').\n"
    "3. Number/plural agreement (e.g., '1 members').\n"
    "4. Tone clash — one string whose register fights the rest of the screen (e.g., a joke "
    "on a destructive/irreversible action).\n"
    "5. Casing inconsistency across parallel elements (e.g., 'Save changes' vs 'Save Changes').\n"
    'Return compact JSON: {"issues": [{"type": str, "strings": [str], "problem": str, '
    '"fix": str}]}. Empty issues list if none.'
)


@app.function(
    gpu=GPU,
    volumes={MODEL_CACHE: weights_vol},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def generate_consistency(
    adapter_repo: str = "gr33r/ux-writing-2.0-combined-qwen36",
    base_model: str = BASE_MODEL,
):
    import json

    import torch
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # postfilter is applied locally by eval/score_real_bug_eval.py

    # Download the dataset repo once -> gives us the local screen image files.
    ds_dir = snapshot_download(DATASET_REPO, repo_type="dataset", cache_dir=HF_CACHE)

    data = load_dataset(DATASET_REPO, split="test")
    data = data.filter(lambda r: r["task"] == "consistency")
    if len(data) == 0:
        print(
            "[info] the test split has NO consistency rows yet — the vision-test set hasn't "
            "been annotated. data_build/annotate_new_eval.py adds it. Returning empty."
        )
        return []
    print(json.dumps({"adapter_repo": adapter_repo, "rows": len(data)}))

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, cache_dir=HF_CACHE)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    model = PeftModel.from_pretrained(base, adapter_repo).eval()

    def gen(img, surface):
        user = "Do a screen-level consistency review."
        if surface:
            user = f"Surface: {surface}\n{user}"
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": V14_SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": user}]},
        ]
        enc = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    rows = []
    for r in data:
        meta = json.loads(r["metadata_json"])
        surface = meta.get("surface", "")
        img = Image.open(os.path.join(ds_dir, r["image"])).convert("RGB")
        w, h = img.size
        s = MAX_LONG_SIDE / max(w, h)
        if s < 1.0:
            img = img.resize((round(w * s), round(h * s)))

        adapter_out = gen(img, surface)
        with model.disable_adapter():
            base_out = gen(img, surface)

        # Return RAW model outputs; postfilter is applied locally by
        # eval/score_real_bug_eval.py. Pass through the manifest/gold fields
        # (gold_types, clean, provenance, split, ...).
        rows.append({
            **meta,
            "id": r["id"],
            "surface": surface,
            "adapter_out": adapter_out,
            "base_out": base_out,
        })
        print(f"{r['id']:<22} surface={surface}")

    return rows


@app.local_entrypoint()
def main(
    adapter_repo: str = "gr33r/ux-writing-2.0-combined-qwen36",
    out: str = "consistency_preds.jsonl",
):
    import json

    rows = generate_consistency.remote(adapter_repo=adapter_repo)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} predictions to {out}")
    print("score locally: python3 eval/score_real_bug_eval.py " + out)
