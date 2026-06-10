"""QLoRA SFT for Qwen3.6-27B on Modal. One entrypoint, two runs.

SELF-CONTAINED on purpose: app, image, volume, and secret are defined inline (no
cross-file imports) so the Modal container never hits a ModuleNotFoundError when it
re-imports this module to run the function.

    modal run modal_app/train.py --run-mode combined --max-steps 8 --no-push   # smoke (~$1)
    modal run modal_app/train.py --run-mode text                               # 1a: text baseline
    modal run modal_app/train.py --run-mode combined                           # 1b: rewrite + vision

Design (ported from the proven snapshot scripts):
  - 4-bit NF4 QLoRA on AutoModelForImageTextToText (train_sft_gemma4_qlora.py).
  - Mixed-modality collator: text-only AND image+text rows in one run; text rows pass an
    empty image list. per_device_batch=1 means each forward is a single row, never ragged.
  - LoRA on the LM projections only; vision tower frozen. max_length=None for VLM rows
    (never truncate image tokens); image long-side capped instead.
"""

import modal

BASE_MODEL = "Qwen/Qwen3.6-27B"
GPU = "A100-80GB"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"
DATASET_REPO = "gr33r/ux-writing-sft"
HUB_IDS = {
    "text": "gr33r/ux-writing-2.0-rewrite-qwen36",
    "combined": "gr33r/ux-writing-2.0-combined-qwen36",
}
IMAGE_LONG_SIDE = 1792  # load-bearing for small-text legibility (serve_consistency.py)

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


@app.function(
    gpu=GPU,
    volumes={MODEL_CACHE: weights_vol},
    secrets=[hf_secret],
    timeout=8 * 60 * 60,
    retries=modal.Retries(max_retries=1),
)
def train(run_mode: str = "text", max_steps: int = -1, push: bool = True):
    import json
    import os
    from collections import Counter

    import torch
    import torch.nn as nn
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    assert run_mode in ("text", "combined"), run_mode
    smoke = bool(max_steps and max_steps > 0)
    push = push and not smoke
    learning_rate = 2e-4 if run_mode == "text" else 1e-4
    max_length = 2048 if run_mode == "text" else None  # None required when images are present

    ds_dir = snapshot_download(DATASET_REPO, repo_type="dataset", cache_dir=HF_CACHE)
    data = load_dataset(DATASET_REPO)

    def take(split):
        d = data[split]
        if run_mode == "text":
            d = d.filter(lambda r: r["task"] == "rewrite")
        return d

    train_ds, eval_ds = take("train"), take("validation")
    print(json.dumps({
        "run_mode": run_mode, "train": len(train_ds), "eval": len(eval_ds),
        "smoke": smoke, "will_push": push, "lr": learning_rate, "max_length": max_length,
    }, indent=2))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    weights_vol.commit()  # persist the ~56GB base weights in HF_HOME for later runs

    # Diagnostic: confirm the LM linear module names match our LoRA targets (folds in arch_check).
    lm_suffixes = Counter()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}:
            if not any(t in name.lower() for t in ("visual", "vision", "vit", "image_encoder")):
                lm_suffixes[name.split(".")[-1]] += 1
    print("LM linear suffixes:", dict(lm_suffixes))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    image_token_id = getattr(model.config, "image_token_id", None)

    def resize(im):
        w, h = im.size
        scale = IMAGE_LONG_SIDE / max(w, h)
        return im.resize((round(w * scale), round(h * scale))) if scale < 1.0 else im

    def collate(examples):
        msgs = [json.loads(ex["messages_json"]) for ex in examples]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs]
        if any(ex.get("image") for ex in examples):
            images = [
                [resize(Image.open(os.path.join(ds_dir, ex["image"])).convert("RGB"))]
                if ex.get("image") else []
                for ex in examples
            ]
            batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        else:
            batch = processor(text=texts, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=f"ux-writing-{run_mode}",
        push_to_hub=False,
        num_train_epochs=2,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        bf16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        # In-training eval disabled: computing full logits over the ~248K-token vocab for a
        # long vision row OOMs an 80GB A100. We evaluate separately on the held-out benchmark.
        eval_strategy="no",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_length,
        report_to="none" if smoke else "trackio",
        project="ux-writing-finetune",
        run_name=f"qwen36-27b-{run_mode}",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None,
        args=cfg,
        peft_config=peft_config,
        data_collator=collate,
    )
    trainer.train()

    if push:
        repo = HUB_IDS[run_mode]
        trainer.model.push_to_hub(repo, private=True)
        print(f"pushed adapter -> https://huggingface.co/{repo}")
    else:
        print("smoke / no-push complete")


@app.local_entrypoint()
def main(run_mode: str = "text", max_steps: int = -1, push: bool = True):
    # .spawn() is fire-and-forget: with `modal run --detach`, the run survives even if the
    # local client disconnects/is killed (a .remote() call gets canceled on disconnect).
    call = train.spawn(run_mode=run_mode, max_steps=max_steps, push=push)
    print(f"spawned {run_mode} run; function_call_id={call.object_id}")
