#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.21.0",
#     "peft>=0.14.0",
#     "transformers>=4.57.0",
#     "datasets>=2.18.0",
#     "bitsandbytes>=0.45.0",
#     "accelerate>=1.0.0",
# ]
# ///
"""Fine-tune ux-writing-1 on YOUR style-guide pairs — built for Hugging Face Jobs.

QLoRA (4-bit NF4, LoRA r=16 α=32 on the LM projections) starting from gr33r/ux-writing-1,
so you inherit the UX-writing tune and add your product's voice on top. See
docs/FINETUNE_GUIDE.md for how to build the dataset and (crucially) how to blind-test
the result before shipping it.

  hf jobs uv run --detach --flavor a100-large --timeout 4h --secrets HF_TOKEN \
    --env DATASET_REPO=yourco/ux-writing-pairs \
    --env HUB_MODEL_ID=yourco/ux-writing-1-yourco \
    https://raw.githubusercontent.com/content-designer/ux-writing-1/main/scripts/train_on_your_styleguide.py

Smoke test first (~$1): add --env MAX_STEPS=8 (trains 8 steps, pushes nothing).
"""

from __future__ import annotations

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("BASE_MODEL", "gr33r/ux-writing-1")
DATASET_REPO = os.environ["DATASET_REPO"]              # required
HUB_MODEL_ID = os.environ["HUB_MODEL_ID"]              # required, e.g. yourco/ux-writing-1-yourco
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))     # >0 = smoke test, no push
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))


def main() -> int:
    smoke = MAX_STEPS > 0
    print(json.dumps({"base": BASE_MODEL, "dataset": DATASET_REPO, "hub_model_id": HUB_MODEL_ID,
                      "smoke": smoke, "epochs": EPOCHS, "lr": LEARNING_RATE}, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(DATASET_REPO)
    train_ds = data["train"]
    if "validation" in data:
        eval_ds = data["validation"]
    else:
        split = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir="ux-writing-styleguide-tune",
        push_to_hub=not smoke,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
        bf16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=None if smoke else eval_ds,
        args=config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    if smoke:
        print("smoke test complete (nothing pushed)")
    else:
        trainer.push_to_hub()
        print(f"adapter pushed -> https://huggingface.co/{HUB_MODEL_ID}")
        print("NEXT: blind-test it against ux-writing-1 before shipping — docs/FINETUNE_GUIDE.md §3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
