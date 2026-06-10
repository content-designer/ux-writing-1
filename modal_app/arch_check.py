"""Step 2 gate (~$0.02): confirm the qwen3_5 architecture loads and enumerate its linear
module names, so we lock the LoRA target_modules and confirm bitsandbytes 4-bit coverage
BEFORE any multi-hour GPU spend.

Builds the model on the meta device from config only — no ~56 GB weight download.

    modal run modal_app/arch_check.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import modal  # noqa: E402
from modal_app.common import BASE_MODEL, app, hf_secret, image  # noqa: E402


@app.function(image=image, secrets=[hf_secret], timeout=20 * 60)
def arch_check(base_model: str = BASE_MODEL):
    from collections import Counter

    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    info = {
        "architectures": getattr(cfg, "architectures", None),
        "model_type": getattr(cfg, "model_type", None),
        "image_token_id": getattr(cfg, "image_token_id", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
    }

    # Build structure only (meta device) to read module names without downloading weights.
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)

    lm_suffixes, vision_suffixes = Counter(), Counter()
    is_linear = lambda m: isinstance(m, nn.Linear) or m.__class__.__name__ in {
        "Linear", "Linear4bit", "Linear8bitLt", "NF4Linear"
    }
    for name, module in model.named_modules():
        if is_linear(module):
            suffix = name.split(".")[-1]
            if any(tag in name.lower() for tag in ("visual", "vision", "image_encoder", "vit")):
                vision_suffixes[suffix] += 1
            else:
                lm_suffixes[suffix] += 1

    info["lm_linear_suffixes"] = dict(lm_suffixes)
    info["vision_linear_suffixes"] = dict(vision_suffixes)
    # Recommended LoRA targets = LM projections that are not embeddings / lm_head.
    recommended = sorted(s for s in lm_suffixes if s not in {"lm_head", "embed_tokens"})
    info["recommended_target_modules"] = recommended
    print(__import__("json").dumps(info, indent=2))
    return info


@app.local_entrypoint()
def main(base_model: str = BASE_MODEL):
    import json

    print(json.dumps(arch_check.remote(base_model), indent=2))
