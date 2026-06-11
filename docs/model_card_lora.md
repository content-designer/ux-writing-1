---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - ux-writing
  - microcopy
  - content-design
  - copy-review
  - lora
  - qlora
---

# ux-writing-1-lora

The QLoRA adapter behind [**`gr33r/ux-writing-1`**](https://huggingface.co/gr33r/ux-writing-1)
— an open UX writing reviewer that takes a UI string + its code context and returns
compact JSON `{"rewrite", "reason", "risk"}`. **Blind-validated at 83%** human preference
over the fair-prompted base model (65/78 decisive comparisons on 90 held-out items —
methodology and eval code in the [repo](https://github.com/content-designer/ux-writing-1)).

Built for the Hugging Face **Build Small** hackathon — try it in the
[⛺ Copy Campfire arena](https://huggingface.co/spaces/build-small-hackathon/copy-campfire).

## Which artifact do you want?

| you want | use |
|---|---|
| Just run the model | [`gr33r/ux-writing-1`](https://huggingface.co/gr33r/ux-writing-1) (merged, vanilla transformers) |
| Run it on a laptop | [`gr33r/ux-writing-1-GGUF`](https://huggingface.co/gr33r/ux-writing-1-GGUF) (Q4_K_M 16.6 GB) |
| **This repo**: attach to base / continue training | the 159 MB LoRA (r=16 α=32, LM projections) |

## Usage (PEFT)

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = "Qwen/Qwen3.6-27B"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "gr33r/ux-writing-1-lora")
# Prompt contract + enable_thinking=False guidance: see the merged model card.
```

Training: QLoRA (4-bit NF4, double-quant, bf16 compute), LoRA on `q,k,v,o,gate,up,down`
projections, 2 epochs on ≈1,400 owner-authored/derived rewrite pairs, one A100-80GB.
To fine-tune further on **your** style guide (≈$2–6 on HF Jobs), see
[FINETUNE_GUIDE.md](https://github.com/content-designer/ux-writing-1/blob/main/docs/FINETUNE_GUIDE.md).

License: Apache-2.0. Attribution appreciated: *ux-writing-1 by gr33r*.
