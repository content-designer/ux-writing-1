---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - ux-writing
  - microcopy
  - content-design
  - lora
  - qlora
---

# ux-writing-2.0-rewrite-qwen36 (1a — text rewrite)

A QLoRA adapter on `Qwen/Qwen3.6-27B` that reviews and rewrites UI copy found in product
code: given a UI string + code context, it returns compact JSON `{"rewrite","reason","risk"}`
that is purposeful, concise, conversational, clear, and accessible — preserving product
intent and never weakening safety-critical (destructive / payment / privacy / security) copy.

Built for the Hugging Face Build Small hackathon. The goal is a self-hostable reviewer that
scans UX writing across large codebases at a fraction of frontier-model cost.

## Results (held-out: 90 hand-authored gold, zero training overlap)

- **Blinded human A/B vs fair-prompted base: 1a preferred in 65/78 decisive items = 83%.**
  (Heuristic scores can't see this — they saturate near 1.0 for any competent model: base
  0.917 vs 1a 0.928. Human judgment is the real measure.)
- Wins across every category; strongest on inline errors, destructive confirmations,
  accessibility labels, buttons, system errors.

## Intended use & how to run

- Scan a repo for UI strings and suggest reviewed rewrites (human-in-the-loop; never
  auto-apply, especially to safety-critical copy).
- `Qwen3.6` is a reasoning model — for direct structured output set
  `apply_chat_template(..., enable_thinking=False)`. The adapter is trained to emit the JSON
  contract concisely (cheaper inference than the base model's default long reasoning).

## Training

- Method: QLoRA (4-bit NF4, double-quant, bf16), LoRA r=16 α=32 on the LM projections
  (`q,k,v,o,gate,up,down`); vision tower frozen.
- Data: ≈1,392 rewrite rows from [gr33r/ux-writing-sft](https://huggingface.co/datasets/gr33r/ux-writing-sft) (derived-only, no verbatim source text).
- 2 epochs on one A100-80GB (Modal). Base is a hybrid Gated-DeltaNet + full-attention model (≈27.8B, ≈248K vocab).

## Limitations

- The 5-metric heuristic eval is a weak proxy; trust the human A/B.
- Reasoning-model: without `enable_thinking=False` it may spend the token budget reasoning.
- For multimodal (screenshot) consistency review, use the combined adapter
  `gr33r/ux-writing-2.0-combined-qwen36` instead.

## License

Apache-2.0. Training data is owner-authored or permissively licensed; see the dataset card and repo `NOTICE`.
