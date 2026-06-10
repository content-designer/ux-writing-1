---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - ux-writing
  - microcopy
  - content-design
  - multimodal
  - lora
  - qlora
---

# ux-writing-2.0-combined-qwen36 (1b — text + vision)

A QLoRA adapter on `Qwen/Qwen3.6-27B` trained multi-task on **both**:
- **rewrite** (text): UI string + code context → `{"rewrite","reason","risk"}`, and
- **consistency** (vision): a screenshot → `{"inventory":[...],"issues":[{type,strings,problem,fix}]}`
  — a screen-level review of relationships between strings (duplication, terminology/brand
  drift, casing, number/plural agreement, tone clashes).

This is the unified multimodal model from the consolidation hypothesis: one adapter that can
both scan code for copy issues and review rendered screens.

## Results

- **Text rewrite (held-out 90, blinded human A/B vs 1a text-specialist): 1a 30 · 1b 21 · tie 39.**
  Combining vision held text quality essentially at par with the specialist (43% ties; a
  small, within-noise edge to 1a). Heuristic overall 0.929 (≈ 1a 0.928).
- **Vision consistency:** real-screenshot eval pending (requires hand-labeling the held-out
  reference screenshots); trained on synthetic screens only (zero real-image leakage).

## Intended use

- A single model for UX-writing review across code **and** screenshots. For bulk text-only
  scanning where cost is paramount, the text specialist `gr33r/ux-writing-2.0-rewrite-qwen36`
  is marginally better and cheaper; use 1b when you also need screen-level consistency review.
- Reasoning model — set `enable_thinking=False` for direct structured output.

## Training

- QLoRA (4-bit NF4, bf16), LoRA r=16 α=32 on LM projections; vision tower frozen.
- Mixed-modality SFT: ~1,392 rewrite (text) + 156 consistency (synthetic screenshots) rows
  from [gr33r/ux-writing-sft](https://huggingface.co/datasets/gr33r/ux-writing-sft);
  custom collator handles text-only and image+text rows in one run (per-device batch 1).
- 2 epochs on one A100-80GB (Modal). In-training eval disabled (full-logits eval over the
  ~248K vocab on long vision rows OOMs 80GB); evaluation is done separately on held-out sets.

## Limitations

- Real-world vision consistency recall is the known weak spot from prior iterations and is
  not yet measured for this model — treat the vision pass as a preview / human-in-the-loop aid.
- Text rewrite is a hair behind the text specialist (within noise).

## License

Apache-2.0. Owner-authored / permissively-licensed data; real screenshots are eval-only. See dataset card and `NOTICE`.
