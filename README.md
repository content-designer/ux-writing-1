# UX-Writing Fine-Tune v2 — Qwen3.6-27B (QLoRA on Modal)

A fine-tuned, open-weights UX-writing reviewer that flags and rewrites interface copy across
**codebases** (text) and **screenshots** (vision), at a fraction of the cost of frontier models.
Built for the Hugging Face [Build Small hackathon](https://huggingface.co/build-small-hackathon).

- **Base model:** [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) — dense ~27.8B, vision-capable, Apache-2.0.
- **Method:** QLoRA (4-bit NF4) via TRL `SFTTrainer` + PEFT, trained on **Modal** GPUs.
- **Storage:** datasets + adapters on the Hugging Face Hub (`gr33r/*`). **Monitoring:** Trackio.

## Why two training runs (not one combined dataset)

We have two distinct tasks, not one corpus in two modalities:

| Task | Modality | Input → Output | Data |
|---|---|---|---|
| **rewrite** | text | UI string + code context → `{rewrite, reason, risk}` | ~1,490 rows |
| **consistency** | vision | screenshot → `{inventory, issues}` | 156 synthetic screens |

Merging both into a single first run would change the base model (4B→27B), the scale, **and** the
task mix at once — so you couldn't tell what helped. Instead we **consolidate the data** into one
versioned, zero-leakage dataset and run:

1. **Run 1a — text baseline:** rewrite task only. Establishes the 27B lift over the old 3B and over prompt-only.
2. **Run 1b — combined:** rewrite + synthetic consistency, multi-task. Compared head-to-head with 1a so we can prove the multimodal mix helps and didn't regress rewrite.

**Real screenshots are eval-only** (zero leakage): we train consistency on synthetic screens and
test on real Cal.com / Ghost / Wealthsimple captures, so the real-world score measures generalization.

## Layout

```
configs/        training.json (base model + run profiles), sources.json (provenance registry)
uxft/           core library: schema, dataset helpers, eval heuristics, OpenAI-compatible client, serving guards
modal_app/      Modal apps: common (image/volume/secret), train (1a/1b), eval_rewrite, eval_consistency, merge_and_push, arch_check
data_build/     build the unified gr33r/ux-writing-sft dataset; annotate the new eval screenshots
eval/           held-out scoring: rewrite benchmark + real-screenshot consistency, leakage gates
docs/           runbook, dataset card, model card template
tests/          schema / postfilter / leakage tests
```

Data blobs (JSONL, screenshots, checkpoints) live on the Hub, **not in git** — see `.gitignore`.

## Run it (staged; a $1 smoke test gates all real spend)

```bash
pip install -e ".[dev]"
python3 -m modal setup                              # browser auth (one time)
modal secret create hf-token HF_TOKEN=$(cat ~/.cache/huggingface/token)

modal run modal_app/arch_check.py                   # 2. confirm qwen3_5 module names for LoRA targets
python data_build/build_unified_dataset.py          # 4. build + push gr33r/ux-writing-sft
modal run modal_app/common.py::download_weights     # 6. cache base weights to a Volume
modal run modal_app/train.py --run-mode combined --max-steps 8 --no-push   # 7. SMOKE TEST (~$1)

modal run modal_app/train.py --run-mode text        # 8.  Run 1a
modal run modal_app/train.py --run-mode combined    # 10. Run 1b
modal run modal_app/eval_rewrite.py                 # rewrite benchmark (1a + 1b vs base)
modal run modal_app/eval_consistency.py             # real-screenshot consistency (1b vs base)
```

Estimated total compute: **~$9–15** of the $250 hackathon credit (A100-80GB @ $2.50/h).

## Fine-tune it on your style guide

The model learns your product's voice from before/after pairs — ~100–500 examples and
**one HF Jobs command** (~$2–6 on an A100), starting from `gr33r/ux-writing-1` so you
keep the UX-writing tune and add your voice on top. The repo also ships the blinded
A/B review tooling to *prove* your tune is better before you ship it.

→ **[docs/FINETUNE_GUIDE.md](docs/FINETUNE_GUIDE.md)** · ready-to-run script: [scripts/train_on_your_styleguide.py](scripts/train_on_your_styleguide.py)

## Results (held-out, 90 hand-authored gold)

Both adapters trained (QLoRA on Modal). Full detail in [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md); cards in [docs/model_card_1a.md](docs/model_card_1a.md) / [docs/model_card_1b.md](docs/model_card_1b.md).

- **1a (text) vs fair-prompted base — blinded human A/B: 1a preferred 65/78 decisive = 83%.** (Heuristics saturate and can't see it: base 0.917 vs 1a 0.928 — human review is the real measure.)
- **1b (combined) vs 1a — blinded human A/B: 30 / 21 / 39 tie.** Combining text + vision held text quality ~at par (within noise) while adding screenshot-consistency review → consolidation hypothesis validated.
- **Deploy:** bulk code/text scanning → 1a (slightly better, cheaper); screenshot review / one unified model → 1b.

Reasoning-model note: `Qwen3.6` thinks by default — use `enable_thinking=False` for direct JSON (the adapters bake in concise structured output → cheaper inference).

## Release artifacts

| artifact | where |
|---|---|
| Merged model (flagship) | [`gr33r/ux-writing-1`](https://huggingface.co/gr33r/ux-writing-1) |
| LoRA adapter | [`gr33r/ux-writing-1-lora`](https://huggingface.co/gr33r/ux-writing-1-lora) |
| Quantized (llama.cpp / LM Studio / Ollama) | [`gr33r/ux-writing-1-GGUF`](https://huggingface.co/gr33r/ux-writing-1-GGUF) |
| ⛺ Copy Campfire arena | [`gr33r/copy-campfire`](https://huggingface.co/spaces/gr33r/copy-campfire) |

Measured economics: **~7,950 strings/hour, ~$0.31 per 1K strings** on one A100, 100%
valid JSON ([docs/COST_NOTES.md](docs/COST_NOTES.md)). Real-codebase demo:
[docs/demo/](docs/demo/) (Cal.com — 13/200 changes suggested, the rest kept as-is).

## Roadmap

- **v2 (DPO)** from Copy Campfire votes + the seeded blinded-review pairs.
- **Vision**: the combined text+vision adapter exists (unreleased) — real-screenshot
  consistency eval needs the 25 reference screenshots hand-labeled first.
- Benchmark v2.1 scenario fixes from reviewer notes (docs/EVAL_RESULTS.md §4).

## License

Apache-2.0 (see `LICENSE`). Third-party attributions in `NOTICE`. Training data is owner-authored
or permissively licensed (MIT/Apache); no proprietary or course-licensed content is included.
