# Fine-tune ux-writing-1 on your own style guide

`ux-writing-1` writes good general-purpose microcopy. Your product has a *voice* — and
the whole point of an open model is that you can teach it yours. This guide takes you
from a style guide to a tuned model **on Hugging Face infrastructure** (no GPUs of your
own), and — just as important — shows you how to *prove* your tune is better, using the
same blinded-review tooling this project was validated with.

The recipe in one line: **examples, not rules.** Models learn your voice from
before/after pairs far better than from prose guidelines.

## 1. Turn your style guide into training pairs

Collect **100–500 examples** of UI copy from your product as `(current → preferred)`
pairs. Three good sources:

1. **Real edits**: every time your content team rewrote a string in review, that's a pair.
2. **Style-guide rules, instantiated**: for each rule ("avoid 'please'", "sentence case
   buttons", "name the object"), write 5–10 concrete before/afters that demonstrate it.
3. **Strings your team would keep**: include ≈20% "already good" examples where the
   rewrite equals the input — this teaches restraint (our model returns copy verbatim
   when it's already right, and says so).

Format each pair as one JSONL row in the model's training contract — same system prompt,
same user shape, assistant replies with compact JSON:

```json
{"id": "yourco-0001",
 "messages": [
   {"role": "system", "content": "<the ux-writing-1 system prompt — copy it from uxft/policy.py>"},
   {"role": "user", "content": "Product surface: YourCo dashboard\nAudience: busy professional user\nUser state: saving billing changes\nContent type: button\nCurrent copy: OK\nCode/context:\n<Button>OK</Button>\nConstraints: Follow the YourCo voice: direct, warm, no exclamation marks. Preserve the intended product behavior."},
   {"role": "assistant", "content": "{\"rewrite\": \"Save billing details\", \"reason\": \"Names the action and object, in YourCo's direct voice.\", \"risk\": \"\"}"}
 ],
 "metadata": {"category": "button", "product_surface": "billing", "example_type": "styleguide_pair"},
 "provenance": {"source_policy_ids": ["yourco-style-guide"]}}
```

Two rules that matter:

- **Keep the system prompt identical** to ours (see `uxft/policy.py:SYSTEM_PROMPT`) and
  put your voice rules in the user turn's `Constraints:` line. That keeps your data
  in-distribution with the base tune — small datasets go much further this way.
- **Never let the same string appear in both train and eval.** Use the dedup splitter:

```bash
# validate every row, then make a leakage-free train/eval split
python -m uxft.schema yourco_pairs.jsonl
python - <<'EOF'
from pathlib import Path
from uxft.dataset import split_dedup, write_jsonl, iter_jsonl_rows
rows = iter_jsonl_rows(Path("yourco_pairs.jsonl"))
train, heldout = split_dedup(rows, eval_size=30, max_per_input=3)
write_jsonl(Path("yourco_train.jsonl"), train)
write_jsonl(Path("yourco_heldout.jsonl"), heldout)
EOF
```

Push both files to a **private** HF dataset (`huggingface-cli upload yourco/ux-writing-pairs ...`).

## 2. Train on Hugging Face Jobs (one command, no infra)

[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) runs the training on a
rented GPU and pushes the adapter to your Hub account (requires a Pro/Team plan). The repo
ships a ready-to-run script — [scripts/train_on_your_styleguide.py](../scripts/train_on_your_styleguide.py):

```bash
hf jobs uv run --detach \
  --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  --env DATASET_REPO=yourco/ux-writing-pairs \
  --env HUB_MODEL_ID=yourco/ux-writing-1-yourco \
  https://raw.githubusercontent.com/content-designer/ux-writing-1/main/scripts/train_on_your_styleguide.py
```

What it does (the exact recipe that produced ux-writing-1): QLoRA — 4-bit NF4 base,
LoRA r=16 α=32 on the LM projections — starting from **`gr33r/ux-writing-1`** so you
inherit the UX writing tune and add your voice on top. With 100–500 pairs expect
**≈20–60 min on one A100 (≈$2–6)**. Defaults that matter:

| knob | value | why |
|---|---|---|
| epochs | 3 (small data needs a bit more) | watch eval loss; stop if it climbs |
| learning rate | 1e-4 | gentler than from-scratch — you're nudging a tuned model |
| batch | 1 × grad-accum 16 | fits one A100-80GB |
| `max_length` | 2048 | matches the contract's typical size |

Alternatives: the full Modal pipeline is in [modal_app/train.py](../modal_app/train.py);
any single 80GB GPU (or 2×24GB with device_map) runs the same script locally.

## 3. Prove it's better (don't skip this)

Heuristics lie; blinded review doesn't. This repo ships the exact tooling used to
validate ux-writing-1 (83% blinded preference over base):

```bash
# 1) generate predictions from BOTH models on your held-out set
#    (serve each via any OpenAI-compatible endpoint, or adapt modal_app/eval_rewrite.py)
# 2) build a blinded A/B sheet — options anonymized + randomized per row:
python eval/make_ab_sheet.py \
  --a-preds before_preds.jsonl --a-label ux-writing-1 \
  --b-preds after_preds.jsonl  --b-label yourco-tune \
  --out reviews/yourco_ab
# 3) your content designer fills in `preferred` WITHOUT seeing the key,
#    then you join reviews/yourco_ab.key.csv on id to tally.
```

Ship the new model only if it wins decisively (we suggest ≥60% on ≥30 decisive items) and
loses nothing on safety-critical categories (destructive/payment/privacy/security).

## 4. Share it back (optional, appreciated)

If your tune isn't proprietary, publish the adapter with a model card noting the lineage
(`base_model: gr33r/ux-writing-1`) — and tell us at the
[Copy Campfire](https://huggingface.co/spaces/gr33r/copy-campfire) discussion tab.
Apache-2.0 all the way down.
