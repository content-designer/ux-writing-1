---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - ux-writing
  - microcopy
  - content-design
  - copy-review
  - qwen3_5
---

# ux-writing-1

**An open UX writing reviewer you can run on your own infrastructure.** Give it a UI
string and its code context; it returns compact JSON — `{"rewrite", "reason", "risk"}` —
that is purposeful, concise, conversational, clear, and accessible. It preserves product
intent, keeps `{{ variables }}` and locale terms intact, and is trained not to weaken
safety-critical (destructive / payment / privacy / security) copy.

Built to scan UX writing across **massive codebases** at a fraction of frontier-API cost
— private, unlimited-reuse, and yours to fine-tune further. Created for the Hugging Face
**Build Small** hackathon (*small models, big adventure*) on ≈$40 of compute.

- **Base:** [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) (Apache-2.0, ≈27.8B dense, hybrid linear+full attention, reasoning model)
- **Method:** QLoRA (4-bit NF4) · LoRA r=16 α=32 on LM projections · 2 epochs · one A100-80GB
- **Adapter:** [`gr33r/ux-writing-1-lora`](https://huggingface.co/gr33r/ux-writing-1-lora) · **Quantized:** [`gr33r/ux-writing-1-GGUF`](https://huggingface.co/gr33r/ux-writing-1-GGUF)
- **Try it / judge it:** [⛺ Copy Campfire arena](https://huggingface.co/spaces/build-small-hackathon/copy-campfire) · **Code:** [content-designer/ux-writing-1](https://github.com/content-designer/ux-writing-1)

## Does the fine-tune actually beat the base model?

Measured the honest way — **blinded human review**, not just automatic metrics. On 90
held-out, hand-authored benchmark items (zero training overlap), an expert content
designer judged anonymized base-vs-fine-tune pairs (sides randomized, revealed after
judging):

| | result |
|---|---|
| **Fine-tune preferred** | **65 / 78 decisive comparisons = 83%** |
| Base preferred | 13 / 78 = 17% |
| No preference | 12 (11 identical outputs, 1 both-bad) |

The fine-tune won every category; strongest on inline errors (9–0), destructive
confirmations (7–0), accessibility labels (6–0), buttons (10–1), system errors (11–3).

**Caveat we want you to know:** on crude automatic heuristics the gap is small (0.928 vs
0.917 — they saturate for any competent model). The 83% is what expert human judgment
sees that the heuristics can't. Both evals, the gold set, and the scoring code are in the
repo — reproduce them.

The fine-tune is also **leaner at inference**: it answers the contract directly with
brief reasoning, where the base model tends to reason at length (in our spot checks,
≈2× fewer output tokens for the same quality answer — that's your serving bill).

## Usage

`Qwen3.6` is a reasoning model. For fast structured output, disable thinking:

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
import torch

repo = "gr33r/ux-writing-1"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

SYSTEM = """You are a senior UX writer reviewing interface copy in product code.
Rewrite the UI copy so it is purposeful, concise, conversational, clear, and accessible.
If the current copy is already clear, accurate, and on-brand, keep it unchanged: return it verbatim as the rewrite and say so in the reason.
Preserve product intent. Do not invent actions, facts, or product behavior that are not in the context.
Keep locale-specific terms (for example, "Postal code" for Canadian addresses) and any {{ variables }} exactly as written.
Never weaken safety-critical copy: destructive, payment, privacy, and security messages must keep their consequence and must not be softened.
Return compact JSON with: rewrite, reason, and risk. Use an empty string for risk when none applies."""

user = """Product surface: existing codebase
Audience: product user
User state: using the screen that contains src/Checkout.tsx:120
Content type: inline_error
Current copy: Invalid
Code/context:
<TextField label="Email" error="Invalid" />
Constraints: Suggest a UX writing rewrite only if the context supports it. Preserve the intended product behavior."""

enc = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    add_generation_prompt=True, return_dict=True, return_tensors="pt",
    enable_thinking=False,   # direct mode: fast JSON. True = reasoning first, then JSON.
).to(model.device)
out = model.generate(**enc, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
# {"rewrite": "Enter a valid email address", "reason": "...", "risk": ""}
```

**Scan a whole codebase** (CLI from the repo): extract UI strings from JS/TSX/Vue/Svelte/
HTML/i18n files and review them against any OpenAI-compatible endpoint serving this model:

```bash
python -m uxft.scan /path/to/your/repo --limit 500 --out candidates.jsonl
python -m uxft.review_repo /path/to/your/repo \
  --endpoint https://your-host/v1/chat/completions --api-key $TOKEN --out review.jsonl
```

The output is diff-friendly JSONL for **human review** — never auto-apply, especially to
destructive/payment/privacy/security copy.

## Fine-tune it further

Teams should tune this to their own voice — ≈100–500 before/after pairs from your style
guide and **one HF Jobs command** (≈$2–6 on an A100), starting from this model. Full
walkthrough, dataset recipe, ready-to-run script, and the blinded A/B tooling to verify
your tune actually wins:
[**FINETUNE_GUIDE.md**](https://github.com/content-designer/ux-writing-1/blob/main/docs/FINETUNE_GUIDE.md).
A preference round (DPO) using arena-style votes is the v2 roadmap; vote at the
[Copy Campfire](https://huggingface.co/spaces/build-small-hackathon/copy-campfire) to contribute.

## Training data

≈1,400 owner-authored / derived synthetic rewrite pairs (no verbatim text from any style
guide), plus permissively-licensed (MIT) real microcopy. Validation/test splits share no
UI string with training (`input_key` dedup). The dataset itself is private; the schema,
builders, and validators are open in the repo.

## Limitations

- English-centric; trained on product-UI microcopy (buttons, errors, empty states,
  notifications, onboarding, confirmations, labels) — long-form content is out of scope.
- A reviewer's assistant, not an authority: keep a human in the loop; treat `risk` notes
  as flags, not verdicts.
- On vague inputs without context it can over-specify (invent plausible-but-unverified
  specifics) — give it the code context.
- The base model is vision-capable but this fine-tune was trained text-only; screenshot
  review is a separate (unreleased) experiment.

## License

Apache-2.0, same as the base model. Attribution appreciated: *ux-writing-1 by gr33r*.
