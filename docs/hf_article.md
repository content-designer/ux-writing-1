# I fine-tuned a 27B model into a UX writer for $30 — then made it review all of PostHog for $3.22

*Draft for a Hugging Face community article (huggingface.co/blog/community). Image slots
are marked `[figure: …]` — screenshot the matching page from the
[campfire gallery](https://copy-campfire-gallery.vercel.app) at 1920×1080. All numbers
trace to artifacts in the [repo](https://github.com/content-designer/ux-writing-1).*

---

Every product ships copy debt: "Invalid", "OK", "An error occurred while processing your
request." It lives in code, at codebase scale — thousands of strings nobody owns. For the
[Build Small hackathon](https://huggingface.co/build-small-hackathon) I wanted to know:
can a small, open, self-hosted model do a senior UX writer's first review pass?

The short version: I fine-tuned [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
(Apache-2.0) with QLoRA on a hand-built dataset of ~1,400 rewrite pairs, for about **$30
of GPU time**. Then I blind-tested it against its own base model — options anonymized,
sides shuffled, 90 held-out items — and picked my fine-tune **83% of the time** without
knowing which was which. Then I pointed it at PostHog's entire frontend and reviewed
**10,000 UI strings in 77 minutes for $3.22**.

Everything is open: [the model](https://huggingface.co/gr33r/ux-writing-1),
[the LoRA adapter](https://huggingface.co/gr33r/ux-writing-1-lora),
[GGUFs for your laptop](https://huggingface.co/gr33r/ux-writing-1-GGUF), the scan/review
CLI, the eval tooling, and a guide to tuning it on *your* style guide.

## The recipe (and the part nobody tells you)

- **Base:** Qwen3.6-27B — a 27.8B hybrid-attention model (48 linear-attention + 16
  full-attention blocks) that fits on one A100, or quantized on a 24 GB laptop.
- **Data:** ~1,400 before/after pairs derived from a UX-writing course I authored, plus
  permissively-licensed open-source UI strings — each pair carrying the code context the
  string lives in, and a JSON contract: `{rewrite, reason, risk}`.
- **Training:** two QLoRA runs on Modal, ≈$30 total. LoRA on the standard
  q/k/v/o/gate/up/down set.
- **The gotcha that matters:** Qwen3.6 is a *thinking* model. With thinking left on, the
  base model produced **0% valid JSON** at 256 tokens on batch eval. Production runs use
  `enable_thinking=False` — direct mode. If your fine-tune target is structured output
  from a reasoning model, decide where thinking goes *before* you train.

The most important training decision was about *restraint*. A review model that rewrites
everything is useless — most shipped copy is fine. The dataset teaches "keep it unchanged
and say so" as a first-class answer. That shows up later in the numbers.

## Evaluating honestly (heuristics will lie to you)

My first eval was automated heuristics — length, clarity markers, terminology checks.
Both the base model and the fine-tune scored ~0.92. Saturated. Useless for telling them
apart, and exactly the kind of number that looks great in a README.

So the headline eval is the one I can defend: **blinded human preference**. 90 held-out
items, model outputs anonymized and shuffled, judged before unblinding. The fine-tune won
**65 of 78 decisive comparisons (83%)** — including 9–0 on error messages, 7–0 on
destructive-action copy, and 6–0 on accessibility labels. (The repo ships the blinded
review tooling, so when you tune this on your own style guide you can *prove* your
version is better rather than vibing it.)

One honesty rail I'll keep repeating: **I claim nothing about quality vs frontier
models.** The 83% is vs the model's own base. The frontier comparison below is about
cost, with the assumptions stated.

## What fine-tuning actually touched

Because the adapter is released, you don't have to trust me about what training did —
you can measure it. For every LoRA pair (A, B) I computed the Frobenius norm of the
delta ‖B·A‖ per (layer, module) — 256 deltas across 64 layers. (Cheap trick:
‖BA‖²_F = Σ((BᵀB)∘(AAᵀ)), so you never materialize the full delta matrix.)

[figure: weights_heatmap — the LoRA fingerprint heatmap]

The fingerprint is legible:

- **MLP modules adapted on all 64 layers; attention only exists on the 16 full-attention
  blocks** — the hybrid architecture, visible in one picture.
- **The biggest changes concentrate late in the stack**: `gate_proj` on layers 56–63
  burns brightest. Style lives near the surface; the base model's knowledge underneath
  stays essentially frozen.

There's an interactive version where hovering any cell explains in plain language what
that weight does and how much it moved: **https://copy-campfire-gallery.vercel.app/weights_heatmap.html** — reproduce
it from the released adapter with one script
([`scripts/adapter_heatmap.py`](https://github.com/content-designer/ux-writing-1/blob/main/scripts/adapter_heatmap.py)).

## PostHog at codebase scale

A demo on 20 cherry-picked strings proves nothing. So: [PostHog](https://github.com/PostHog/posthog)
(MIT), pinned commit, full `frontend/` + `products/` scan — **152,713 raw strings**,
26,061 after UI-copy filters (tests/stories/identifiers/Tailwind classes out), a seeded
random **10,000 reviewed**, end to end on one rented A100-80GB.

Measured, not estimated:

| | |
|---|---|
| Wall-clock | **77.2 minutes** (incl. model load) |
| Cost | **$3.22** — $0.32 per 1,000 strings (A100 @ $2.50/h list) |
| Tokens | 3,590,383 prompt + 313,293 completion |
| JSON contract | **9,999 / 10,000 valid (99.99%)** |
| Verdicts | **994 changed · 9,006 kept as-is** |

That last row is the restraint showing up at scale: the model left **90% of PostHog's
strings alone** — and said why, per string. The suggestions it did make look like a
colleague's review comments, file and line included:

[figure: before_after — four real rewrite cards]

- `Invalid` → `Invalid API key` — *names the exact problem*
- `Done` → `Save changes` — *uses the primary verb for the action*
- `must be string` → `Enter a single line of text` — *plain language for a form constraint*
- `Lucky you!` → `You're on the YC plan` — *clarity over cuteness*

(Interactive: **https://copy-campfire-gallery.vercel.app/before_after.html** — every card is an unedited row from the
run artifact; suggestions are review output for humans to accept or reject, never
auto-applied.)

## The bill, honestly

Here is the cost framing, with every assumption on the table. Take the *measured* token
counts from the run above and price the identical workload at public list prices
(pulled 2026-06-12):

| same workload | bill | per 1K strings | vs measured |
|---|---|---|---|
| **ux-writing-1, one rented A100 (measured)** | **$3.22** | $0.32 | — |
| Qwen3.6-27B via DeepInfra (estimate) | $2.15 | $0.21 | 0.7× |
| Claude Opus 4.8 (estimate) | $25.78 | $2.58 | **≈8×** |
| GPT-5.5 (estimate) | $27.35 | $2.73 | **≈8.5×** |

[figure: cost_compare — the bill ladder]

Caveats, because they're the point: the frontier rows are **list-price estimates on my
measured token counts**, not measured frontier runs; tokenizers differ (±~15%);
reasoning-mode APIs bill hidden thinking tokens as output, which would *raise* their
bills; and again — **no quality claim vs frontier models**. Notice the DeepInfra row,
too: renting the *base* model via API is even cheaper than my GPU. The economics
argument for small open models isn't "my GPU is magic" — it's that the workload prices
like a commodity, you can run it on hardware you control, your unshipped product copy
never leaves your infrastructure, and the model can learn *your* style guide for about
$5 of training.

## The campfire

The launch demo is ⛺ [**Copy Campfire**](https://huggingface.co/spaces/build-small-hackathon/copy-campfire) —
paste your worst error message, two anonymous campers rewrite it, you vote, then the
reveal. Votes are blinded (length fingerprints the base model, so reasons and metadata
hide until after you choose) and every vote becomes preference data for v2. The arena is
literally the DPO data flywheel.

## Take it home

- **Model / adapter / GGUF:** [gr33r/ux-writing-1](https://huggingface.co/gr33r/ux-writing-1) ·
  [gr33r/ux-writing-1-lora](https://huggingface.co/gr33r/ux-writing-1-lora) ·
  [gr33r/ux-writing-1-GGUF](https://huggingface.co/gr33r/ux-writing-1-GGUF) (Q4_K_M runs
  on a 24 GB laptop in LM Studio/Ollama)
- **Scan your own repo:** `python -m uxft.scan` + `python -m uxft.review_repo` —
  [repo](https://github.com/content-designer/ux-writing-1)
- **Tune it on your style guide:** ~100 before/after pairs, one job, ≈$5 —
  [FINETUNE_GUIDE](https://github.com/content-designer/ux-writing-1/blob/main/docs/FINETUNE_GUIDE.md),
  blinded-review tooling included so you can prove it worked.

Small model, hand-built dataset, honest numbers. Come vote at the campfire — bring your
worst error message.
