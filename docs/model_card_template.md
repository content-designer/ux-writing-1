---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
pipeline_tag: image-text-to-text
language:
- en
tags:
- ux-writing
- vision
- consistency-review
- microcopy
- lora
---

# ux-writing-1.4-vision — screen-level UX consistency reviewer (release candidate · preview)

A vision fine-tune of **Qwen3-VL-4B-Instruct** that reviews **one app screenshot** and reports
**cross-element consistency problems** in its copy — issues that exist only in the *relationships
between strings* on a screen, which per-string review is structurally blind to.

This is **Pass 2** of the planned two-pass v1.4 (the per-string voice-rewrite Pass 1 is future
work). It is a **human-in-the-loop review aid**, not an autonomous gate.

- **Adapter:** `gr33r/ux-writing-1.4-consistency-preview-v2` (LoRA)
- **Merged standalone:** `gr33r/ux-writing-1.4-vision-rc-merged` (serve without PEFT)
- **Base:** `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0, vision — chosen for commercial use)

## What it detects

Cross-element issues, returned as JSON: **duplication** (e.g. an empty-state body that just
repeats the page subtitle), **terminology/brand inconsistency** (`Team` vs `Teams`, `Github` vs
`GitHub`), **number/plural agreement** (`1 members`), **tone clash** (a joke on a destructive
action), and **casing inconsistency** across parallel elements.

It does **not** rewrite individual strings (that's the text model `gr33r/ux-writing-1-preview`).

## Serving recipe — all three are LOAD-BEARING

The model only behaves correctly when served exactly this way. Each was established empirically
(see `consistency_realbug_results.md`); the canonical implementation is `scripts/serve_consistency.py`.

1. **Issues-only prompt** (below). The adapter was trained on this format and **collapses to
   silence** on inventory-first ("transcribe every string, then judge") prompts.
2. **`MAX_LONG_SIDE = 1792`.** Dense real screenshots silently lose small-text detection below this.
3. **`consistency_postfilter`** — a deterministic guard that drops "identical strings flagged as
   inconsistent" hallucinations (a terminology/brand/casing issue requires the strings to differ).

System prompt:

```
You are a senior UX writer doing a SCREEN-LEVEL CONSISTENCY review of one screen.
Your job is NOT to restyle individual strings. Find problems that exist only in the RELATIONSHIPS between strings on the screen:
1. Duplication — two strings that say the same thing, where one should add new value (e.g., an empty-state body that just repeats the page subtitle).
2. Terminology/brand inconsistency — the same concept named differently (e.g., 'Team' vs 'Teams'), or a brand spelled wrong/inconsistently (e.g., 'Github' vs 'GitHub').
3. Number/plural agreement (e.g., '1 members').
4. Tone clash — one string whose register fights the rest of the screen (e.g., a joke on a destructive/irreversible action).
5. Casing inconsistency across parallel elements (e.g., 'Save changes' vs 'Save Changes').
Return compact JSON: {"issues": [{"type": str, "strings": [str], "problem": str, "fix": str}]}. Empty issues list if none.
```

User turn: the screenshot + `Surface: <label>\nDo a screen-level consistency review.`

## Evaluation (honest, real screens)

Held-out, leakage-verified **real-bug eval** — 18 real screenshots (3 with genuine within-screen
bugs + 15 well-written restraint screens), served per the recipe above. Adapter vs **prompt-only
base** (same model, adapter off):

| metric | base | **this model** |
|---|---|---|
| recall (natural bugs) | 0.25 | **0.25** |
| precision / F1 | 0.05 / 0.08 | **0.33 / 0.29** |
| over-flags on 15 clean real screens | **16** | **1** |

The headline is **restraint**: at equal recall the model is **≈16× quieter** than base on
well-written screens (base "matches" recall only by spraying 6 issue types onto clean screens).
It caught a real empty-state duplication exactly.

> Honesty notes (this project holds a strict leakage-free / blinded bar): the **real-screen**
> numbers above are the trustworthy ones. In-distribution synthetic `eval_loss` (≈0.03) is **not**
> a quality proxy. The real-bug set is still **small (n=3 bug screens)**, so recall is
> underpowered — each gold instance moves it 25%.

## Limitations

- **Low recall (0.25)** on the current real-bug set; misses subtler terminology/casing drift.
- **Multi-issue collapse** — on a screen with two problems it tends to report at most one.
- **Eval coverage is thin** — duplication/terminology/casing are exercised; plural/brand/tone are
  not yet in the real-bug set (capture pending).
- **Prompt-sensitive** — the issues-only prompt is mandatory (see recipe); wrong prompt → silence.

## Intended use & status

**Release candidate, preview grade.** Ship as a **human-in-the-loop review aid**: a UX writer
triages its flags. It is **not** validated for strict autonomous production. Use it to surface
*candidate* cross-screen inconsistencies for human confirmation.

## Training data & provenance

Diversified synthetic consistency screens (HTML→headless-Chrome render, exact planted-bug gold,
≈25% clean negatives) + 9 real Cal.com screens (MIT) as real-layout negatives. Held-out eval =
real Cal.com + Ghost screens (MIT, business email redacted). Base is Apache-2.0; all training/eval
sources are permissively licensed — **commercial-use clean**. ~minutes of A10G/L40S compute.
