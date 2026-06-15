---
title: Copy Campfire
emoji: ⛺
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.49.1"
app_file: app.py
pinned: false
license: apache-2.0
short_description: UX writing arena — base model vs fine-tune. You judge.
tags:
  - backyard-ai
  - best-demo
  - tiny-titan
  - judges-wildcard
  - modal
---

# ⛺ Copy Campfire — the UX writing arena

Two UX writers by the fire — one went to training camp. Paste your UI copy, get two
blind rewrites (base **Qwen3.6-27B** vs the **ux-writing-1** QLoRA fine-tune), vote,
then see the reveal. Votes are stored privately as preference data that trains the next
version — the arena is the data flywheel.

## The idea

Every product ships copy debt — "Invalid", "OK", "An error occurred while processing
your request" — thousands of interface strings nobody owns. **ux-writing-1** is a small,
open model fine-tuned to review that copy the way a senior UX writer would: rewrite
what's weak, **keep what's already right**, and say why. It runs at codebase scale for
cents, on hardware you control — exactly the kind of bulk, structured LLM work that
shouldn't be metered at frontier prices.

## How it was built

- **Base:** [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) — Apache-2.0, 27.8B
  params (under the 32B limit).
- **Training:** QLoRA on ~1,400 hand-built before/after pairs, each with a
  `{rewrite, reason, risk}` contract and the code context the string lives in. Two runs,
  ≈$30 of GPU on **Modal**.
- **Serving:** Modal OpenAI-compatible endpoint; this Gradio arena calls it live.
- **Honest eval:** 83% blinded human preference vs the base model (90 held-out items) —
  not automated heuristics, which saturated.
- **At codebase scale:** reviewed 10,000 of PostHog's UI strings in 77 min for **$3.22**
  — changed 994, **kept 9,006** as-is. Restraint is trained, not hoped for.

## Tech

Qwen3.6-27B · QLoRA (PEFT / TRL) · Modal (training + serving) · Gradio (this arena) ·
llama.cpp / GGUF (zero-cost laptop inference) · Hugging Face Hub (model, adapter,
dataset, blinded vote store).

## Take it home

- **Model:** [`gr33r/ux-writing-1`](https://huggingface.co/gr33r/ux-writing-1) ·
  [adapter](https://huggingface.co/gr33r/ux-writing-1-lora) ·
  [GGUF](https://huggingface.co/gr33r/ux-writing-1-GGUF)
- **Code & eval tooling:** [`content-designer/ux-writing-1`](https://github.com/content-designer/ux-writing-1)
- **Interactive figures:** [the campfire gallery](https://copy-campfire-gallery.vercel.app)

## Submission

- 🎬 **Demo video:** _add the public video link here before submitting_
- 📣 **Social post:** _add the social post link here before submitting_

---

**Space secrets required:** `BATTLE_URL`, `AUTH_TOKEN` (Modal backend), `HF_TOKEN`
(write access for vote storage); optional `VOTES_DATASET`.
