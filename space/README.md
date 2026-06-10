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
---

# ⛺ Copy Campfire — the UX writing arena

Two UX writers by the fire — one went to training camp. Paste your UI copy, get two
blind rewrites (base **Qwen3.6-27B** vs the **ux-writing-1** QLoRA fine-tune), vote,
then see the reveal. Votes are stored privately as preference data that trains the next version.

Built for the Hugging Face **Build Small** hackathon — small models, big adventure.

- Model: [`gr33r/ux-writing-1`](https://huggingface.co/gr33r/ux-writing-1)
- Code: [`content-designer/ux-writing-1`](https://github.com/content-designer/ux-writing-1)

**Space secrets required:** `BATTLE_URL`, `AUTH_TOKEN` (Modal backend), `HF_TOKEN`
(write access for vote storage); optional `VOTES_DATASET`.
