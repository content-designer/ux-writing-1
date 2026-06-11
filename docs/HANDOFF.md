# HANDOFF — ux-writing-1 release (updated 2026-06-10, post-launch)

State of the project for the next working session. **LAUNCHED**: everything public,
submitted-in-place for the hackathon. Remaining human steps: demo video + social post,
dropped via the **Gradio Discord** by **2026-06-15**.

## One-paragraph status

`ux-writing-1` (QLoRA fine-tune of Qwen3.6-27B for UX writing review) is trained,
**blind-validated at 83%** preference over fair-prompted base (65/78 decisive, 90
held-out items, owner-judged), merged, quantized to GGUF, **public**, and live behind the
⛺ **Copy Campfire** arena — hosted at **`build-small-hackathon/copy-campfire`** (org
hosting is the eligibility requirement), fully restyled with the owner's Claude Design
system, with anti-bias voting feeding a private preference dataset for a future DPO
round. Measured economics: **$0.31/1K strings batched** on one A100 (100% valid JSON).

## Artifacts

| artifact | where | visibility |
|---|---|---|
| Merged flagship (54.7 GB, verified ≡ adapter) | `gr33r/ux-writing-1` | **public** (launched 2026-06-10) |
| LoRA adapter (the validated 1a) | `gr33r/ux-writing-1-lora` | **public** (launched 2026-06-10) |
| GGUF (Q4_K_M 16.6 GB, Q8_0 28.6 GB) | `gr33r/ux-writing-1-GGUF` | **public** (launched 2026-06-10) |
| ⛺ Copy Campfire arena (SUBMISSION) | `build-small-hackathon/copy-campfire` (Space) | **public** (launched 2026-06-10) |
| Campfire personal copy (kept in sync) | `gr33r/copy-campfire` (Space) | private (avoid split votes) |
| Adapter mirror (org Models-tab visibility) | `build-small-hackathon/ux-writing-1-lora` | **public** (canonical = gr33r) |
| Code (pipeline, CLI, evals, Space src) | github.com/content-designer/ux-writing-1 | **public** |
| Training dataset (unified, zero-leakage) | `gr33r/ux-writing-sft` | private, stays |
| Votes / DPO corpus (168 seeded + live) | `gr33r/ux-writing-arena-votes` | private, stays |
| 1b combined text+vision adapter (unreleased) | `gr33r/ux-writing-2.0-combined-qwen36` | private, stays |
| Archived 4B-era adapter | `gr33r/ux-writing-preview-4b` | private |

## Validated results (sources: docs/EVAL_RESULTS.md, docs/COST_NOTES.md)

- **Blinded human A/B vs fair base: 83%** (65/78 decisive; wins every category).
- 1b (combined +vision) ≈ 1a on text (30/21/39 ties) — multi-task didn't regress text.
- Heuristics saturate (0.928 vs 0.917) — human review is the measure; say so everywhere.
- **Throughput (measured): 7,951 strings/h, $0.31/1K, 100% valid JSON** (A100, batch 16).
- Cal.com demo: 400 strings reviewed, 365 kept (restraint), 0 errors → docs/demo/.
- Live tell from serving logs: lantern-mode base = 1,020 thinking tokens/145 s vs
  fine-tune 79 tokens/15 s for the same brief.

## The public flip (EXECUTED 2026-06-10 — kept for reference)

The submission Space lives in the hackathon org (`build-small-hackathon/copy-campfire`,
required for eligibility); the owner's personal copy (`gr33r/copy-campfire`) is private
to avoid splitting traffic/votes. The org Space's HF_TOKEN is a fine-grained token
("campfire-votes-only") scoped to the votes dataset + org repos — **post-hackathon
cleanup: remove its org permission** at settings/tokens.

```python
from huggingface_hub import HfApi
api = HfApi()
for repo in ["gr33r/ux-writing-1", "gr33r/ux-writing-1-lora", "gr33r/ux-writing-1-GGUF"]:
    api.update_repo_settings(repo, private=False)
api.update_repo_settings("build-small-hackathon/copy-campfire", private=False, repo_type="space")
```

For launch/video windows, keep the GPU warm (skips the ≈3-min cold start):
edit `modal_app/serve_openai.py` `@app.cls(... min_containers=1)` → `modal deploy` →
revert after ($2.50/h while pinned).

## Infrastructure reference

- **Modal** workspace `content-designer` (~$150 of $250 credits left).
  - Deployed app `uxw1-serve`: `chat` (OpenAI-compatible, adapter on) + `battle`
    (both arms; `arm` param for progressive render) endpoints. Auth = `_auth` field in
    the JSON payload, checked against secret `ux-arena-auth` (`AUTH_TOKEN`).
  - Token also at `~/.uxw1_arena_token` (local) and in the Space's secrets.
  - Volume `qwen36-27b-weights` caches base weights (don't delete — saves 10 min/run).
  - Other apps: train (`modal_app/train.py`, spawn+detach), `uxw1-gguf`, `uxw1-bench`.
- **Space secrets** (both copies): `BATTLE_URL`, `AUTH_TOKEN`, `HF_TOKEN`, `VOTES_DATASET`.
  The ORG copy's `HF_TOKEN` = fine-grained token "campfire-votes-only" (votes dataset +
  org repos write; local copy at `~/.campfire_votes_token`). **Post-hackathon cleanup:
  remove its org permission at settings/tokens.**
- **HF auth**: the CLI's cached token ("Claude Code", fine-grained, user-scope only —
  it CANNOT write to the hackathon org; use `HfApi(token=<campfire_votes_token>)` for
  any update/restart of `build-small-hackathon/copy-campfire`). Modal secret `hf-token`.
- **Design system**: the Claude Design handoff bundle lives at
  `~/Downloads/Copy Campfire Design System-handoff.zip` (tokens, components, UI kit) —
  the source of truth for any future Campfire UI work; implemented in `space/app.py`.
- **GitHub**: `content-designer/ux-writing-1`, branch `main`.

## Hard-won gotchas (read before touching anything)

1. **Qwen3.6 is a thinking model**: `enable_thinking=False` for direct JSON. Base emits
   0% parseable JSON at 256 tokens otherwise, and ≈1K thinking tokens when allowed.
2. **GGUF conversion**: transformers drops the MTP draft layer but config still declares
   it → patch `mtp_num_hidden_layers=0` pre-conversion or llama.cpp wants a phantom
   `blk.64` (handled in `modal_app/convert_gguf.py`).
3. **Modal**: use `.spawn()` + `modal run --detach` (a `.remote()` dies with the local
   handle); app files must be **self-contained** (cross-file imports →
   ModuleNotFoundError in the container); `pip_install_from_requirements` reads *local*
   paths; `ephemeral_disk` min is 512 GiB; one serving container only (a second means a
   second 56 GB cold load).
4. **transformers 5.x**: `push_to_hub` dropped `safe_serialization` → save_pretrained +
   `HfApi.upload_folder`. In-training eval OOMs on vision rows (248K-vocab logits).
5. **Gradio Space**: generators must heartbeat-yield every ~2 s or the SSE stream dies as
   a bare "Error"; width snapping fixed via `fill_width=True` + clamping every wrapper;
   Gradio theme tokens re-pointed at the design system in CSS.
6. **Arena integrity**: pre-vote cards hide reason/chips/thinking (length fingerprints
   the base model); forensics render post-vote. Keep it that way or votes degrade.
7. **Unfinished thinking**: in lantern mode the model may never emit `</think>` — the
   server treats that as thinking-only (empty answer → budget-exhausted card), and the
   card extracts the LAST parseable flat `{…}` with a `rewrite` (a greedy first-to-last
   brace regex renders leaked reasoning as the answer). Same scan in `review_repo`.
8. **Two Space copies**: ship every `space/app.py` change to BOTH
   `build-small-hackathon/copy-campfire` (canonical, needs the campfire-votes token) and
   `gr33r/copy-campfire` (private mirror), then restart the org one.
9. This-machine quirks: background shells reset cwd (use absolute paths + PYTHONPATH);
   stock python urllib needs certifi (handled in `uxft/review_repo.py`).

## Deferred work (in priority order)

1. **Video + social post** (owner, by 2026-06-15): beats + verified numbers in
   docs/VIDEO_OUTLINE.md; **submission drop happens via the Gradio Discord**. For the
   recording window, pin a warm GPU (`min_containers=1`, see flip section above).
2. **DPO v2** once ≥~300 campfire votes: corpus = `ux-writing-arena-votes` (seed rows
   tagged `source=author_blind_review`, live rows `copy_campfire`). TRL `DPOTrainer` on
   the same Modal pipeline; ~$10–20 reserved. Re-validate blind before shipping.
3. **Vision v2**: hand-label the 25 reference screenshots (gold scaffold at
   `/tmp/ux-writing-neweval/` — regenerate via `data_build/annotate_new_eval.py` if tmp
   was cleared), append as test/consistency rows, run `modal_app/eval_consistency.py` on
   the 1b adapter. Note field-name shim: its output uses `adapter_out`/`base_out` for
   `eval/score_real_bug_eval.py`.
4. **Benchmark v2** — see next section (owner's chosen next step).

## Next: ux-writing-bench v2 + cost-vs-performance chart (owner's brief)

Goal: a **realistic UX writing benchmark** with far better examples and richer context
than the current 90-item set, then run ux-writing-1 against other base models and chart
**cost vs performance**.

Why the current benchmark falls short (owner's blinded-review notes, EVAL_RESULTS §4):
~11 trivial items where models converge; ambiguous content-type/context; a few flawed
golds; under-specified inputs that invite hallucination.

Suggested design (starting points, not decisions):
- **Seed from the Micro Microcopy Challenge** (owner-authored course; already modeled in
  the old snapshot at `SNAP/uxft/microcopy_challenge.py` — 170 rewrite + 24 keep + 10
  voice scenarios, 14 categories incl. tooltip/error_page/interstitial/toggle). SNAP =
  `/Users/christophergreer/Documents/Codex/2026-05-30/hugging-face-plugin-hugging-face-openai`.
- **Context-rich items**: every item carries real(istic) code context (component + props
  + sibling strings), surface, audience, and user state — the things the reviewer notes
  said were missing. Source real context from the MIT-licensed corpus already curated
  (Cal.com / Ghost / Excalidraw) via `uxft/scan.py`.
- **Discriminative by construction**: drop any item where base and fine-tune converge
  (pilot-run both, keep disagreements + hard keeps); include restraint traps and
  safety-critical items scored asymmetrically.
- **Scoring**: blinded human A/B remains the gold standard (tooling: `eval/make_ab_sheet.py`).
  For breadth across 4–6 models, add an LLM-judge pass **calibrated against the human
  subset** (report agreement rate; never headline judge-only numbers) — consistent with
  the project's honest-eval principle.
- **Contenders**: ux-writing-1, Qwen3.6-27B base, plus ≤32B-class peers (e.g. current
  Gemma/Mistral/Llama instruct models — pick by hackathon-era leaderboards), with
  frontier API list-price points as reference dots.
- **Chart**: x = measured $/1K strings (reuse `modal_app/bench_throughput.py` per model),
  y = win-rate vs ux-writing-1 (or Elo from pairwise judging). One scatter; that's the
  hero image for the README and any follow-up post.
- Infra reuse: `modal_app/eval_rewrite.py` already parameterizes `model_repo`; the
  unified dataset schema + `split_dedup` keep leakage out; Campfire could even host the
  multi-model pairs later.

## Key file map

```
uxft/            scan, review_repo (CLI), schema (both task shapes), eval heuristics,
                 benchmark client, consistency postfilter, dataset helpers
modal_app/       train (1a/1b QLoRA), serve_openai (chat+battle), eval_rewrite,
                 eval_consistency, merge_and_push, convert_gguf, bench_throughput, arch_check
space/           Copy Campfire (design-system Gradio app + battle_corpus.json)
data_build/      build_unified_dataset, annotate_new_eval, seed_arena_votes
eval/            score_rewrite_preds (honest scorer), make_ab_sheet (blinded A/B),
                 score_real_bug_eval, verify_realbug_leakage, build_real_bug_eval
docs/            EVAL_RESULTS, COST_NOTES, FINETUNE_GUIDE, VIDEO_OUTLINE, HANDOFF (this),
                 model cards, DATASET_CARD, demo/ (Cal.com artifact), runbook
scripts/         train_on_your_styleguide.py (community HF Jobs script)
reviews/         blinded A/B sheets + keys (the 83% evidence)
```
