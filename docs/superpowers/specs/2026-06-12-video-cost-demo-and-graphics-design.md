# Video cost demo + Campfire graphics — design

*2026-06-12. For the Build Small submission video (deadline 2026-06-15). Approved approach:
batched-only cost run (no live-endpoint segment) + animated HTML graphics from real data.*

## Goal

Two additions that make the demo video more compelling:

1. **A measured cost showcase at real-codebase scale** — run ux-writing-1 (batched, Modal
   A100) over the UI strings of a large open-source repo, capture actual token counts and
   wall-clock, and compare the measured cost against what the *same tokens* would cost at
   GPT-5.5 and Claude Opus list prices.
2. **Campfire-themed animated graphics** for screen recording — including a real
   "how the weights changed" visualization computed from the released LoRA adapter.

## Part 1 — PostHog cost run

**Target repo:** `PostHog/posthog` (MIT), pinned to a recorded commit SHA at clone time.
Chosen because it is permissively licensed, recognizable to the developer audience, rich in
real product UI strings, and has no overlap with this model's training data. (Cal.com,
Ghost, and Excalidraw are excluded — they contributed strings to training. A handful of
other repos are excluded for unrelated project conflicts.) Fallback if string yield is
poor: `supabase/supabase` (Apache-2.0), same procedure.

**Pipeline:**

1. **Scanner hardening** — port the two guards from the benchmark project's vendored
   scanner into `uxft/scan.py`: skip files > 1 MB (`MAX_FILE_BYTES`) and clamp extracted
   code context to 2K chars (`MAX_CONTEXT_CHARS`). Known failure without them: OOM on
   minified single-line bundles. Add a small regression test.
2. **Scan** — shallow-clone PostHog, run `uxft.scan` over the frontend source, record:
   commit SHA, files scanned, candidate strings found. Cap reviewed strings at 10,000
   (decided at run time based on yield; cap recorded in the artifact, disclosed in docs).
3. **Batched review on Modal** — extend the existing `modal_app/bench_throughput.py`
   pattern (same serving config that produced the published 7,951 strings/hr number:
   direct mode, `enable_thinking=False`, greedy, batch 16, max 192 new tokens) to run the
   PostHog string set end-to-end. Capture per-request **prompt tokens, completion tokens**,
   total wall-clock (including model load, reported separately), and JSON-contract
   validity. Respect the established Modal gotchas: self-contained app file, detached run.
4. **Cost math** — one small script (`scripts/cost_compare.py`) that takes the run
   artifact and produces `docs/demo/posthog_cost_report.json`:
   - **Measured:** GPU-seconds × $2.50/h (A100-80GB list).
   - **Estimated (same workload at API list prices):** measured prompt/completion token
     totals × GPT-5.5 ($5/M in, $30/M out) and Claude Opus 4.8 ($5/M in, $25/M out).
     Optionally a third rung: the same base model served via API (DeepInfra Qwen3.6-27B,
     $0.32/$3.20) as the "open model, someone else's GPU" reference point.
   - Prices are pulled from the existing pricing collector at run time and cited with
     pull date + source URLs in the JSON.

**Honesty constraints (non-negotiable, consistent with existing docs):**

- Frontier figures are **list-price estimates on our measured token counts** — labeled as
  such everywhere. Token counts come from our tokenizer; other providers' tokenizers
  differ (caveat stated, ±~15%).
- **No quality claim vs frontier models.** The only quality claim remains 83% blinded
  preference vs the base model.
- Restraint stat (strings kept as-is vs changed) reported for PostHog exactly as it falls.

**Doc updates:** new worked example + claims rows in `docs/COST_NOTES.md`; updated beat 5
and claims table in `docs/VIDEO_OUTLINE.md` (live-terminal beat replaced by graphics over
the run artifact); demo artifacts in `docs/demo/`.

## Part 2 — Campfire animated graphics (`docs/video_assets/`)

Self-contained HTML files (inline CSS/JS/data, no build step, no network dependency except
Google Fonts with system-font fallback), each framed as a 1920×1080 stage for clean screen
recording. All styled with the Copy Campfire tokens taken verbatim from `space/app.py`
(parch/bark/fire/ember/gold/pine palette; Archivo display, Newsreader serif, Nunito sans).

1. **`cost_compare.html`** — animated cost ladder. Bars grow and dollar counters tick up:
   ux-writing-1 (measured) vs GPT-5.5 vs Opus 4.8 (estimates), with the per-1K-strings
   framing and the "same tokens, list price" caption. Data injected from
   `posthog_cost_report.json` by a tiny build script — no hand-typed numbers.
2. **`weights_heatmap.html`** — the real fingerprint. A script
   (`scripts/adapter_heatmap.py`) downloads `gr33r/ux-writing-1-lora`, loads the
   safetensors on CPU, computes ‖B·A‖_F per (layer, target module), normalizes, and emits
   JSON. The page renders the layers × modules grid with an ember-glow color scale
   (parchment → gold → fire), animated reveal, with a one-line honest caption ("Frobenius
   norm of the LoRA delta per module — what fine-tuning actually touched"). Module naming
   follows whatever keys the adapter actually contains (the 27B is a hybrid-attention
   stack; group by layer index and module suffix as found).
3. **`stats_banner.html`** — hero stat cards for cutaways: 83% blinded preference,
   strings/hr from the PostHog run, 100% valid JSON (if it holds — report what falls),
   ≈$30 training cost. Staggered fade-up animation.

## Sequencing & budget

1. Scanner guards + scan PostHog (local, free) — confirms yield before any spend.
2. Batched Modal run (detached). Expected ≤ 2 A100-hours ≈ **$5 of the ~$150 credit**.
3. `cost_compare.py` → report JSON → graphics build.
4. Adapter heatmap (local CPU; adapter download is the only heavy fetch).
5. Doc updates last, from artifacts.

## Risks

- **Scan yield unknown** — gate the Modal spend on a sane local scan result; Supabase
  fallback.
- **Modal run dies with local handle** — use `.spawn()` / `modal run --detach` (known
  gotcha, already documented in HANDOFF).
- **Adapter key layout** — heatmap script must introspect keys rather than assume the
  full-attention naming; handled by grouping on observed suffixes.
- **100% JSON validity may not survive 10K unseen strings** — report the real number;
  the claims table only ever cites measured artifacts.

## Testing

- Regression test for the scanner size/context guards (fixture with an oversized file).
- `cost_compare.py` unit-tested on a synthetic run artifact (token math + price citation).
- `adapter_heatmap.py` smoke-tested on a tiny synthetic safetensors fixture.
- Graphics: visual check at 1920×1080; numbers cross-checked against the JSON artifacts.
