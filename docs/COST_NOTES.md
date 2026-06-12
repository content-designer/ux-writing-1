# Cost notes — what reviewing UX copy with ux-writing-1 actually costs

All first-party numbers below are **measured** (June 10–12, 2026), reproducible with
[modal_app/bench_throughput.py](../modal_app/bench_throughput.py) and
[modal_app/review_posthog.py](../modal_app/review_posthog.py); raw result JSON is archived
at `gr33r/ux-writing-sft · eval_preds/`. Frontier-API figures are **list-price
estimates**, labeled as such.

## Measured: PostHog at codebase scale (June 12, 2026)

The headline run: a 10,000-string review of [PostHog](https://github.com/PostHog/posthog)
(MIT) pinned at `e30273b` — 152,713 raw strings scanned from `frontend/` + `products/`,
26,061 after UI-copy filters, 10,000 reviewed (seeded sample; provenance in
[demo/posthog_scan_meta.json](demo/posthog_scan_meta.json)). Same serving config as the
throughput bench below; per-request token accounting.

| metric | value |
|---|---|
| Strings reviewed | **10,000** |
| Wall-clock | **77.2 min** (76.3 review + 0.9 model load) on one A100-80GB |
| **Measured cost** | **$3.22** ($0.32 per 1,000 strings, GPU @ $2.50/h list) |
| Throughput | 7,865 strings/hour |
| Tokens | 3,590,383 prompt + 313,293 completion |
| Valid JSON contract | **9,999 / 10,000 (99.99%)** |
| Restraint | **994 changed, 9,006 kept as-is** |

**The same tokens at API list prices** (pulled 2026-06-12, sources in
[demo/llm_api_prices.json](demo/llm_api_prices.json)) — estimates, not measured runs:

| model | same-workload bill | per 1K strings | vs measured |
|---|---|---|---|
| Qwen3.6-27B via DeepInfra (the base model, rented) | $2.15 | $0.21 | 0.7× |
| Claude Opus 4.8 | $25.78 | $2.58 | **≈8×** |
| GPT-5.5 | $27.35 | $2.73 | **≈8.5×** |

Caveats (also embedded in [demo/posthog_cost_report.json](demo/posthog_cost_report.json)):
token counts use the Qwen3.6 tokenizer (others differ ±~15%); reasoning-mode APIs bill
hidden thinking tokens as output, which would raise the frontier bills; **no quality
comparison vs frontier models is claimed** — the quality claim remains 83% blinded human
preference vs this model's own base ([EVAL_RESULTS.md](EVAL_RESULTS.md)). This measured
run supersedes the ~450-token assumption in the older estimate section below.

## Measured: batched review on one rented A100-80GB

| metric | value |
|---|---|
| Mode | direct (`enable_thinking=False`), greedy, batch 16, max 192 new tokens |
| Throughput | **7,951 strings/hour** (192 strings in 86.9 s) |
| **Cost** | **$0.31 per 1,000 strings** (Modal A100-80GB @ $2.50/h list) |
| Valid JSON contract | **192/192 = 100%** |
| Avg output | 33.1 tokens per review |

## Measured: unbatched, through the OpenAI-compatible endpoint

Real-world CLI run (`uxft.review_repo`, 8 workers) against the live Modal endpoint,
reviewing 200 strings scanned from the Cal.com monorepo: **9.9 minutes end-to-end**
including container cold-start ≈ 1,200 strings/hour ≈ **≈$2 per 1,000 strings**. This is
the lazy path — no batching, scale-to-zero serving. Batch for the 6× saving above.

## Estimated: the same workload on a frontier API

Per review item ≈ ≈450 input tokens (system contract + string + code context) and
≈80–120 output tokens. At Claude Opus 4.5 list prices ($5/M input, $25/M output):

- ~**$4.75 per 1,000 strings** (standard API)
- ~**$2.40 per 1,000 strings** (with the 50% batch-API discount)

So, at list prices: **≈8–15× cheaper** than Opus when batched, roughly at parity with
Opus-batch if you use our endpoint lazily. Estimates, not measurements — frontier prices
and your prompt sizes will vary. Quality comparison vs frontier models is **not claimed**
anywhere; our measured quality claim is vs the base model (83% blinded preference,
[EVAL_RESULTS.md](EVAL_RESULTS.md)).

## The part pricing tables miss

- **Privacy**: unshipped product copy and code context never leave your infrastructure.
- **No metering anxiety**: a flat GPU-hour covers ≈8K strings; rerun on every PR.
- **$0 marginal local**: the Q4_K_M GGUF runs on a 24 GB laptop (LM Studio / Ollama).
- **Tunable**: the model can learn *your* style guide for ≈$5 ([FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)).

## Worked example: a real codebase

Scanning Cal.com (open-source scheduling app, ≈7,700 files): 200 candidate strings
extracted in seconds, reviewed in 9.9 min for ≈$0.40 through the unbatched endpoint —
the model suggested 13 changes and **kept 187 strings as-is** (restraint is trained, not
hoped for). Artifact: [demo/](demo/).
