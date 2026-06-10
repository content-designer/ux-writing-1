# Cost notes — what reviewing UX copy with ux-writing-1 actually costs

All first-party numbers below are **measured** (June 10, 2026), reproducible with
[modal_app/bench_throughput.py](../modal_app/bench_throughput.py); the raw result JSON is
archived at `gr33r/ux-writing-sft · eval_preds/throughput.json`. Frontier-API figures are
**list-price estimates**, labeled as such.

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
including container cold-start ≈ 1,200 strings/hour ≈ **~$2 per 1,000 strings**. This is
the lazy path — no batching, scale-to-zero serving. Batch for the 6× saving above.

## Estimated: the same workload on a frontier API

Per review item ≈ ~450 input tokens (system contract + string + code context) and
~80–120 output tokens. At Claude Opus 4.5 list prices ($5/M input, $25/M output):

- ~**$4.75 per 1,000 strings** (standard API)
- ~**$2.40 per 1,000 strings** (with the 50% batch-API discount)

So, at list prices: **~8–15× cheaper** than Opus when batched, roughly at parity with
Opus-batch if you use our endpoint lazily. Estimates, not measurements — frontier prices
and your prompt sizes will vary. Quality comparison vs frontier models is **not claimed**
anywhere; our measured quality claim is vs the base model (83% blinded preference,
[EVAL_RESULTS.md](EVAL_RESULTS.md)).

## The part pricing tables miss

- **Privacy**: unshipped product copy and code context never leave your infrastructure.
- **No metering anxiety**: a flat GPU-hour covers ~8K strings; rerun on every PR.
- **$0 marginal local**: the Q4_K_M GGUF runs on a 24 GB laptop (LM Studio / Ollama).
- **Tunable**: the model can learn *your* style guide for ~$5 ([FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)).

## Worked example: a real codebase

Scanning Cal.com (open-source scheduling app, ~7,700 files): 200 candidate strings
extracted in seconds, reviewed in 9.9 min for ~$0.40 through the unbatched endpoint —
the model suggested 13 changes and **kept 187 strings as-is** (restraint is trained, not
hoped for). Artifact: [demo/](demo/).
