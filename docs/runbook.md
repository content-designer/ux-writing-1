# Runbook

## Phase 1: Build the Derived Dataset

1. Run Firecrawl ingestion only for sources you are allowed to use as references.
2. Keep raw scraped content in `data/raw`; do not use it as training completions.
3. Distill public pages with `python3 -m uxft.distill_policy`.
4. Generate the SFT and benchmark files with `python3 -m uxft.dataset --validate`.
4. Review rows with `metadata.example_type=repo_candidate_needs_review` before training on them.

## Phase 2: Establish a Baseline

1. Run the benchmark through a prompted base model.
2. Save predictions as JSONL rows with `id`, `rewrite`, `reason`, `risk`, and `confidence`.
3. Score predictions with `python3 -m uxft.eval --predictions path/to/predictions.jsonl`.
4. Record useful rewrite rate from human review; heuristic scores are only a smoke test.

Command template:

```bash
python3 -m uxft.benchmark_model \
  --endpoint OPENAI_COMPATIBLE_CHAT_COMPLETIONS_URL \
  --model BASELINE_MODEL \
  --api-key-env API_KEY_ENV_VAR \
  --out data/eval/baseline_predictions.jsonl

python3 -m uxft.eval \
  --benchmark data/eval/benchmark.jsonl \
  --predictions data/eval/baseline_predictions.jsonl \
  --out data/eval/baseline_scores.json
```

## Phase 3: Train the Adapter

1. Upload `train.jsonl` and `benchmark.jsonl` as a private dataset repo.
2. Start with `Qwen/Qwen2.5-3B-Instruct` or a currently verified small Gemma instruct model.
3. Train LoRA adapters with `scripts/train_sft_hf_job.py`.
4. Compare post-SFT benchmark predictions to the prompt-only baseline.

## Phase 4: Repo-Scale Review

1. Scan one small repo first with `python3 -m uxft.review_repo`.
2. Check false positives and harmful ambiguity manually.
3. Then run against a larger repo and measure cost per 1,000 candidate strings.
4. Do not auto-apply suggestions until the model consistently beats the baseline.

## AutoScientist-Inspired Loop

Use AutoScientist as a product reference, not as a dependency. The local loop is:

1. Add or revise data slices.
2. Train a small adapter.
3. Score against the fixed benchmark.
4. Promote only slices and recipes that improve human-accepted rewrites.
