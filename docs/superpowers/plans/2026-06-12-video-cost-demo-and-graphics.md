# Video Cost Demo + Campfire Graphics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measured cost showcase (ux-writing-1 batched over PostHog's UI strings vs GPT-5.5/Opus list prices) + three Campfire-themed animated HTML graphics for the Build Small video.

**Architecture:** Local scan (hardened `uxft/scan.py`) → candidates JSONL uploaded to the private dataset repo → one detached Modal A100 batched-review job (self-contained app file, same serving config as the published throughput bench, with per-row token accounting) → results JSON pulled locally → `scripts/cost_compare.py` produces a cited cost report → `scripts/build_video_assets.py` injects report + adapter-delta JSON into three self-contained 1920×1080 HTML pages.

**Tech Stack:** Python 3.11+, Modal (app `uxw1-bench` pattern), transformers bf16 batched generation, huggingface_hub, safetensors+numpy (adapter math, no torch needed locally), vanilla HTML/CSS/JS with Copy Campfire tokens.

**Working dir:** `/Users/christophergreer/ux-writing-fine-tune` (run all commands from here). Tests: `python3 -m pytest tests/ -q`.

---

### Task 1: Scanner guards (port from bench)

**Files:**
- Modify: `uxft/scan.py` (iter_files + scan_file)
- Test: `tests/test_scan_guards.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
"""Guards ported from ux-writing-bench: oversized files skipped, degenerate contexts dropped."""
from pathlib import Path

from uxft.scan import MAX_CONTEXT_CHARS, MAX_FILE_BYTES, scan_repo


def _write(root: Path, name: str, text: str) -> Path:
    p = root / name
    p.write_text(text, encoding="utf-8")
    return p


def test_oversized_file_is_skipped(tmp_path):
    _write(tmp_path, "ok.ts", 'const a = { label: "Save your changes" };\n')
    big = 'const x = { label: "Minified bundle string" };' + ("x" * (MAX_FILE_BYTES + 100))
    _write(tmp_path, "bundle.min.js", big)
    found = scan_repo(tmp_path)
    assert any(c.path == "ok.ts" for c in found)
    assert not any(c.path == "bundle.min.js" for c in found)


def test_degenerate_context_is_dropped(tmp_path):
    # one enormous single line -> context window exceeds MAX_CONTEXT_CHARS
    pad = "y" * (MAX_CONTEXT_CHARS + 500)
    _write(tmp_path, "oneline.ts", f'const q = {{ label: "Delete this workspace" }}; // {pad}\n')
    assert scan_repo(tmp_path) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_scan_guards.py -q`
Expected: FAIL — `ImportError: cannot import name 'MAX_CONTEXT_CHARS'`

- [ ] **Step 3: Implement guards in `uxft/scan.py`**

Add constants after `SKIP_DIRS` (line ~22):

```python
MAX_FILE_BYTES = 1_000_000   # skip generated/minified bundles (OOM guard, ported from bench)
MAX_CONTEXT_CHARS = 2_000    # drop candidates with degenerate contexts
```

In `iter_files`, after the SKIP_DIRS check:

```python
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
```

In `scan_file`, guard the candidate append — compute context first:

```python
            context = context_for(lines, line_no)
            if len(context) > MAX_CONTEXT_CHARS:
                continue
            candidates.append(
                Candidate(
                    path=str(path.relative_to(root)),
                    line=line_no,
                    kind=classify(prefix, value, path),
                    current_copy=value,
                    context=context,
                )
            )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_scan_guards.py tests/ -q`
Expected: new tests PASS, full suite stays green.

- [ ] **Step 5: Commit** — `git add uxft/scan.py tests/test_scan_guards.py && git commit -m "feat: port file-size and context guards into uxft.scan"`

---

### Task 2: Scan PostHog (local, free — gates all spend)

**Files:**
- Create (artifact, not committed): `/tmp/posthog` clone, `/tmp/posthog_candidates.jsonl`
- Create: `docs/demo/posthog_scan_meta.json` (committed — provenance)

- [ ] **Step 1: Shallow-clone and pin**

```bash
git clone --depth 1 https://github.com/PostHog/posthog /tmp/posthog
git -C /tmp/posthog rev-parse HEAD   # record SHA
```

- [ ] **Step 2: Scan, capped at 10,000**

```bash
python3 -m uxft.scan /tmp/posthog --limit 10000 --out /tmp/posthog_candidates.jsonl
```

Expected: `wrote N candidates` with N likely = 10000 (cap hit). Sanity-eyeball ~10 rows (`head`) — strings should look like real UI copy, not identifiers. If yield is junk-heavy or < ~2,000, rerun scoped to `frontend/` (`python3 -m uxft.scan /tmp/posthog/frontend ...` — paths become frontend-relative; note it in meta). Fallback repo per spec: `supabase/supabase`.

- [ ] **Step 3: Write provenance meta**

`docs/demo/posthog_scan_meta.json` (hand-write from the run output):

```json
{
  "repo": "https://github.com/PostHog/posthog",
  "license": "MIT",
  "commit": "<sha from step 1>",
  "scanned_at": "2026-06-12",
  "scan_root": "<repo root or frontend/>",
  "candidate_cap": 10000,
  "candidates_written": "<N>",
  "scanner": "uxft.scan @ <this repo's HEAD sha>"
}
```

- [ ] **Step 4: Upload candidates for the Modal job**

```bash
python3 - <<'EOF'
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/tmp/posthog_candidates.jsonl",
                    path_in_repo="eval_preds/posthog_candidates.jsonl",
                    repo_id="gr33r/ux-writing-sft", repo_type="dataset")
print("uploaded")
EOF
```

- [ ] **Step 5: Commit** — `git add docs/demo/posthog_scan_meta.json && git commit -m "docs: PostHog scan provenance for cost demo"`

---

### Task 3: Modal batched review with token accounting

**Files:**
- Create: `modal_app/review_posthog.py` (SELF-CONTAINED — inline the system prompt, the per-candidate prompt, and the JSON extractor; cross-file imports break in the container)

- [ ] **Step 1: Write the app file**

```python
"""Batched review of PostHog candidate strings on one A100 — measured tokens for the cost demo.

Same serving config as the published throughput bench (direct mode, greedy, batch 16,
max 192 new tokens). Adds per-row prompt/completion token accounting and uploads partial
results every 50 batches so a dying run still leaves an artifact.

    modal run --detach modal_app/review_posthog.py
"""

import modal

MERGED_REPO = "gr33r/ux-writing-1"
DATASET_REPO = "gr33r/ux-writing-sft"
CANDIDATES_PATH = "eval_preds/posthog_candidates.jsonl"
RUN_JSON_PATH = "eval_preds/posthog_run.json"
REVIEW_JSONL_PATH = "eval_preds/posthog_review.jsonl"
MODEL_CACHE = "/cache"
HF_CACHE = "/cache/hf"
A100_USD_PER_HOUR = 2.50
BATCH_SIZE = 16
MAX_NEW_TOKENS = 192
CHECKPOINT_EVERY = 50  # batches

# --- inlined training contract (mirrors uxft.policy.SYSTEM_PROMPT) ---
SYSTEM_PROMPT = """You are a senior UX writer reviewing interface copy in product code.
Rewrite the UI copy so it is purposeful, concise, conversational, clear, and accessible.
If the current copy is already clear, accurate, and on-brand, keep it unchanged: return it verbatim as the rewrite and say so in the reason.
Preserve product intent. Do not invent actions, facts, or product behavior that are not in the context.
Keep locale-specific terms (for example, "Postal code" for Canadian addresses) and any {{ variables }} exactly as written.
Never weaken safety-critical copy: destructive, payment, privacy, and security messages must keep their consequence and must not be softened.
Return compact JSON with: rewrite, reason, and risk. Use an empty string for risk when none applies."""


def prompt_for(c: dict) -> str:
    # mirrors uxft.review_repo.prompt_for
    return (
        "Product surface: existing codebase\n"
        "Audience: product user\n"
        f"User state: using the screen that contains {c['path']}:{c['line']}\n"
        f"Content type: {c['kind']}\n"
        f"Current copy: {c['current_copy']}\n"
        f"Code/context:\n{c['context']}\n"
        "Constraints: Suggest a UX writing rewrite only if the context supports it. "
        "Preserve the intended product behavior."
    )


def extract_contract_json(text: str):
    # mirrors uxft.review_repo.extract_contract_json (raw_decode scan, {{var}}-safe)
    import json
    if not isinstance(text, str):
        return None
    tail = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    parsed = None
    index = 0
    while True:
        start = tail.find("{", index)
        if start == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(tail[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("rewrite"), str):
            parsed = obj
        index = start + max(consumed, 1)
    return parsed


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers>=4.57.0", "accelerate>=1.0.0",
        "huggingface_hub>=0.34.0", "hf_transfer",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("uxw1-posthog-review", image=image)
weights_vol = modal.Volume.from_name("qwen36-27b-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")


@app.function(gpu="A100-80GB", volumes={MODEL_CACHE: weights_vol}, secrets=[hf_secret],
              timeout=6 * 60 * 60)
def review():
    import json
    import time

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    api = HfApi()

    cand_file = hf_hub_download(DATASET_REPO, CANDIDATES_PATH, repo_type="dataset",
                                cache_dir=HF_CACHE)
    candidates = [json.loads(l) for l in open(cand_file, encoding="utf-8") if l.strip()]
    print(f"{len(candidates)} candidates")

    t_load0 = time.time()
    tok = AutoTokenizer.from_pretrained(MERGED_REPO, trust_remote_code=True, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        MERGED_REPO, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, cache_dir=HF_CACHE,
    ).eval()
    load_s = time.time() - t_load0

    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt_for(c)}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        for c in candidates
    ]

    # warmup (excluded from timing)
    warm = tok(texts[:BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=32, do_sample=False, pad_token_id=tok.pad_token_id)

    rows, prompt_tokens, completion_tokens, valid, changed = [], 0, 0, 0, 0
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()
    for b, i in enumerate(range(0, len(texts), BATCH_SIZE)):
        enc = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True).to(model.device)
        prompt_tokens += int(enc["attention_mask"].sum())
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        completion_tokens += int((new != tok.pad_token_id).sum())
        for c, raw in zip(candidates[i:i + BATCH_SIZE],
                          tok.batch_decode(new, skip_special_tokens=True)):
            parsed = extract_contract_json(raw)
            ok = parsed is not None
            valid += ok
            suggested = (parsed or {}).get("rewrite", "")
            is_change = bool(suggested) and " ".join(suggested.split()) != " ".join(c["current_copy"].split())
            changed += is_change
            rows.append({**c, "suggested_copy": suggested,
                         "reason": (parsed or {}).get("reason", "" if ok else raw[-300:]),
                         "risk": (parsed or {}).get("risk", "" if ok else "non_json_output"),
                         "valid_json": ok, "changed": is_change})
        if (b + 1) % CHECKPOINT_EVERY == 0 or b + 1 == n_batches:
            elapsed = time.time() - t0
            summary = _summary(len(rows), elapsed, load_s, prompt_tokens,
                               completion_tokens, valid, changed, done=(b + 1 == n_batches))
            api.upload_file(path_or_fileobj=json.dumps(summary, indent=2).encode(),
                            path_in_repo=RUN_JSON_PATH, repo_id=DATASET_REPO, repo_type="dataset")
            api.upload_file(
                path_or_fileobj="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode(),
                path_in_repo=REVIEW_JSONL_PATH, repo_id=DATASET_REPO, repo_type="dataset")
            print(f"batch {b + 1}/{n_batches} | {summary['strings_per_hour']}/hr | "
                  f"valid {valid}/{len(rows)} | changed {changed}")
    print(json.dumps(_summary(len(rows), time.time() - t0, load_s, prompt_tokens,
                              completion_tokens, valid, changed, done=True), indent=2))


def _summary(n, elapsed, load_s, ptok, ctok, valid, changed, done):
    per_hour = n / elapsed * 3600 if elapsed else 0
    return {
        "model": MERGED_REPO, "gpu": "A100-80GB", "gpu_usd_per_hour": A100_USD_PER_HOUR,
        "mode": "direct (enable_thinking=False), greedy",
        "batch_size": BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
        "strings": n, "review_elapsed_s": round(elapsed, 1), "model_load_s": round(load_s, 1),
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "strings_per_hour": round(per_hour),
        "usd_per_1k_strings": round(1000 / per_hour * A100_USD_PER_HOUR, 2) if per_hour else None,
        "measured_gpu_usd": round((elapsed + load_s) / 3600 * A100_USD_PER_HOUR, 2),
        "valid_json": valid, "changed": changed, "kept": n - changed,
        "complete": done,
    }


@app.local_entrypoint()
def main():
    call = review.spawn()
    print(f"spawned posthog review; fc={call.object_id}")
```

- [ ] **Step 2: Syntax-check + commit before launching**

Run: `python3 -m py_compile modal_app/review_posthog.py`
`git add modal_app/review_posthog.py && git commit -m "feat: Modal batched PostHog review with token accounting"`

- [ ] **Step 3: Launch detached**

Run: `modal run --detach modal_app/review_posthog.py`
Expected: prints `spawned posthog review; fc=fc-...`. Track via `modal app logs uxw1-posthog-review` (or proceed and poll the HF artifact — checkpoints land every 50 batches).

- [ ] **Step 4 (while it runs): proceed to Tasks 4–6; poll completion**

```bash
python3 - <<'EOF'
import json
from huggingface_hub import hf_hub_download
p = hf_hub_download("gr33r/ux-writing-sft", "eval_preds/posthog_run.json",
                    repo_type="dataset", force_download=True)
print(json.dumps(json.load(open(p)), indent=2))
EOF
```

Done when `"complete": true`.

---

### Task 4: Pricing snapshot + `scripts/cost_compare.py`

**Files:**
- Create: `docs/demo/llm_api_prices.json` (snapshot of the collector pull)
- Create: `scripts/cost_compare.py`
- Test: `tests/test_cost_compare.py`

- [ ] **Step 1: Snapshot prices** — pull the Prometheus collector (`prometheus scripts data RAHfjFf-Md7V4LaC4AbvD`), keep only the rows we cite, and save:

```json
{
  "pulled_at": "2026-06-12",
  "collector": "Prometheus 'LLM API pricing' (self-healing, weekly)",
  "prices_usd_per_mtok": {
    "gpt-5.5":        {"input": 5.0,  "output": 30.0, "source": "https://developers.openai.com/api/docs/pricing"},
    "claude-opus-4.8":{"input": 5.0,  "output": 25.0, "source": "https://claude.com/pricing (collector-verified)"},
    "qwen3.6-27b-deepinfra": {"input": 0.32, "output": 3.2, "source": "deepinfra (manual check, configs/models.yaml 2026-06-10)"}
  }
}
```

(Fill `source` for Anthropic with the collector's actual `source_url` field.)

- [ ] **Step 2: Write the failing test**

```python
import json

from scripts.cost_compare import build_report

RUN = {"strings": 10000, "review_elapsed_s": 4500.0, "model_load_s": 300.0,
       "prompt_tokens": 5_000_000, "completion_tokens": 350_000,
       "strings_per_hour": 8000, "usd_per_1k_strings": 0.31,
       "measured_gpu_usd": 3.33, "gpu_usd_per_hour": 2.50,
       "valid_json": 9990, "changed": 700, "kept": 9300, "complete": True}
PRICES = {"pulled_at": "2026-06-12", "collector": "test",
          "prices_usd_per_mtok": {
              "gpt-5.5": {"input": 5.0, "output": 30.0, "source": "s1"},
              "claude-opus-4.8": {"input": 5.0, "output": 25.0, "source": "s2"}}}


def test_build_report_math():
    rep = build_report(RUN, PRICES)
    est = {e["model"]: e for e in rep["estimates_same_tokens_at_list_price"]}
    # 5M in * $5/M + 0.35M out * $30/M = 25 + 10.5 = 35.5
    assert est["gpt-5.5"]["usd"] == 35.5
    # 5M * 5 + 0.35M * 25 = 25 + 8.75 = 33.75
    assert est["claude-opus-4.8"]["usd"] == 33.75
    assert rep["measured"]["usd"] == 3.33
    assert est["gpt-5.5"]["multiple_vs_measured"] == round(35.5 / 3.33, 1)
    assert rep["caveats"]  # honesty rails always present
```

- [ ] **Step 3: Run to verify failure** — `python3 -m pytest tests/test_cost_compare.py -q` → FAIL (no module).

- [ ] **Step 4: Implement `scripts/cost_compare.py`**

```python
"""Build docs/demo/posthog_cost_report.json from the Modal run + price snapshot.

Measured cost is real (GPU-seconds x list $/h). Frontier figures are LIST-PRICE
ESTIMATES on OUR measured token counts — clearly labeled, never a quality claim.

    python3 scripts/cost_compare.py            # reads HF artifact + price snapshot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CAVEATS = [
    "Frontier figures are list-price estimates applied to ux-writing-1's measured "
    "prompt/completion token counts for the same workload; they are not measured runs.",
    "Token counts use the Qwen3.6 tokenizer; other providers' tokenizers differ (~±15%).",
    "Estimates assume no hidden reasoning tokens; reasoning-mode APIs bill those as "
    "output, which would raise the frontier cost.",
    "No quality comparison vs frontier models is claimed anywhere; the model's quality "
    "claim is 83% blinded human preference vs its own base model (EVAL_RESULTS.md).",
]


def build_report(run: dict, prices: dict) -> dict:
    ptok, ctok = run["prompt_tokens"], run["completion_tokens"]
    measured = {
        "model": "ux-writing-1 (batched, A100-80GB @ $%.2f/h)" % run["gpu_usd_per_hour"],
        "usd": run["measured_gpu_usd"],
        "strings": run["strings"],
        "strings_per_hour": run["strings_per_hour"],
        "usd_per_1k_strings": run["usd_per_1k_strings"],
        "wall_clock_min": round((run["review_elapsed_s"] + run["model_load_s"]) / 60, 1),
        "valid_json": run["valid_json"], "changed": run["changed"], "kept": run["kept"],
    }
    estimates = []
    for model, p in prices["prices_usd_per_mtok"].items():
        usd = round(ptok / 1e6 * p["input"] + ctok / 1e6 * p["output"], 2)
        estimates.append({
            "model": model, "usd": usd,
            "usd_per_1k_strings": round(usd / run["strings"] * 1000, 2),
            "multiple_vs_measured": round(usd / measured["usd"], 1) if measured["usd"] else None,
            "price_in_per_mtok": p["input"], "price_out_per_mtok": p["output"],
            "source": p["source"],
        })
    return {
        "workload": {"prompt_tokens": ptok, "completion_tokens": ctok,
                     "strings": run["strings"]},
        "measured": measured,
        "estimates_same_tokens_at_list_price": sorted(estimates, key=lambda e: e["usd"]),
        "prices_pulled_at": prices["pulled_at"],
        "caveats": CAVEATS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", type=Path, default=None,
                    help="local posthog_run.json; default: fetch from the dataset repo")
    ap.add_argument("--prices", type=Path, default=Path("docs/demo/llm_api_prices.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/demo/posthog_cost_report.json"))
    args = ap.parse_args()

    if args.run_json:
        run = json.loads(args.run_json.read_text())
    else:
        from huggingface_hub import hf_hub_download
        run = json.loads(Path(hf_hub_download(
            "gr33r/ux-writing-sft", "eval_preds/posthog_run.json",
            repo_type="dataset", force_download=True)).read_text())
    if not run.get("complete"):
        raise SystemExit("run artifact is a partial checkpoint — wait for complete: true")
    report = build_report(run, json.loads(args.prices.read_text()))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(Add empty `scripts/__init__.py` if pytest can't import; repo already has `scripts/` as plain dir — import via `from scripts.cost_compare import build_report` works with rootdir on sys.path; otherwise add `conftest.py` with `sys.path.insert(0, ".")`.)

- [ ] **Step 5: Run tests** — `python3 -m pytest tests/test_cost_compare.py -q` → PASS.

- [ ] **Step 6: Commit** — `git add scripts/cost_compare.py tests/test_cost_compare.py docs/demo/llm_api_prices.json && git commit -m "feat: cost comparison report builder with honesty caveats"`

---

### Task 5: Adapter delta heatmap data (`scripts/adapter_heatmap.py`)

**Files:**
- Create: `scripts/adapter_heatmap.py`
- Test: `tests/test_adapter_heatmap.py`
- Output: `docs/video_assets/data/adapter_deltas.json`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from safetensors.numpy import save_file

from scripts.adapter_heatmap import compute_deltas, fro_norm_ba


def test_fro_norm_ba_matches_dense():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 64)).astype(np.float32)   # lora_A: r x in
    B = rng.normal(size=(32, 8)).astype(np.float32)   # lora_B: out x r
    dense = float(np.linalg.norm(B @ A))
    assert abs(fro_norm_ba(A, B) - dense) / dense < 1e-5


def test_compute_deltas_groups_by_layer_and_module(tmp_path):
    rng = np.random.default_rng(1)
    tensors = {}
    for layer in (0, 1):
        for mod in ("q_proj", "gate_proj"):
            base = f"base_model.model.model.layers.{layer}.x.{mod}"
            tensors[f"{base}.lora_A.weight"] = rng.normal(size=(4, 16)).astype(np.float32)
            tensors[f"{base}.lora_B.weight"] = rng.normal(size=(16, 4)).astype(np.float32)
    f = tmp_path / "adapter_model.safetensors"
    save_file(tensors, str(f))
    deltas = compute_deltas(f)
    assert len(deltas) == 4
    assert {d["module"] for d in deltas} == {"q_proj", "gate_proj"}
    assert {d["layer"] for d in deltas} == {0, 1}
    assert all(d["norm"] > 0 for d in deltas)
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_adapter_heatmap.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
"""Compute per-(layer, module) Frobenius norms of the LoRA delta for the weights visual.

||B@A||_F is computed via Gram matrices — trace identity:
||BA||_F^2 = sum((B^T B) * (A A^T)) — so we never materialize the (out x in) delta.

    python3 scripts/adapter_heatmap.py        # downloads gr33r/ux-writing-1-lora
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ADAPTER_REPO = "gr33r/ux-writing-1-lora"
KEY_RE = re.compile(r"\.layers\.(?P<layer>\d+)\.(?:.*\.)?(?P<module>[A-Za-z0-9_]+)\.lora_A\.weight$")


def fro_norm_ba(A: np.ndarray, B: np.ndarray) -> float:
    A = A.astype(np.float32); B = B.astype(np.float32)
    return float(np.sqrt(np.sum((B.T @ B) * (A @ A.T))))


def compute_deltas(safetensors_path: Path) -> list[dict]:
    from safetensors.numpy import load_file
    tensors = load_file(str(safetensors_path))
    out = []
    for key, A in tensors.items():
        m = KEY_RE.search(key)
        if not m:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in tensors:
            continue
        out.append({"layer": int(m.group("layer")), "module": m.group("module"),
                    "norm": fro_norm_ba(A, tensors[b_key])})
    out.sort(key=lambda d: (d["layer"], d["module"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-file", type=Path, default=None,
                    help="local adapter_model.safetensors; default: download from the Hub")
    ap.add_argument("--out", type=Path, default=Path("docs/video_assets/data/adapter_deltas.json"))
    args = ap.parse_args()

    path = args.adapter_file
    if path is None:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(ADAPTER_REPO, "adapter_model.safetensors"))
    deltas = compute_deltas(path)
    if not deltas:
        raise SystemExit("no lora_A/lora_B pairs matched — inspect adapter keys")
    peak = max(d["norm"] for d in deltas)
    for d in deltas:
        d["intensity"] = round(d["norm"] / peak, 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "adapter": ADAPTER_REPO,
        "metric": "Frobenius norm of B@A per (layer, module)",
        "layers": max(d["layer"] for d in deltas) + 1,
        "modules": sorted({d["module"] for d in deltas}),
        "deltas": deltas,
    }, indent=2) + "\n")
    print(f"wrote {len(deltas)} deltas to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests** — `python3 -m pytest tests/test_adapter_heatmap.py -q` → PASS (pip install safetensors numpy if missing).

- [ ] **Step 5: Run for real** — `python3 scripts/adapter_heatmap.py` (downloads the adapter; if module names differ from q/k/v/o/gate/up/down — the hybrid stack has in_proj_* names on linear-attention blocks — the regex's generic `module` group already captures them; just verify the JSON looks sane).

- [ ] **Step 6: Commit** — `git add scripts/adapter_heatmap.py tests/test_adapter_heatmap.py docs/video_assets/data/adapter_deltas.json && git commit -m "feat: real LoRA delta norms for the weights visual"`

---

### Task 6: Campfire animated graphics

**Files:**
- Create: `docs/video_assets/campfire.css` (shared tokens)
- Create: `docs/video_assets/templates/cost_compare.html`, `templates/weights_heatmap.html`, `templates/stats_banner.html` (each with a `/*__DATA__*/` injection point)
- Create: `scripts/build_video_assets.py`
- Output: `docs/video_assets/{cost_compare,weights_heatmap,stats_banner}.html`

Design rules for all three pages (tokens verbatim from `space/app.py:418-436`):
- 1920×1080 fixed stage centered on `--parch-50` (#FCF8EF); text `--bark-900`/`--bark-800`; accents `--fire-500` (#F2641B), `--ember-500` (#EE8E1E), `--gold-400` (#D8A94B), `--pine-600` (#5C6E33).
- Fonts: Google Fonts link for Archivo (900), Newsreader (italic 500), Nunito (400/700/800) + system fallbacks.
- Every page: animation auto-plays 600ms after load, and clicking anywhere replays it (multiple recording takes).
- Footnote strip in `--font-mono`-style small caps for caveats/sources — honesty rails visible *in* the video frame.

- [ ] **Step 1: Shared CSS + the three templates.** `cost_compare.html` — three horizontal bars (ux-writing-1 measured, Opus 4.8 est, GPT-5.5 est) growing left→right over 2.5s with eased dollar counters; subtitle "Same workload. Same tokens. List prices."; per-bar $/1K-strings chip; ember-particle drift behind the measured bar. `weights_heatmap.html` — layers×modules grid (CSS grid, cells colored parchment→gold→fire by `intensity`), column-staggered reveal like flames catching, axis labels, caption "Frobenius norm of the LoRA delta per module — what fine-tuning actually touched." `stats_banner.html` — four cards (83% blinded preference; strings/hr from the run; valid-JSON %; ≈$30 training) fading up staggered with Newsreader italic kickers. Data arrives as `const DATA = /*__DATA__*/;`.

- [ ] **Step 2: `scripts/build_video_assets.py`**

```python
"""Inject run artifacts into the video asset templates. No hand-typed numbers.

    python3 scripts/build_video_assets.py
"""
import json
from pathlib import Path

ASSETS = Path("docs/video_assets")

JOBS = [
    ("cost_compare.html", ["docs/demo/posthog_cost_report.json"]),
    ("weights_heatmap.html", ["docs/video_assets/data/adapter_deltas.json"]),
    ("stats_banner.html", ["docs/demo/posthog_cost_report.json"]),
]


def main() -> int:
    for name, data_paths in JOBS:
        template = (ASSETS / "templates" / name).read_text()
        payload = [json.loads(Path(p).read_text()) for p in data_paths]
        data = payload[0] if len(payload) == 1 else payload
        out = template.replace("/*__DATA__*/", json.dumps(data))
        if out == template:
            raise SystemExit(f"{name}: no /*__DATA__*/ injection point found")
        (ASSETS / name).write_text(out)
        print(f"built {ASSETS / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Build + visual check** — run the build, `open docs/video_assets/cost_compare.html` etc.; verify 1920×1080 framing, animation, replay-on-click, numbers match the JSON artifacts.

- [ ] **Step 4: Commit** — `git add docs/video_assets scripts/build_video_assets.py && git commit -m "feat: Campfire animated video assets built from run artifacts"`

(Note: stats_banner needs the run report — if the Modal job is still running, build weights_heatmap first and finish the other two when `complete: true`.)

---

### Task 7: Documentation updates (after the run completes)

**Files:**
- Modify: `docs/COST_NOTES.md` (new worked example section)
- Modify: `docs/VIDEO_OUTLINE.md` (beat 5 + claims table + b-roll checklist)

- [ ] **Step 1: COST_NOTES.md** — add a "Worked example: PostHog at scale" section below the Cal.com example with the measured numbers (strings, wall-clock, measured $, tokens) and a small table of the same-tokens list-price estimates (GPT-5.5, Opus 4.8, optional DeepInfra rung), each row citing `docs/demo/posthog_cost_report.json` + the caveats. Keep the existing "Estimated" section but note the PostHog example supersedes the guessed ~450-token assumption with measured counts.

- [ ] **Step 2: VIDEO_OUTLINE.md** — rewrite beat 5 ("scan a real codebase") for approach A: scan b-roll optional, the hero is the animated `cost_compare.html` + the headline sentence pattern: "All N UI strings in PostHog: X minutes, $Y on one rented GPU — the same tokens at GPT-5.5 list prices: $Z." Add claims-table rows (PostHog strings/cost/restraint/valid-JSON; same-tokens estimates with 'labeled estimate' flags). Add the three video assets + weights heatmap to the b-roll checklist; add a beat-3 note to flash `weights_heatmap.html` when QLoRA is mentioned.

- [ ] **Step 3: Run full test suite** — `python3 -m pytest tests/ -q` → green.

- [ ] **Step 4: Commit** — `git add docs/COST_NOTES.md docs/VIDEO_OUTLINE.md && git commit -m "docs: PostHog cost showcase numbers + video outline beats"`

---

## Self-review notes

- Spec coverage: scanner guards (T1), scan+provenance (T2), batched run with tokens (T3), cost math + citations + caveats (T4), real adapter heatmap (T5), three animated assets + build script (T6), doc updates (T7). Optional DeepInfra rung included via the price snapshot.
- Honesty rails appear in three places: report caveats (machine-readable), on-frame footnotes (video), COST_NOTES wording.
- Sequencing: T2 gates spend; T3 runs detached while T4–T6 proceed; T6 stats page + T7 wait on `complete: true`.
