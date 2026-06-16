"""Batched review of PostHog candidate strings on one A100 — measured tokens for the cost demo.

Same serving config as the published throughput bench (direct mode, greedy, batch 16,
max 192 new tokens). Adds per-row prompt/completion token accounting and uploads partial
results every 50 batches so a dying run still leaves an artifact.

    modal run --detach modal_app/review_posthog.py
"""

import re

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

# --- inlined training contract (mirrors uxft.policy.SYSTEM_PROMPT; this file must be
# self-contained: cross-file imports break inside the Modal container) ---
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


# mirrors uxft.escape_postfilter.is_contract_escape (inlined: this file must be self-contained
# for the Modal container — cross-file imports break with ModuleNotFoundError).
_PLACEHOLDER_RE = re.compile(r"\{\{\s*[\w.]+\s*\}\}")  # simple {{ var }} only (a ternary inside braces is still an escape)
_TERNARY_RE = re.compile(r"\{[^{}]*\?[^{}]*:[^{}]*\}")
_OPERATOR_RE = re.compile(r"===|!==|=>")
_JSX_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")


def is_contract_escape(suggested: str) -> bool:
    s = (suggested or "").strip()
    if not s:
        return False
    stripped = _PLACEHOLDER_RE.sub("", s)
    if _TERNARY_RE.search(stripped) or _OPERATOR_RE.search(stripped) or _JSX_TAG_RE.search(stripped):
        return True
    return bool(s.startswith("{") and s.endswith("}") and not re.fullmatch(r"\{\{[^{}]*\}\}", s))


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
def review(candidates_path: str = CANDIDATES_PATH, run_path: str = RUN_JSON_PATH,
           review_path: str = REVIEW_JSONL_PATH):
    import json
    import time

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    api = HfApi()

    cand_file = hf_hub_download(DATASET_REPO, candidates_path, repo_type="dataset",
                                cache_dir=HF_CACHE)
    candidates = [json.loads(l) for l in open(cand_file, encoding="utf-8") if l.strip()]
    print(f"{len(candidates)} candidates")

    t_load0 = time.time()
    tok = AutoTokenizer.from_pretrained(MERGED_REPO, trust_remote_code=True, cache_dir=HF_CACHE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation
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
            escape = bool(suggested) and is_contract_escape(suggested)
            is_change = (not escape) and bool(suggested) and " ".join(suggested.split()) != " ".join(c["current_copy"].split())
            changed += is_change
            rows.append({**c, "suggested_copy": suggested,
                         "reason": (parsed or {}).get("reason", "" if ok else raw[-300:]),
                         "risk": "contract_escape" if escape else (parsed or {}).get("risk", "" if ok else "non_json_output"),
                         "valid_json": ok, "changed": is_change})
        if (b + 1) % CHECKPOINT_EVERY == 0 or b + 1 == n_batches:
            elapsed = time.time() - t0
            summary = _summary(len(rows), elapsed, load_s, prompt_tokens,
                               completion_tokens, valid, changed, done=(b + 1 == n_batches))
            api.upload_file(path_or_fileobj=json.dumps(summary, indent=2).encode(),
                            path_in_repo=run_path, repo_id=DATASET_REPO, repo_type="dataset")
            api.upload_file(
                path_or_fileobj="\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode(),
                path_in_repo=review_path, repo_id=DATASET_REPO, repo_type="dataset")
            print(f"batch {b + 1}/{n_batches} | {summary['strings_per_hour']}/hr | "
                  f"valid {valid}/{len(rows)} | changed {changed}")
    print(json.dumps(_summary(len(rows), time.time() - t0, load_s, prompt_tokens,
                              completion_tokens, valid, changed, done=True), indent=2))


@app.local_entrypoint()
def main(candidates: str = CANDIDATES_PATH, run_json: str = RUN_JSON_PATH,
         review_out: str = REVIEW_JSONL_PATH):
    call = review.spawn(candidates, run_json, review_out)
    print(f"spawned posthog review; fc={call.object_id}")
    print(f"  candidates={candidates} -> review={review_out}, run={run_json}")
