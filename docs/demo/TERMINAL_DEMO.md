# Terminal demo cheatsheet — running ux-writing-1 on camera

Two beats, both 100% local (no API, no metering — the exact claim the video makes).
**Pull the GGUF before you record** — the 16 GB download is not a camera moment.

```bash
ollama pull hf.co/gr33r/ux-writing-1-GGUF:Q4_K_M   # ~16.6 GB, one time, off-camera
```

---

## Beat 1 — one string, live (punchy, ~15s)

Build the friendly local model once, then run it:

```bash
ollama create ux-writing-1 -f docs/demo/Modelfile   # one time
ollama run ux-writing-1
```

At the prompt, paste a real offender:

```
Current copy: Invalid
Code/context: <input type="email" aria-label="Email" />
```

It returns the contract:

```json
{"rewrite": "Invalid email address", "reason": "Names the exact problem instead of a bare 'Invalid'.", "risk": ""}
```

> **Rehearse the thinking toggle.** Qwen3.6 is a thinking model. If it reasons out loud
> before the JSON, type `/set nothink` once in the session (or just use Beat 2, which is
> immune — its extractor strips any `</think>` reasoning). The Modelfile pre-loads the
> review contract so you don't paste a system prompt on camera.

---

## Beat 2 — review a whole repo, fully local (the money shot, ~40s)

Ollama exposes an OpenAI-compatible endpoint at `localhost:11434`, and `uxft.review_repo`
speaks it — so the entire scan → review pipeline runs on your laptop. Point it at any
repo you have checked out:

```bash
# Step 1: extract candidate strings (instant, free, local)
python -m uxft.scan ~/code/some-oss-app --limit 60 --out /tmp/scan.jsonl

# Step 2: review them through the local GGUF — diff-friendly JSONL of suggestions
python -m uxft.review_repo ~/code/some-oss-app \
  --endpoint http://localhost:11434/v1/chat/completions \
  --model hf.co/gr33r/ux-writing-1-GGUF:Q4_K_M \
  --limit 40 --out /tmp/review.jsonl

# Step 3: show the suggestions scrolling
cat /tmp/review.jsonl | python -m json.tool   # or: jq -c 'select(.suggested_copy != "")' /tmp/review.jsonl
```

The terminal prints `wrote N review rows … (M suggested changes)` — the restraint stat,
live. No `--api-key` needed: local Ollama ignores auth.

> This beat is robust to the thinking toggle — `review_repo`'s `extract_contract_json`
> takes the text after the last `</think>`, so reasoning never breaks the output.

---

## Say the right number for the right machine

| You're showing… | The honest line |
|---|---|
| The laptop GGUF (these commands) | "No API, no metering — **$0 marginal**, private, on my hardware." |
| The batched A100 (the cost graphic) | "**~8,000 strings/hour, $0.32 per 1,000** — on one rented GPU." |

Don't quote the 8K/hr throughput over the laptop demo — that's the A100 number. The
laptop's story is *zero marginal cost and privacy*, not speed.

---

## If you'd rather hit the hosted endpoint instead of the laptop

The Modal app `uxw1-serve` is OpenAI-compatible too. Once you have its URL, swap into
Beat 2: `--endpoint https://<your-modal-host>/v1/chat/completions --api-key "$(cat ~/.uxw1_arena_token)"`.
The local path above is recommended for the video — it needs nothing but your machine.
