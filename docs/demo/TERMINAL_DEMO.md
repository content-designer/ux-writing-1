# Terminal demo cheatsheet — running ux-writing-1 on camera

## Which path? (disk-aware)

The local-GGUF demo needs ~20 GB free for comfort (model is 16.5 GB). If `df -h /`
shows less than that, use the **hosted endpoint** path — same terminal visuals, nothing
to download.

| You have… | Use | Story |
|---|---|---|
| < ~20 GB free | **Path B — hosted endpoint** | identical demo, zero local footprint |
| ≥ ~20 GB free | **Path A — local GGUF** | strongest: "running on my laptop, $0 marginal" |
| no time to set up | **skip it** | Copy Campfire IS the live model; graphics carry the rest |

---

## Path B — hosted endpoint (no download)

`uxw1-serve` is deployed on Modal (scales to zero; first request cold-starts ~30–60s).
`uxft.review_repo` talks to it directly — identical to Path A's Beat 2, just remote.

**Get the URL once (it's not in the repo — it's a Space secret):** Hugging Face → your
`copy-campfire` Space → Settings → Variables & secrets → copy `BATTLE_URL`. The review
endpoint is that URL with `-battle` swapped to `-chat`
(e.g. `https://<host>--uxw1-serve-uxwriting1-chat.modal.run`). The Modal dashboard
(modal.com → uxw1-serve → web endpoints) lists both too.

**Confirm it before recording** (your token, your endpoint — safe to run yourself):

```bash
curl -s -X POST "https://<host>--uxw1-serve-uxwriting1-chat.modal.run" \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Current copy: Invalid\"}],\"_auth\":\"$(tr -d '[:space:]' < ~/.uxw1_arena_token)\"}"
# expect: {"choices":[{"message":{"content":"{\"rewrite\": \"Invalid email address\", ...}"}}]}
```

**The on-camera beat** (scan local repo → review through the hosted model → JSONL):

```bash
python -m uxft.scan ~/code/some-oss-app --limit 60 --out /tmp/scan.jsonl
python -m uxft.review_repo ~/code/some-oss-app \
  --endpoint "https://<host>--uxw1-serve-uxwriting1-chat.modal.run" \
  --api-key "$(tr -d '[:space:]' < ~/.uxw1_arena_token)" \
  --limit 40 --out /tmp/review.jsonl
jq -c 'select(.suggested_copy != "" and .suggested_copy != .current_copy)' /tmp/review.jsonl
```

Prints `wrote N review rows (M suggested changes)` — the restraint stat, live. This is
the unbatched path (~$2/1K, the lazy one); the cheap $0.32/1K number is the batched A100.

---

## Path A — local GGUF (needs ~20 GB free)

**Pull before you record** — the 16 GB download is not a camera moment.

```bash
ollama pull hf.co/gr33r/ux-writing-1-GGUF:Q4_K_M   # ~16.5 GB, one time, off-camera
```

---

## Path A, Beat 1 — one string, live (punchy, ~15s)

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

## Path A, Beat 2 — review a whole repo, fully local (the money shot, ~40s)

Ollama exposes an OpenAI-compatible endpoint at `localhost:11434`, and `uxft.review_repo`
speaks it — so the entire scan → review pipeline runs on your laptop, no cloud at all.
Point it at any repo you have checked out:

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
