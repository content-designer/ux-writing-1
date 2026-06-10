# Video outline — Build Small submission: ux-writing-1 + Copy Campfire

*Structure + verified numbers + demo beats. You write the script — it's your craft.
Target 2–3 minutes. Numbers marked ⏳ get filled from COST_NOTES.md / the demo artifact
once those runs land; everything else is final.*

## Suggested beats

**1. The hook (≈15s) — make it personal**
> "I blind-tested an open 27-billion-parameter model against a version of itself I
> fine-tuned on UX writing. Options anonymized, sides shuffled, 90 rounds. I picked my
> fine-tune **83% of the time** — and I couldn't see which was which."

On screen: the blinded review spreadsheet, then the reveal moment.

**2. The problem (≈20s) — speak to content designers**
- Every product ships copy debt: "Invalid", "OK", "An error occurred while processing your request."
- It lives in code, at codebase scale — thousands of strings nobody owns.
- Frontier-API review works but: cost at scale, and your unshipped product copy leaves
  your infrastructure.

**3. The Build Small story (≈20s)**
- One hackathon credit. Base: Qwen3.6-27B (Apache-2.0, fits-on-a-laptop class).
- QLoRA on a hand-built dataset of ≈1,400 derived rewrite pairs; **≈$30 of training**, two runs.
- Evaluated the honest way: held-out gold set + blinded human review (the 83%).

**4. Live demo 1 — ⛺ Copy Campfire (≈40s)**
- Paste a real string ("Invalid" on an email field). Two campers answer; vote; reveal.
- Flip on 🔦 lantern mode: *watch the token counters* — base reasons at length, the
  fine-tune answers brisk and tight (in our first test: 66 tokens vs 31 for the same
  quality answer). "That difference is your serving bill."
- "Every vote is preference data — the campfire literally trains v2."

**5. Live demo 2 — scan a real codebase (≈40s)**
- `python -m uxft.scan` on Cal.com (open-source scheduling app): 200 strings extracted
  in seconds.
- `python -m uxft.review_repo` → diff-friendly JSONL of suggestions; show the highlights
  table (docs/demo/) — and the restraint stat: **187 of 200 strings kept as-is**.
- The honest cost line (measured, COST_NOTES.md): "**≈8,000 strings an hour** on one
  rented A100 — about **31 cents per thousand strings** — private, on hardware you
  control, with **100% valid JSON** output."

**6. Take it home + CTA (≈25s)**
- Apache-2.0: merged model, LoRA adapter, GGUF for your laptop (LM Studio/Ollama).
- "It learns *your* style guide in an afternoon: ≈100 before/after pairs, one HF Jobs
  command, ≈$5. The repo even ships the blinded-review tooling so you can *prove* your
  version is better." (FINETUNE_GUIDE.md)
- CTA: **"Come vote at the Copy Campfire. Bring your worst error message."**

## Verified numbers you can claim (sources in repo)

| Claim | Number | Source |
|---|---|---|
| Blinded human preference vs base | **83% (65/78 decisive, 90 items)** | docs/EVAL_RESULTS.md §2 |
| Wins by category | errors 9–0, destructive 7–0, a11y 6–0 | EVAL_RESULTS.md §2 |
| Heuristic scores (honesty caveat) | 0.928 vs 0.917 — saturated; human review is the measure | EVAL_RESULTS.md §1 |
| JSON contract reliability (direct mode) | 90/90 valid for fine-tune AND base | EVAL_RESULTS.md §1 |
| Token efficiency (first live battle) | fine-tune 31 tokens vs base 66 | serve test, this repo |
| Training cost | ≈$30 of a $250 Modal credit | Modal dashboard |
| Throughput (batched, A100) | **7,951 strings/hour, $0.31/1K, 100% valid JSON** | docs/COST_NOTES.md (measured) |
| Unbatched endpoint run | 200 Cal.com strings in 9.9 min (≈$2/1K) | docs/COST_NOTES.md (measured) |
| vs Opus list price | ≈8–15× cheaper batched (estimate, labeled) | docs/COST_NOTES.md |
| Cal.com demo | 200 strings → 13 changes, 187 kept (restraint) | docs/demo/ |

**Don't claim:** a specific multiple vs Opus pricing (depends on batching + their list
price changes); anything about screenshot/vision review (unreleased).

## Social post draft (submission requirement)

> I fine-tuned a 27B open model on UX writing for ≈$30 — then blind-tested it against
> its own base model. I picked the fine-tune 83% of the time.
>
> ⛺ Copy Campfire: paste your worst error message, judge the blind rewrite battle, and
> your vote trains v2: [space link]
>
> Model, data recipe, scan-your-codebase CLI, and a guide to tuning it on YOUR style
> guide — all open: [model link] #BuildSmall @huggingface

## B-roll / screen-capture checklist
- [ ] Blinded spreadsheet scroll + the unblind tally moment
- [ ] Campfire battle: paste → two cards → vote → reveal
- [ ] Lantern mode token counters side by side
- [ ] Terminal: scan + review commands on Cal.com, JSONL scrolling
- [ ] Model card + GGUF page in LM Studio/Ollama
- [ ] Trail log tab showing community votes accumulating
