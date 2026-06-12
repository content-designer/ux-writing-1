# The most useful thing my fine-tune learned was restraint

*Draft for a Hugging Face community article. Image slots are marked `[figure: …]` —
screenshot the matching page from the [campfire gallery](https://copy-campfire-gallery.vercel.app)
at 1920×1080. All numbers trace to artifacts in the
[repo](https://github.com/content-designer/ux-writing-1).*

---

June 2026 is the month the token bill became a board-level topic.
[Uber burned through its entire 2026 AI budget in four months](https://fortune.com/2026/05/26/uber-coo-ai-spending-tokens-claude-code/)
and now [caps engineers at $1,500 a month per coding tool](https://techcrunch.com/2026/06/02/uber-caps-employee-ai-spending-after-blowing-through-budget-in-four-months/),
with its COO openly questioning the ROI.
[Microsoft's internal numbers showed per-engineer AI costs of $500–$2,000 a month](https://fortune.com/2026/05/22/microsoft-ai-cost-problem-tokens-agents/),
and its [move of GitHub Copilot to usage-based token billing](https://dataconomy.com/2026/06/01/github-copilot-token-pricing-backlash/)
gave thousands of developers their first meter shock.
[Goldman projects a 24× increase in token consumption by 2030](https://www.eenewseurope.com/en/ai-token-costs-uber-microsoft-goldman/).

Most of that pain comes from agentic coding — long contexts, many turns, open-ended
work, where frontier models genuinely earn their price. But a lot of the LLM work inside
a company isn't that shape at all. It's bulk, repetitive, and structured: thousands of
small, independent items that each need a few hundred tokens of judgment. That work is
quietly being metered at frontier prices too.

For the [Build Small hackathon](https://huggingface.co/build-small-hackathon) I picked
one job with exactly that shape — reviewing UI copy — and fine-tuned a small open model
to do it. The headline isn't the cost (though we'll get to a fun number). It's this:
pointed at 10,000 of PostHog's UI strings, the model changed 994 and **left 9,006
alone**. The most useful thing it learned was restraint.

## Why UI copy is the perfect small-model job

Every product ships copy debt: "Invalid", "OK", "An error occurred while processing your
request." It lives in code, at codebase scale — thousands of strings nobody owns. And it
has three properties that make it unusually well suited to a small, tuned model:

1. **Each item is tiny and independent.** A string plus ±3 lines of surrounding code is
   all the context a reviewer needs — ~360 prompt tokens per item in my runs. No agentic
   loop, no long context, no multi-turn reasoning.
2. **The craft has teachable standards.** Purposeful, concise, conversational, clear;
   never weaken safety-critical copy; keep `{{ variables }}` intact. You can write these
   down — which means you can train on them.
3. **The volume is huge and the stakes per item are low.** A wrong suggestion costs a
   reviewer two seconds of "reject." That's the right risk profile for automation.

Does this transfer to docs, code comments, error logs? The *recipe* should — anything
string-shaped with local context fits the same mold. This particular model is tuned for
interface copy; the repo ships the pipeline so you can re-aim it.

## The data is the product

The model is [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (Apache-2.0) with a
QLoRA adapter — two training runs, ≈$30 of GPU time. The interesting part is the
~1,400-pair dataset, which took far longer than the training:

- **Course-derived pairs.** I authored a UX-writing course; its exercises became
  before/after pairs with *reasons* — not just "this is better" but why: front-load the
  action, name the consequence, match tone to stakes.
- **Real strings in real code.** Pairs from permissively-licensed open-source UIs, each
  carrying the surrounding code as context, so the model learns to read a string *in
  situ* — is this a button, a tooltip, an aria-label?
- **A JSON contract.** Every answer is `{rewrite, reason, risk}` — review output a human
  can scan, accept, or reject. Never auto-applied.
- **Keeps as a first-class answer.** A large share of training rows teach "this copy is
  already right: return it verbatim and say so." This is the restraint training, and it
  was the most deliberate decision in the dataset. A review model that rewrites
  everything is worse than useless — it buries the three real problems under two hundred
  preference edits. Restraint has to be trained; it does not emerge from "be helpful."

One gotcha worth your time if you're doing this: Qwen3.6 is a *thinking* model, and with
thinking on, the base model produced **0% valid JSON** at 256 tokens in batch eval.
Production runs use `enable_thinking=False`. Decide where thinking goes *before* you
train for structured output.

## What the training actually changed

Before trusting any benchmark, it's worth looking at what fine-tuning physically did.
The adapter is public, so this is measurable: for every LoRA pair (A, B) I computed the
Frobenius norm of the delta ‖B·A‖ per (layer, module) — 256 deltas across 64 layers.
(Cheap trick: ‖BA‖²_F = Σ((BᵀB)∘(AAᵀ)), so you never materialize the full matrix.)

[figure: weights_heatmap — the LoRA fingerprint]

Two things jump out:

- **MLP modules adapted on all 64 layers; attention only exists on the 16 full-attention
  blocks** — Qwen3.6's hybrid architecture, visible in one picture.
- **The biggest deltas concentrate late in the stack**: `gate_proj` on layers 56–63
  burns brightest. Style lives near the surface; the layers that store what the model
  *knows* barely moved.

That's the fingerprint you'd hope for from a style-and-judgment fine-tune — and it's why
the next section's eval focuses on writing quality, not knowledge. There's an
[interactive version](https://copy-campfire-gallery.vercel.app/weights_heatmap.html)
where hovering any cell explains in plain language what that weight does; reproduce it
from the released adapter with
[one script](https://github.com/content-designer/ux-writing-1/blob/main/scripts/adapter_heatmap.py).

## The eval that didn't work, and the one that did

My first eval was automated heuristics — length, clarity markers, terminology checks.
The base model scored 0.917. The fine-tune scored 0.928. Saturated, useless for telling
them apart, and exactly the kind of number that looks great in a README.

So the eval I actually trust is a blind one: 90 held-out items, both models' outputs
anonymized and shuffled, judged before unblinding. The fine-tune won **65 of 78
decisive comparisons (83%)** — including 9–0 on error messages, 7–0 on
destructive-action copy, and 6–0 on accessibility labels. The repo ships the blinding
tooling, so when you tune the model on your own style guide you can run the same test
instead of eyeballing cherry-picked outputs.

(The 83% is against the model's own base. I have no eval comparing it to frontier
models, and this article makes no quality claim about them.)

## PostHog: 10,000 strings, one GPU, 77 minutes

A demo on twenty curated strings proves nothing, so the showcase run is
[PostHog](https://github.com/PostHog/posthog) (MIT) at a pinned commit: 152,713 raw
strings scanned from `frontend/` + `products/`, 26,061 after filters that drop tests,
stories, identifiers, and Tailwind classes, and a seeded random 10,000 reviewed on one
rented A100-80GB.

| | |
|---|---|
| Wall-clock | **77.2 minutes**, including model load |
| Measured cost | **$3.22** — $0.32 per 1,000 strings (A100 @ $2.50/h list) |
| Tokens | 3,590,383 prompt + 313,293 completion |
| JSON contract | 9,999 / 10,000 valid |
| Verdicts | **994 changed · 9,006 kept as-is** |

The suggestions read like a colleague's review comments, file and line attached:
`Invalid` → `Invalid API key`. `Done` → `Save changes`. `must be string` → `Enter a
single line of text`. `Lucky you!` → `You're on the YC plan`.
([Interactive cards](https://copy-campfire-gallery.vercel.app/before_after.html) — every
one an unedited artifact row.)

[figure: before_after — rewrite cards]

And the same workload, priced three other ways using the measured token counts and
public list prices (pulled 2026-06-12):

| same tokens | bill | per 1K strings |
|---|---|---|
| **ux-writing-1, rented A100 (measured)** | **$3.22** | $0.32 |
| Qwen3.6-27B via DeepInfra (estimate) | $2.15 | $0.21 |
| Claude Opus 4.8 (estimate) | $25.78 | $2.58 |
| GPT-5.5 (estimate) | $27.35 | $2.73 |

[figure: cost_compare — the bill ladder]

Assumptions stated once: the frontier rows are list-price estimates on *my* token
counts, not measured runs; tokenizers differ by roughly ±15%; reasoning-mode APIs bill
hidden thinking tokens as output, which would raise their rows. Note the DeepInfra line
undercutting my GPU: the point was never "my GPU is magic." It's that this *workload
shape* prices like a commodity — ~8× under frontier list for the identical tokens — and
at commodity prices you can re-run the review on every PR instead of rationing it. With
[Uber capping seats](https://techcrunch.com/2026/06/02/uber-caps-employee-ai-spending-after-blowing-through-budget-in-four-months/)
and [Copilot now metered](https://dataconomy.com/2026/06/01/github-copilot-token-pricing-backlash/),
"which jobs actually need frontier tokens?" is the question every platform team is being
asked. Bulk structured review is one with a clean answer. It also runs on hardware you
control, so unshipped product copy never leaves your infrastructure.

## Where it goes wrong

Wins are easy to screenshot, so here is the other side, from a seeded random sample of
60 of the 994 suggested changes (seed `20260613`, reproducible from the run artifact).
Three failure modes show up:

1. **Garbage in, confident rewrite out.** My scanner truncates strings at escaped
   apostrophes, and the model happily "fixes" the fragment by inventing plausible copy
   from the surrounding code — `Don` became `Automatic backups — no setup required.`
   (`MoveToPostHogCloud.tsx:66`). Reasonable-looking, but it's rewriting a string that
   doesn't exist as scanned. The single JSON-contract failure in the whole run was the
   same class of problem: the scanner fed it a `package.json` build command ending in a
   backslash, and the escaping broke.
2. **Treating non-UI strings as copy.** A color constant `Green` → `OliveDrab`
   (`utils.ts:71`); an internal logging key `load prompts success (…)` → `Prompts loaded`
   (`llmPromptsLogicType.ts:43`) — a "fix" that would silently break analytics if anyone
   applied it. The model's restraint is about *copy quality*; it doesn't yet know that
   some strings aren't copy at all. That's a scanner-precision problem as much as a
   model problem, and it's where the next filtering work goes.
3. **Occasional meaning drift and contract escapes.** `Survey can appear anywhere on
   your site` → `Show the survey on every page` shifts the meaning
   (`WhereStep.tsx:254`); once in a while it proposes JSX instead of copy
   (`Create` → `{category ? 'Update' : 'Create'}`, `NewCategoryModal.tsx:46`).

What I can't tell you is a precision number — of the 994 changes, how many would a
senior UX writer accept? That needs human labels at scale, which neither I nor the
blinded 90-item eval can provide for a 10,000-string run. Which is exactly why the demo
isn't a static report.

## The arena is the missing measurement

⛺ [**Copy Campfire**](https://huggingface.co/spaces/build-small-hackathon/copy-campfire)
is the live demo: paste your worst error message, two anonymous campers rewrite it, you
vote, then the reveal. Votes are blinded — reasons and metadata stay hidden until after
you choose, because response length alone fingerprints the base model. Every vote is a
human preference label, and those labels are both the precision measurement this article
is missing *and* the DPO training data for v2. The demo is the data flywheel.

## Take it home

- **Model / adapter / GGUF:** [gr33r/ux-writing-1](https://huggingface.co/gr33r/ux-writing-1) ·
  [gr33r/ux-writing-1-lora](https://huggingface.co/gr33r/ux-writing-1-lora) ·
  [gr33r/ux-writing-1-GGUF](https://huggingface.co/gr33r/ux-writing-1-GGUF) — the Q4_K_M
  runs on a 24 GB laptop in LM Studio or Ollama, where the marginal cost of a review is
  zero.
- **Scan your own repo:** `python -m uxft.scan` + `python -m uxft.review_repo` —
  [repo](https://github.com/content-designer/ux-writing-1).
- **Tune it on your style guide:** ~100 before/after pairs, one job, ≈$5 —
  [FINETUNE_GUIDE](https://github.com/content-designer/ux-writing-1/blob/main/docs/FINETUNE_GUIDE.md),
  blinding tooling included, so you can prove the result instead of vibing it.

Small model, hand-built dataset, restraint as a feature. Come vote at the campfire —
bring your worst error message.
