# Evaluation results — UX-Writing Fine-Tune v2 (Qwen3.6-27B)

Base model: `Qwen/Qwen3.6-27B` (dense ~27.8B, hybrid Gated-DeltaNet + full-attention, ~248K vocab, reasoning model).
Adapters (QLoRA, private): `gr33r/ux-writing-2.0-rewrite-qwen36` (1a, text), `gr33r/ux-writing-2.0-combined-qwen36` (1b, text+vision).
Held-out rewrite benchmark: 90 hand-authored gold items (`benchmark.jsonl`), zero training overlap.

## 1. Heuristic scores (rewrite, direct mode `enable_thinking=False`, 256 tok, 0 JSON failures)

| Model | not_generic | concise | specific_action | no_blame | changed_when_needed | overall |
|---|---|---|---|---|---|---|
| base (fair) | 0.989 | 1.000 | 0.594 | 1.000 | 1.000 | **0.917** |
| 1a (text) | 1.000 | 0.980 | 0.661 | 1.000 | 1.000 | **0.928** |
| 1b (combined) | 1.000 | 0.984 | 0.661 | 1.000 | 1.000 | **0.929** |

**Caveat:** these 5 heuristics saturate near 1.0 for any competent model and cannot distinguish the three. They are a weak proxy; the blinded human review below is the real measure. (Note: in the model's *default* thinking mode, base produces 0% parseable JSON within a 256-tok budget while 1a produces 93% — the fine-tune bakes in reliable, concise structured output without needing the `enable_thinking=False` flag.)

## 2. Blinded human A/B — base vs 1a (90 items, reviewer judged options anonymized)

- **1a preferred: 65 · base preferred: 13 · no-preference: 12** (all 12 no-pref were identical text (11) or both-bad (1)).
- **Decisive win rate: 1a 65/78 = 83.3%** vs base 16.7%.
- 1a wins in every category; strongest in inline_error (9–0), destructive_confirmation (7–0), accessibility_label (6–0), button (10–1), system_error (11–3).
- Clears the project bar (≥30% more accepted rewrites than prompt-only baseline) by a wide margin.

**Takeaway:** heuristics said base≈1a; blinded human judgment says 1a wins ~83%. The fine-tune's value is real and large on human-judged quality — invisible to the crude heuristics.

## 3. 1b vs 1a — combined did NOT meaningfully regress text
- Heuristic overall: 0.929 (1b) vs 0.928 (1a) — indistinguishable.
- **Blinded human A/B (90 items): 1a 30 · 1b 21 · tie 39.** Of 51 decisive: 1a 59% / 1b 41% — a mild, within-noise lean to the text specialist (n=51).
- Category split is mixed (1a better on destructive_confirmation 5–1, notification 5–1; 1b better on onboarding 2–4, inline_error 2–4, empty_state 3–4).
- **Conclusion:** combining text + vision held text quality at par with the text-only specialist (43% ties + a small noisy edge) while adding the consistency capability 1a lacks. Original consolidation hypothesis validated — one multimodal model, minimal text tax.

**Deployment recommendation:** bulk code/text scanning → **1a** (marginally better on text, smaller/cheaper); screenshot review or a single unified model → **1b** (adds vision at negligible text cost).

## 4. Scenario-improvement opportunities (from reviewer notes)
Feedback to improve the **benchmark/training scenarios** in the next data rev:
- **Trivial duplicates (low signal):** ~11 items where base and 1a produced identical text (e.g. "Go", "ZIP code", "Preferred pronouns", "Get data"). Replace with harder, more discriminative cases.
- **Ambiguous content-type/context:** several items don't make clear whether the string is alt text, UI copy, or review feedback ("Menu" → "Assuming this is alt text"; "Too much text." → "review feedback not UI copy"). Add explicit `content_type`/surface context.
- **Flawed gold/scenario logic:** form label phrased as a question ("Score" → "Form labels shouldn't be questions"); out-of-stock error suggesting "choose another quantity" ("wouldn't make sense if totally out of stock"). Fix the expected behavior.
- **Hallucination on vague inputs:** models invent specifics for under-specified inputs ("Image" → "Hallucinated details, but better as alt text"; "Max" → "reasonable but possibly inaccurate jump to 'seats'"). Add anti-hallucination signal / more context.
- **Tone nits to encode in training:** unnecessary "please" ("Consent required."); over-harsh consequence ("Cancel now?" → "'Lost it' is a bit harsh"). Refine voice in the rewrite data.

## 5. Pending
- **Vision consistency eval** for 1b on real screenshots (Cal.com / Ghost / Wealthsimple + 25 newly-annotated refs) — requires hand-labeling the 25 references (gold scaffold staged at `/tmp/ux-writing-neweval/`).
- Scenario-improvement opportunities (§4) → fold into a future dataset rev (no retrain planned now).
