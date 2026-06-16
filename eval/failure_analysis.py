"""Reproducible failure-mode analysis of the PostHog 10K review run.

Answers the question raised in review: of the 994 suggested changes, how do the
three failure modes break down, and what does that imply at full scale?

Two layers:
  1. MEASURED on all 994 changes (no sampling): contract escapes and
     apostrophe-truncation fragments are mechanically detectable and exact-ish.
  2. EXTRAPOLATED from a hand-labeled random sample of 60 (seed 20260613), with
     Wilson 95% intervals. The sample is a uniform random draw from the 994, so
     scaling the per-mode proportions up to 994 is valid (this is the point the
     review raised — composition of the 994 IS estimable, unlike acceptance
     precision, which needs human labels we don't have).

Run:  python3 eval/failure_analysis.py
Pulls eval_preds/posthog_review.jsonl from gr33r/ux-writing-sft (needs HF auth).
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from pathlib import Path

SEED = 20260613
SAMPLE_N = 60
DATASET = "gr33r/ux-writing-sft"
REVIEW_PATH = "eval_preds/posthog_review.jsonl"
OUT_CSV = Path(__file__).parent / "failure_sample_60.csv"

# Hand labels for the 60-row sample, by sample index (reconstruction is
# deterministic given the seed + the fixed artifact). mode ∈ {M1,M2,M3,CLEAN}:
#   M1 scanner garbage-in / truncation, M2 non-UI string treated as copy,
#   M3 meaning drift or contract escape, CLEAN legit substantive edit.
MODE = {
    0: "CLEAN", 1: "CLEAN", 2: "M2", 3: "M1", 4: "CLEAN", 5: "M1", 6: "CLEAN",
    7: "M2", 8: "CLEAN", 9: "M1", 10: "M1", 11: "M3", 12: "CLEAN", 13: "M1",
    14: "CLEAN", 15: "CLEAN", 16: "M1", 17: "CLEAN", 18: "CLEAN", 19: "CLEAN",
    20: "CLEAN", 21: "M2", 22: "M3", 23: "M2", 24: "CLEAN", 25: "CLEAN",
    26: "CLEAN", 27: "CLEAN", 28: "M3", 29: "CLEAN", 30: "CLEAN", 31: "M2",
    32: "CLEAN", 33: "CLEAN", 34: "CLEAN", 35: "CLEAN", 36: "CLEAN", 37: "CLEAN",
    38: "CLEAN", 39: "CLEAN", 40: "CLEAN", 41: "M1", 42: "M2", 43: "M2",
    44: "CLEAN", 45: "M3", 46: "CLEAN", 47: "M1", 48: "CLEAN", 49: "CLEAN",
    50: "M3", 51: "M1", 52: "CLEAN", 53: "CLEAN", 54: "M1", 55: "M1",
    56: "CLEAN", 57: "CLEAN", 58: "CLEAN", 59: "CLEAN",
}
M1_HARMFUL = {10}            # fabricated WRONG copy (rest of M1 = benign reconstruction)
M2_BREAKS = {2, 7, 21, 23, 43}   # would break code/analytics if applied
M3_ESCAPE = {11, 28}             # contract escape (emit JSX/code); rest = meaning drift

# Five examples the article cites by file:line — used to verify the sample matches.
ANCHORS = ["MoveToPostHogCloud.tsx:66", "utils.ts:71", "llmPromptsLogicType.ts:43",
           "WhereStep.tsx:254", "NewCategoryModal.tsx:46"]


def norm(s: str) -> str:
    return " ".join((s or "").split())


def is_escape(s: str) -> bool:
    s = s or ""
    return bool(
        re.search(r"\{[^}]*\?[^}]*:[^}]*\}", s) or re.search(r"===|!==|=>", s)
        or re.search(r"</?[a-zA-Z][^>]*>", s)
        or (s.strip().startswith("{") and s.strip().endswith("}"))
    )


def is_apostrophe_trunc(r: dict) -> bool:
    # current_copy was cut mid-literal at a contraction/possessive: the source
    # continues with '<letter> right where the scanned string ends.
    return bool(re.search(re.escape(norm(r["current_copy"])) + r"'[a-zA-Z]", r.get("context", "")))


def wilson(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def disposition(i: int) -> str:
    m = MODE[i]
    if m == "CLEAN":
        return "clean"
    if m == "M1":
        return "error_fabrication" if i in M1_HARMFUL else "artifact_benign"
    if m == "M2":
        return "error_non_ui" + ("_breaks_code" if i in M2_BREAKS else "")
    return "error_contract_escape" if i in M3_ESCAPE else "error_meaning_drift"


def main() -> int:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, REVIEW_PATH, repo_type="dataset")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = [r for r in rows if r.get("changed")]
    n = len(changed)
    assert n == 994, f"expected 994 changed, got {n}"

    # ---- layer 1: measured on all 994 ----
    esc = [r for r in changed if is_escape(r["suggested_copy"])]
    apo = [r for r in changed if is_apostrophe_trunc(r)]
    print(f"changed = {n}")
    print("MEASURED on all 994:")
    print(f"  contract escapes (code as 'copy')   {len(esc):3} = {len(esc)/n*100:.1f}%")
    print(f"  apostrophe-truncation fragments     {len(apo):3} = {len(apo)/n*100:.1f}%")

    # ---- layer 2: hand-labeled random sample of 60, scaled with CIs ----
    random.seed(SEED)
    sample = random.sample(changed, SAMPLE_N)
    tags = [f"{r['path'].split('/')[-1]}:{r['line']}" for r in sample]
    for a in ANCHORS:
        assert a in tags, f"anchor {a} missing — sample reconstruction drifted"

    from collections import Counter
    disp = Counter(disposition(i) for i in range(SAMPLE_N))
    mode = Counter(MODE[i] for i in range(SAMPLE_N))
    assert mode["M1"] == 11 and mode["M2"] == 7 and mode["M3"] == 5 and mode["CLEAN"] == 37

    errors = sum(v for k, v in disp.items() if k.startswith("error"))
    benign = disp["artifact_benign"]
    clean = disp["clean"]
    print(f"\nSAMPLE n=60 -> scaled to {n} (Wilson 95%):")
    for name, x in [("clean good edits", clean), ("benign scanner artifacts", benign),
                    ("genuine errors (reviewer-reject)", errors),
                    ("  of which fabricated wrong copy", disp['error_fabrication'])]:
        lo, hi = wilson(x, SAMPLE_N)
        print(f"  {name:34} {x:2}/60 -> ~{round(x/SAMPLE_N*n):3}  ({round(lo*n)}-{round(hi*n)})")

    # ---- write the labeled sample for audit ----
    cols = ["idx", "mode", "disposition", "path", "line", "kind",
            "current_copy", "suggested_copy", "reason", "risk"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(sample):
            w.writerow([i, MODE[i], disposition(i), r["path"], r["line"], r["kind"],
                        r["current_copy"], r["suggested_copy"], r.get("reason", ""), r.get("risk", "")])
    print(f"\nwrote {OUT_CSV.relative_to(Path.cwd())} ({SAMPLE_N} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
