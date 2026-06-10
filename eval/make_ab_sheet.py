"""Build a BLINDED A/B review sheet from two sets of rewrite predictions.

Per benchmark item, shows the context + two anonymized options (A/B), randomized
deterministically per id so you can't infer the model. A separate KEY file maps A/B back
to the model. Fill in `preferred` (A / B / tie) blind, then join on id with the key.

    python3 eval/make_ab_sheet.py \
        --a-preds /tmp/poll_nt/eval_preds/base_nothink.jsonl --a-label base \
        --b-preds /tmp/poll_nt/eval_preds/1a_nothink.jsonl   --b-label 1a \
        --out reviews/base_vs_1a
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval.score_rewrite_preds import parse_rewrite  # noqa: E402
from uxft.schema import iter_jsonl  # noqa: E402

DEFAULT_BENCHMARK = (
    "/Users/christophergreer/Documents/Codex/2026-05-30/"
    "hugging-face-plugin-hugging-face-openai/data/eval/benchmark.jsonl"
)


def load_context(path: Path) -> dict[str, dict]:
    ctx = {}
    for row in iter_jsonl(path):
        user = row["messages"][1]["content"]
        current = user.split("Current copy:", 1)[1].split("\n", 1)[0].strip() if "Current copy:" in user else ""
        gold = ""
        try:
            import json
            gold = json.loads(row["messages"][2]["content"]).get("rewrite", "")
        except Exception:
            pass
        ctx[str(row["id"])] = {
            "category": row["metadata"].get("category", ""),
            "product_surface": row["metadata"].get("product_surface", ""),
            "current_copy": current,
            "gold": gold,
        }
    return ctx


def load_rewrites(path: Path) -> dict[str, str]:
    out = {}
    for r in iter_jsonl(path):
        out[str(r["id"])] = parse_rewrite(r.get("prediction", "")) or "(no valid output)"
    return out


def a_is_first(cid: str, la: str, lb: str) -> bool:
    h = hashlib.sha1(f"{cid}|{la}|{lb}".encode()).hexdigest()
    return int(h, 16) % 2 == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-preds", required=True, type=Path)
    ap.add_argument("--a-label", required=True)
    ap.add_argument("--b-preds", required=True, type=Path)
    ap.add_argument("--b-label", required=True)
    ap.add_argument("--benchmark", type=Path, default=Path(DEFAULT_BENCHMARK))
    ap.add_argument("--out", required=True, help="output prefix, e.g. reviews/base_vs_1a")
    args = ap.parse_args()

    ctx = load_context(args.benchmark)
    ra, rb = load_rewrites(args.a_preds), load_rewrites(args.b_preds)
    out_review = Path(REPO_ROOT) / f"{args.out}.review.csv"
    out_key = Path(REPO_ROOT) / f"{args.out}.key.csv"
    out_review.parent.mkdir(parents=True, exist_ok=True)

    with out_review.open("w", newline="", encoding="utf-8") as rf, \
         out_key.open("w", newline="", encoding="utf-8") as kf:
        rw = csv.writer(rf)
        kw = csv.writer(kf)
        rw.writerow(["id", "category", "product_surface", "current_copy",
                     "option_A", "option_B", "preferred (A/B/tie)", "why (optional)"])
        kw.writerow(["id", "option_A_model", "option_B_model", "gold_reference"])
        n = 0
        for cid, c in ctx.items():
            if cid not in ra or cid not in rb:
                continue
            first = a_is_first(cid, args.a_label, args.b_label)
            opt_a, opt_b = (ra[cid], rb[cid]) if first else (rb[cid], ra[cid])
            mdl_a, mdl_b = (args.a_label, args.b_label) if first else (args.b_label, args.a_label)
            rw.writerow([cid, c["category"], c["product_surface"], c["current_copy"], opt_a, opt_b, "", ""])
            kw.writerow([cid, mdl_a, mdl_b, c["gold"]])
            n += 1

    print(f"wrote {n} rows -> {out_review}")
    print(f"answer key      -> {out_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
