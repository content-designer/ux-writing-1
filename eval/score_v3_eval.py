#!/usr/bin/env python3
"""Score base/v4/v1.1 predictions on the v3 validation set and build a blinded review CSV.

Downloads the prediction files from the eval datasets, scores each prediction file
against data/eval/benchmark.v3.jsonl using uxft.eval.score_rewrite (the repo's
source-of-truth heuristic metrics), prints overall + per-category aggregates, and
writes a randomized A/B (v4 vs v1.1) review CSV + answer key for human judgment.

Usage:
    python3 scripts/score_v3_eval.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from uxft.eval import score_rewrite
from uxft.dataset import input_key

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data/eval/benchmark.v3.jsonl"
NEW_CATEGORIES = {"tooltip", "error_page", "interstitial", "settings_toggle"}

# (label, eval dataset repo, filename) — base is identical across runs; take it from the v1.1 run.
SOURCES = [
    ("base", "gr33r/ux-writing-1.1-eval", "predictions/base_predictions.jsonl"),
    ("v4", "gr33r/ux-writing-v4-eval-v3", "predictions/adapter_predictions.jsonl"),
    ("v1.1", "gr33r/ux-writing-1.1-eval", "predictions/adapter_predictions.jsonl"),
]


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_bench() -> dict[str, dict]:
    bench = {}
    for row in load_jsonl(BENCH):
        user = row["messages"][1]["content"]
        current = user.split("Current copy:", 1)[1].split("\n", 1)[0].strip() if "Current copy:" in user else ""
        bench[row["id"]] = {
            "category": row["metadata"]["category"],
            "current": current,
            "reference": json.loads(row["messages"][2]["content"])["rewrite"],
        }
    return bench


def fetch_preds(repo: str, filename: str) -> dict[str, dict]:
    local = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
    return {p["id"]: p for p in load_jsonl(local)}


def score_model(bench: dict, preds: dict) -> tuple[dict, dict]:
    """Return (overall_avg_metrics, per_category_overall_score)."""
    agg: dict[str, float] = defaultdict(float)
    by_cat: dict[str, list[float]] = defaultdict(list)
    n = 0
    for rid, b in bench.items():
        p = preds.get(rid)
        rewrite = (p or {}).get("rewrite", "").strip()
        if not rewrite:
            rewrite = ""  # empty prediction scores poorly, as it should
        m = score_rewrite(b["current"], rewrite, b["category"])
        row_overall = sum(m.values()) / len(m)
        for k, v in m.items():
            agg[k] += v
        by_cat[b["category"]].append(row_overall)
        n += 1
    overall = {k: round(v / n, 4) for k, v in agg.items()}
    overall["overall"] = round(sum(overall.values()) / len(overall), 4)
    per_cat = {c: round(sum(s) / len(s), 4) for c, s in by_cat.items()}
    return overall, per_cat


def main() -> int:
    bench = load_bench()
    models = {label: fetch_preds(repo, fn) for label, repo, fn in SOURCES}

    print("=== Overall heuristic metrics (v3 validation, %d rows) ===" % len(bench))
    overalls, per_cats = {}, {}
    for label in ("base", "v4", "v1.1"):
        o, pc = score_model(bench, models[label])
        overalls[label], per_cats[label] = o, pc
        print(f"\n[{label}] {json.dumps(o)}")

    print("\n=== Per-category 'overall' score: v4 vs v1.1 (Δ = v1.1 − v4) ===")
    cats = sorted({b["category"] for b in bench.values()},
                  key=lambda c: (c not in NEW_CATEGORIES, c))
    print(f"{'category':<26}{'n':>4}{'base':>9}{'v4':>9}{'v1.1':>9}{'Δ':>9}  new?")
    for c in cats:
        n = sum(1 for b in bench.values() if b["category"] == c)
        bv = per_cats["base"].get(c, 0.0)
        v4 = per_cats["v4"].get(c, 0.0)
        v11 = per_cats["v1.1"].get(c, 0.0)
        star = "  *NEW*" if c in NEW_CATEGORIES else ""
        print(f"{c:<26}{n:>4}{bv:>9.3f}{v4:>9.3f}{v11:>9.3f}{v11 - v4:>+9.3f}{star}")

    # Blinded A/B review CSV (v4 vs v1.1), randomized, with answer key.
    rng = random.Random(13)
    review_rows, key_rows = [], []
    for rid, b in bench.items():
        a_label, b_label = ("v4", "v1.1") if rng.random() < 0.5 else ("v1.1", "v4")
        a_text = models[a_label].get(rid, {}).get("rewrite", "")
        b_text = models[b_label].get(rid, {}).get("rewrite", "")
        review_rows.append({
            "id": rid, "category": b["category"], "is_new_category": b["category"] in NEW_CATEGORIES,
            "current_copy": b["current"], "option_A": a_text, "option_B": b_text,
            "winner_A_B_tie": "",
        })
        key_rows.append({"id": rid, "option_A_is": a_label, "option_B_is": b_label})

    review_path = ROOT / "data/eval/human_review_v3.csv"
    key_path = ROOT / "data/eval/human_review_v3_key.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys()))
        w.writeheader(); w.writerows(review_rows)
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)
    print(f"\nWrote blinded review: {review_path}  (key: {key_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
