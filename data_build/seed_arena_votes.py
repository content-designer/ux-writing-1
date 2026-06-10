"""Seed the private arena-votes dataset with the author's blinded A/B review judgments.

These 90+90 blinded comparisons (base vs 1a, 1a vs 1b) are the highest-quality
preference data we have — an expert content designer judging anonymized pairs. They
seed `gr33r/ux-writing-arena-votes` (private) as the starting corpus for a future DPO
run, alongside live Copy Campfire votes.

    python3 data_build/seed_arena_votes.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

VOTES_DATASET = "gr33r/ux-writing-arena-votes"
REPO_ROOT = Path(__file__).resolve().parent.parent
XLSX = "/Users/christophergreer/Downloads/blind-results (1).xlsx"  # both sheets, final
SHEETS = {  # sheet name -> (comparison tag, key csv)
    "base_vs_1a.review": ("base_vs_1a", REPO_ROOT / "reviews/base_vs_1a.key.csv"),
    "1a_vs_1b.review": ("1a_vs_1b", REPO_ROOT / "reviews/1a_vs_1b.key.csv"),
}


def rows_for(sheet: str, comparison: str, key_path: Path) -> list[dict]:
    rev = pd.read_excel(XLSX, sheet_name=sheet)
    key = pd.read_csv(key_path)
    merged = rev.merge(key, on="id", how="inner")
    out = []
    for _, r in merged.iterrows():
        pref = str(r["preferred (A/B/tie)"]).strip().lower()
        if not pref or pref == "nan":
            continue
        c = pref[0]
        choice = {"a": "A", "b": "B", "t": "tie"}.get(c)
        if choice is None:
            continue
        winner = r["option_A_model"] if choice == "A" else (
            r["option_B_model"] if choice == "B" else "tie")
        note = str(r.get("why (optional)", "") or "").strip()
        out.append({
            "id": str(uuid.uuid4()),
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": "author_blind_review",
            "comparison": comparison,
            "category": r["category"],
            "surface": r.get("product_surface", ""),
            "current": r["current_copy"],
            "thinking_mode": False,  # judged outputs were direct-mode generations
            "choice": choice,
            "winner": winner,        # base | 1a | 1b | tie
            "a_is": r["option_A_model"],
            "text_a": r["option_A"],
            "text_b": r["option_B"],
            "note": note if note.lower() != "nan" else "",
            "benchmark_id": r["id"],
        })
    return out


def main() -> int:
    rows = []
    for sheet, (comparison, key_path) in SHEETS.items():
        got = rows_for(sheet, comparison, key_path)
        print(f"{sheet}: {len(got)} judged rows")
        rows.extend(got)

    api = HfApi()
    api.create_repo(VOTES_DATASET, repo_type="dataset", private=True, exist_ok=True)
    local = Path("/tmp/seed_author_blind_review.jsonl")
    with local.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    api.upload_file(path_or_fileobj=str(local),
                    path_in_repo="seed/author_blind_review.jsonl",
                    repo_id=VOTES_DATASET, repo_type="dataset")
    print(f"seeded {len(rows)} judgments -> https://huggingface.co/datasets/{VOTES_DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
