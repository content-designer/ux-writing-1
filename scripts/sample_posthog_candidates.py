"""Filter + sample PostHog scan output for the cost-demo review run.

Reproduces docs/demo/posthog_scan_meta.json exactly:

    python3 -m uxft.scan /tmp/posthog/frontend --out /tmp/ph_frontend_all.jsonl
    python3 -m uxft.scan /tmp/posthog/products --out /tmp/ph_products_all.jsonl
    python3 scripts/sample_posthog_candidates.py \
        --frontend /tmp/ph_frontend_all.jsonl --products /tmp/ph_products_all.jsonl \
        --out /tmp/posthog_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

SEED = 20260612
SAMPLE_SIZE = 10_000

NON_UI_PATH = re.compile(
    r"(\.test\.|\.spec\.|\.stories\.|__mocks__|/test/|/tests/|cypress|/e2e/|\.cy\."
    r"|mocks?\.tsx?$|fixtures|\.schemas?\.ts$|generated|schema\.json$)",
    re.I,
)
# css-ish token: lowercase AND contains a structural char (hyphen/digit/colon/bracket/slash/dot)
CSS_TOKEN = re.compile(r"^(?=.*[-0-9:\[\]/.%#])[a-z0-9:\[\]/.%#-]+$")


def looks_like_ui_copy(value: str) -> bool:
    if "_" in value:
        return False
    words = value.split()
    if len(words) >= 2:
        css_ish = sum(1 for w in words if CSS_TOKEN.fullmatch(w))
        return css_ish / len(words) < 0.5
    return bool(re.fullmatch(r"[A-Z][a-z]{2,}", words[0]))


def load(path: Path, prefix: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["path"] = prefix + row["path"]
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", type=Path, required=True)
    ap.add_argument("--products", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.frontend, "frontend/") + load(args.products, "products/")
    eligible = [
        r for r in rows
        if not NON_UI_PATH.search(r["path"]) and looks_like_ui_copy(r["current_copy"])
    ]
    seen: set[tuple] = set()
    deduped = []
    for r in eligible:
        key = (r["path"], r["line"], r["current_copy"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    random.seed(SEED)
    sample = random.sample(deduped, SAMPLE_SIZE)
    with args.out.open("w", encoding="utf-8") as handle:
        for r in sample:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"raw={len(rows)} eligible={len(deduped)} sampled={len(sample)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
