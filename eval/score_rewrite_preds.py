"""Score eval_rewrite predictions with the repo heuristics — HONESTLY.

Unlike uxft.eval.score (which falls back to the GOLD rewrite when a prediction is
missing/empty, inflating the score), this penalizes malformed/empty model output: a
prediction whose raw text isn't valid JSON with a non-empty `rewrite` scores 0 on every
metric. `current` copy and `category` are read from the gold benchmark, joined by id.

    python3 eval/score_rewrite_preds.py --preds base_preds.jsonl --label base
    python3 eval/score_rewrite_preds.py --preds text_preds.jsonl --label 1a-text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from uxft.eval import score_rewrite  # noqa: E402
from uxft.schema import iter_jsonl  # noqa: E402

DEFAULT_BENCHMARK = (
    "/Users/christophergreer/Documents/Codex/2026-05-30/"
    "hugging-face-plugin-hugging-face-openai/data/eval/benchmark.jsonl"
)
METRICS = ["not_generic", "concise", "specific_action", "no_blame", "changed_when_needed"]


def parse_rewrite(raw: str):
    """Extract the rewrite from a (possibly reasoning-prefixed) output.

    Qwen3.6 is a thinking model: outputs may contain chain-of-thought and a `</think>`
    marker before the final JSON. We take the text after the last `</think>` and parse the
    trailing {...} object. Returns None if no valid JSON object with a non-empty `rewrite`
    is present (e.g. the model spent its whole budget reasoning) — scored as a failure.
    """
    if not isinstance(raw, str):
        return None
    text = raw.rsplit("</think>", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    rw = obj.get("rewrite") if isinstance(obj, dict) else None
    return rw.strip() if isinstance(rw, str) and rw.strip() else None


def load_benchmark(path: Path) -> dict[str, tuple[str, str]]:
    bench = {}
    for row in iter_jsonl(path):
        user = row["messages"][1]["content"]
        current = user.split("Current copy:", 1)[1].split("\n", 1)[0].strip() if "Current copy:" in user else ""
        bench[str(row["id"])] = (current, row["metadata"]["category"])
    return bench


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--benchmark", type=Path, default=Path(DEFAULT_BENCHMARK))
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    bench = load_benchmark(args.benchmark)
    agg = {m: 0.0 for m in METRICS}
    n = 0
    parse_failures = 0
    for r in iter_jsonl(args.preds):
        cid = str(r["id"])
        if cid not in bench:
            continue
        current, category = bench[cid]
        rewrite = parse_rewrite(r.get("prediction", ""))
        if rewrite is None:
            parse_failures += 1
            metrics = {m: 0.0 for m in METRICS}
        else:
            metrics = score_rewrite(current, rewrite, category)
        for k, v in metrics.items():
            agg[k] += v
        n += 1

    agg = {k: round(v / n, 4) for k, v in agg.items()} if n else agg
    agg["overall"] = round(sum(agg.values()) / len(agg), 4) if n else 0.0
    out = {"label": args.label, "rows": n, "json_parse_failures": parse_failures, **agg}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
