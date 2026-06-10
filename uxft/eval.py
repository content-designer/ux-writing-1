"""Lightweight benchmark scoring for UX writing rewrite predictions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from uxft.schema import assistant_json, iter_jsonl

GENERIC = {"ok", "submit", "confirm", "proceed", "click here", "read more", "done", "error"}
BLAME_WORDS = {"invalid", "illegal", "bad", "wrong"}
SPECIFIC_VERBS = {
    "add",
    "cancel",
    "change",
    "choose",
    "clear",
    "close",
    "connect",
    "delete",
    "download",
    "enable",
    "retry",
    "save",
    "search",
    "send",
    "sign",
    "try",
    "update",
    "view",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())


def score_rewrite(current: str, rewrite: str, category: str) -> dict[str, float]:
    current_words = words(current)
    rewrite_words = words(rewrite)
    rewrite_lower = rewrite.lower().strip()
    token_count = len(rewrite_words)

    scores = {
        "not_generic": 1.0 if rewrite_lower not in GENERIC else 0.0,
        "concise": 1.0 if token_count <= 12 else max(0.0, 1.0 - ((token_count - 12) / 12)),
        "specific_action": 1.0 if SPECIFIC_VERBS.intersection(rewrite_words) else 0.5,
        "no_blame": 0.0 if BLAME_WORDS.intersection(rewrite_words) and category != "inline_error" else 1.0,
        "changed_when_needed": 0.0 if rewrite_lower == " ".join(current_words) and rewrite_lower in GENERIC else 1.0,
    }
    return scores


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        predictions[str(row["id"])] = row
    return predictions


def score(benchmark: Path, predictions_path: Path | None) -> dict[str, Any]:
    benchmark_rows = iter_jsonl(benchmark)
    predictions = load_predictions(predictions_path) if predictions_path else {}
    scored_rows = []

    for row in benchmark_rows:
        expected = assistant_json(row["messages"][2]["content"])
        prediction = predictions.get(row["id"], {})
        predicted_rewrite = prediction.get("rewrite") or expected["rewrite"]
        current = row["messages"][1]["content"].split("Current copy:", 1)[1].split("\n", 1)[0].strip()
        metrics = score_rewrite(current, predicted_rewrite, row["metadata"]["category"])
        scored_rows.append({"id": row["id"], "metrics": metrics})

    aggregate: dict[str, float] = {}
    for scored in scored_rows:
        for key, value in scored["metrics"].items():
            aggregate[key] = aggregate.get(key, 0.0) + value
    if scored_rows:
        aggregate = {key: round(value / len(scored_rows), 4) for key, value in aggregate.items()}
        aggregate["overall"] = round(sum(aggregate.values()) / len(aggregate), 4)

    return {"rows": len(scored_rows), "aggregate": aggregate, "details": scored_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score UX writing rewrite predictions.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/eval/benchmark.jsonl"))
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/eval/scores.json"))
    args = parser.parse_args()

    result = score(args.benchmark, args.predictions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

