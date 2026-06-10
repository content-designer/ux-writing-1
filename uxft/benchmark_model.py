"""Run the fixed UX writing benchmark against an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from uxft.schema import iter_jsonl


def call_chat(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"rewrite": "", "reason": content, "risk": "Model returned non-JSON output."}
    return parsed


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["id"]) for row in iter_jsonl(path)}


def normalize_prediction(row_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "rewrite": str(parsed.get("rewrite", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "risk": str(parsed.get("risk", "")).strip(),
        "confidence": parsed.get("confidence", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark predictions with a chat model.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/eval/benchmark.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/baseline_predictions.jsonl"))
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible /v1/chat/completions URL.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default=None, help="Read bearer token from this env var.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    done = existing_ids(args.out) if args.resume else set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_jsonl(args.benchmark)
    written = 0

    with args.out.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row["id"])
            if row_id in done:
                continue
            try:
                parsed = call_chat(args.endpoint, args.model, row["messages"][:2], api_key, args.timeout)
            except urllib.error.URLError as exc:
                raise SystemExit(f"request failed for {row_id}: {exc}") from exc
            handle.write(json.dumps(normalize_prediction(row_id, parsed), ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if args.limit and written >= args.limit:
                break
            if args.sleep:
                time.sleep(args.sleep)

    print(f"wrote {written} predictions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
