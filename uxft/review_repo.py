"""Review UI strings in a repo with a trained or baseline OpenAI-compatible model.

Scans a frontend codebase for candidate UI strings (uxft.scan), sends each to an
OpenAI-compatible /v1/chat/completions endpoint, and writes diff-friendly JSONL —
a human-in-the-loop review artifact, never auto-applied.

    python3 -m uxft.review_repo /path/to/repo \
        --endpoint https://<modal-host>/v1/chat/completions \
        --api-key $UXW1_TOKEN --limit 200 --out review.jsonl

The user prompt matches the model's TRAINING contract for repo candidates
(uxft.dataset.row_from_candidate): Product surface / Audience / User state /
Content type / Current copy / Code/context / Constraints.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle — python.org macOS installs often lack system roots."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

from uxft.escape_postfilter import is_contract_escape
from uxft.policy import SYSTEM_PROMPT
from uxft.scan import Candidate, scan_repo


def prompt_for(candidate: Candidate) -> str:
    """In-distribution repo-candidate prompt (mirrors uxft.dataset.row_from_candidate)."""
    return (
        "Product surface: existing codebase\n"
        "Audience: product user\n"
        f"User state: using the screen that contains {candidate.path}:{candidate.line}\n"
        f"Content type: {candidate.kind}\n"
        f"Current copy: {candidate.current_copy}\n"
        f"Code/context:\n{candidate.context}\n"
        "Constraints: Suggest a UX writing rewrite only if the context supports it. "
        "Preserve the intended product behavior."
    )


def extract_contract_json(text: str) -> dict | None:
    """Pull the {rewrite, reason, risk} object out of a possibly reasoning-prefixed reply.

    Qwen3.6 is a thinking model: take the text after the last </think>, then keep the
    LAST parseable JSON object that has a string rewrite — a greedy first-{…last-} span
    grabs draft JSON inside leaked reasoning. Scans with json.JSONDecoder().raw_decode
    rather than a flat {…} regex: rewrites that correctly keep {{ variables }} contain
    braces, which a brace regex can never match (such outputs were mis-flagged as
    non-JSON and dropped). Returns None when no valid object exists.
    """
    if not isinstance(text, str):
        return None
    tail = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    parsed: dict | None = None
    index = 0
    while True:
        start = tail.find("{", index)
        if start == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(tail[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("rewrite"), str):
            parsed = obj
        index = start + max(consumed, 1)
    return parsed


def call_openai_compatible(endpoint: str, model: str, candidate: Candidate,
                           api_key: str | None = None, timeout: int = 180) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for(candidate)},
        ],
        "temperature": 0.1,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        payload["_auth"] = api_key  # body-carried token for endpoints that can't read headers
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = extract_contract_json(content)
    if parsed is None:
        parsed = {"rewrite": "", "reason": content, "risk": "Model returned non-JSON output.",
                  "confidence": 0}
    return parsed


def dry_run_prediction(candidate: Candidate) -> dict:
    return {
        "rewrite": "",
        "reason": "Dry run only. Connect an OpenAI-compatible endpoint to generate rewrite suggestions.",
        "risk": "",
        "confidence": 0,
    }


def review(candidates: list[Candidate], endpoint: str | None, model: str,
           api_key: str | None, workers: int = 8) -> list[dict]:
    def one(candidate: Candidate) -> dict:
        if not endpoint:
            prediction = dry_run_prediction(candidate)
        else:
            try:
                prediction = call_openai_compatible(endpoint, model, candidate, api_key)
            except Exception as exc:  # keep the run going; record the failure
                prediction = {"rewrite": "", "reason": f"request failed: {exc}",
                              "risk": "endpoint_error", "confidence": 0}
        suggested = prediction.get("rewrite", "")
        risk = prediction.get("risk", "")
        if suggested and is_contract_escape(suggested):
            risk = "contract_escape"  # JSX/ternary, not copy — flag, don't count as a change
        return {
            **asdict(candidate),
            "suggested_copy": suggested,
            "reason": prediction.get("reason", ""),
            "risk": risk,
            "confidence": prediction.get("confidence", 0),
        }

    if not endpoint:
        return [one(c) for c in candidates]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, candidates))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a repo and produce UX writing rewrite suggestions.")
    parser.add_argument("repo", type=Path)
    parser.add_argument("--endpoint", default=None, help="OpenAI-compatible /v1/chat/completions endpoint.")
    parser.add_argument("--model", default="gr33r/ux-writing-1")
    parser.add_argument("--api-key", default=None, help="Bearer token for the endpoint.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("repo_review.jsonl"))
    args = parser.parse_args()

    candidates = scan_repo(args.repo.resolve(), limit=args.limit)
    rows = review(candidates, args.endpoint, args.model, args.api_key, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    changed = sum(1 for r in rows if r["suggested_copy"] and r["risk"] != "contract_escape"
                  and r["suggested_copy"] != r["current_copy"])
    print(f"wrote {len(rows)} review rows to {args.out} ({changed} suggested changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
