#!/usr/bin/env python3
"""Deterministic serving guard for the consistency pass.

Take-4 tried to kill the "two identical strings flagged as inconsistent" hallucination
with training-side fixes (legit-repeat negatives + a prompt clause). It did NOT work —
v3 still flagged identical 'Ghost(Pro)' strings. A deterministic post-filter is the
reliable fix: a terminology / brand / casing issue REQUIRES the strings to actually
differ. Applying this to v2's real outputs cut Ghost false-positives 2 -> 1 (removed the
hallucination; kept the one defensible 'Ghost' vs 'ghost' casing flag).

Use as the serving wrapper around any consistency-pass output.
"""
from __future__ import annotations

import json

DIFF_REQUIRED = {"terminology", "brand", "casing"}


def _norm(s: str) -> str:
    # Case-SENSITIVE on purpose: the guard drops only truly-identical strings (the
    # "two identical strings flagged as inconsistent" hallucination, e.g. Ghost(Pro)/
    # Ghost(Pro)). Case IS a real difference for a brand/casing bug, so 'Github' vs
    # 'GitHub' must survive — lowercasing here would silently eat legitimate flags.
    return " ".join(s.split()).strip()


def _canon(t: str) -> str:
    t = (t or "").lower()
    if "terminolog" in t or "consist" in t:
        return "terminology"
    if "brand" in t:
        return "brand"
    if "cas" in t:
        return "casing"
    return t


def keep_issue(issue: dict) -> bool:
    """False for issues that claim a difference between strings that are actually identical."""
    if _canon(issue.get("type", "")) in DIFF_REQUIRED:
        strings = [_norm(x) for x in issue.get("strings", []) if isinstance(x, str)]
        if len(strings) >= 2 and len(set(strings)) < 2:
            return False
    return True


def filter_output(raw_text: str) -> dict:
    """Parse a consistency-pass JSON output and drop hallucinated identical-string issues."""
    try:
        obj = json.loads(raw_text)
    except Exception:
        return {"inventory": [], "issues": [], "parse_error": True}
    obj["issues"] = [it for it in obj.get("issues", []) if keep_issue(it)]
    return obj


if __name__ == "__main__":
    import sys
    print(json.dumps(filter_output(sys.stdin.read()), ensure_ascii=False, indent=2))
