"""Regression asserts for the consistency-pass serving guard (consistency_postfilter).

Locks in the case-SENSITIVE identical-string guard: it must drop only truly-identical
strings (the 'two identical strings flagged as inconsistent' hallucination) while letting
case-only brand/casing differences survive (those ARE the bug). See the 2026-06-02 fix.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.consistency_postfilter import filter_output, keep_issue


def _issue(t, strings):
    return {"type": t, "strings": strings, "problem": "x", "fix": "y"}


def test_postfilter():
    # DROP: truly identical (whitespace-insensitive) — the hallucination the guard exists for.
    assert not keep_issue(_issue("terminology", ["Ghost(Pro)", "Ghost(Pro)"]))
    assert not keep_issue(_issue("brand", ["Ghost(Pro)", "Ghost(Pro) "]))  # trailing ws still identical

    # KEEP: case-only differences are real brand/casing bugs and must survive.
    assert keep_issue(_issue("brand", ["Github", "GitHub"]))
    assert keep_issue(_issue("brand", ["Facetime", "FaceTime"]))
    assert keep_issue(_issue("casing", ["Ghost", "ghost"]))

    # KEEP: differences beyond case, and non-DIFF_REQUIRED types, are untouched.
    assert keep_issue(_issue("terminology", ["Team", "Teams"]))
    assert keep_issue(_issue("duplication", ["a", "a"]))  # duplication is not in DIFF_REQUIRED

    # filter_output end-to-end: identical dropped, case-only kept.
    raw = json.dumps({"inventory": [], "issues": [
        _issue("terminology", ["Ghost(Pro)", "Ghost(Pro)"]),
        _issue("brand", ["Github", "GitHub"]),
    ]})
    kept = filter_output(raw)["issues"]
    assert len(kept) == 1 and kept[0]["type"] == "brand", kept

    print("check_postfilter OK")
