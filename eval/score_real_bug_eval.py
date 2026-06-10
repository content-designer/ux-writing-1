#!/usr/bin/env python3
"""Provenance-sliced scorer for the real-bug eval.

Reads the eval-harness output jsonl (each row carries gold_types, clean, provenance,
base_out, adapter_out), applies the deterministic postfilter to the ADAPTER output (the
production serving config), and reports the two honest numbers the spec requires:
  - naturally-occurring-only recall (the trustworthy number; planted screens excluded)
  - full recall (incl. planted-on-real)
plus per-type recall, multi-issue recall (probes the collapse weakness), and clean-screen
over-flag counts. Adapter(+postfilter) vs prompt-only base.

Usage:
  python3 scripts/score_real_bug_eval.py path/to/harness_output.jsonl [--md OUT.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.consistency_postfilter import filter_output  # deterministic serving guard
from eval.build_real_bug_eval import RECOGNIZED  # single source of truth (score->build->verify)


def canon_set(t: str) -> set:
    """Local mirror of eval_consistency_adapter.canon_set (that file imports torch)."""
    t = (t or "").lower(); out = set()
    if "dup" in t: out.add("duplication")
    if any(k in t for k in ("plural", "number", "agreement")): out.add("plural")
    if "brand" in t: out.add("brand")
    if any(k in t for k in ("casing", "capital")): out.add("casing")
    if "tone" in t: out.add("tone")
    if any(k in t for k in ("terminolog", "consist", "inconsist")) and "brand" not in t:
        out.add("terminology")
    if "terminolog" in t and "brand" in t:
        out |= {"terminology", "brand"}
    return out


def _types_from_obj(obj: dict) -> set:
    s = set()
    for it in obj.get("issues", []):
        s |= canon_set(it.get("type", ""))
    return s


def adapter_types_filtered(raw: str) -> set:
    """Adapter prediction types AFTER the deterministic postfilter (production config)."""
    return _types_from_obj(filter_output(raw))


def base_types(raw: str) -> set:
    try:
        obj = json.loads(raw)
    except Exception:
        return set()
    return _types_from_obj(obj)


def _n_issues(raw: str, filtered: bool) -> int:
    if filtered:
        return len(filter_output(raw).get("issues", []))
    try:
        return len(json.loads(raw).get("issues", []))
    except Exception:
        return 0


def _recall(rows, pred_fn) -> dict:
    tp = fn = 0
    for r in rows:
        g = set(r["gold_types"]); p = pred_fn(r)
        tp += len(g & p); fn += len(g - p)
    return {"tp": tp, "fn": fn, "recall": tp / (tp + fn) if tp + fn else 0.0}


def _side(rows, pred_fn, n_fn):
    bugs = [r for r in rows if not r["clean"]]
    clean = [r for r in rows if r["clean"]]
    natural = [r for r in bugs if r.get("provenance") == "natural"]
    multi = [r for r in bugs if len(r["gold_types"]) >= 2]
    per_type = {}
    for t in RECOGNIZED:
        sub = [r for r in bugs if t in r["gold_types"]]
        per_type[t] = _recall(sub, pred_fn) if sub else None
    return {
        "natural": _recall(natural, pred_fn),
        "full": _recall(bugs, pred_fn),
        "multi": _recall(multi, pred_fn),
        "per_type": per_type,
        "clean_overflags": sum(n_fn(r) for r in clean),
    }


def score(rows: list[dict]) -> dict:
    return {
        "adapter": _side(rows, lambda r: adapter_types_filtered(r["adapter_out"]),
                         lambda r: _n_issues(r["adapter_out"], filtered=True)),
        "base": _side(rows, lambda r: base_types(r["base_out"]),
                      lambda r: _n_issues(r["base_out"], filtered=False)),
        "n_bug": sum(1 for r in rows if not r["clean"]),
        "n_natural_bug": sum(1 for r in rows if not r["clean"] and r.get("provenance") == "natural"),
        "n_clean": sum(1 for r in rows if r["clean"]),
    }


def to_md(s: dict) -> str:
    a, b = s["adapter"], s["base"]
    lines = ["# Real-bug eval — adapter(+postfilter) vs base", ""]
    lines.append(f"- bug screens: {s['n_bug']} (natural {s['n_natural_bug']}); clean: {s['n_clean']}")
    lines.append("")
    lines.append("| metric | base | adapter(+postfilter) |")
    lines.append("|---|---|---|")
    lines.append(f"| **natural-only recall** | {b['natural']['recall']} | **{a['natural']['recall']}** |")
    lines.append(f"| full recall (incl. planted) | {b['full']['recall']} | {a['full']['recall']} |")
    lines.append(f"| multi-issue recall | {b['multi']['recall']} | {a['multi']['recall']} |")
    lines.append(f"| clean over-flags | {b['clean_overflags']} | {a['clean_overflags']} |")
    lines.append("")
    lines.append("| type | base recall | adapter recall |")
    lines.append("|---|---|---|")
    for t in RECOGNIZED:
        ar, br = a["per_type"][t], b["per_type"][t]
        if ar is not None and br is not None:
            lines.append(f"| {t} | {br['recall']} | {ar['recall']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    if "--md" in sys.argv and sys.argv.index("--md") == len(sys.argv) - 1:
        print("error: --md requires a path argument"); return 2
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    s = score(rows)
    print(json.dumps(s, indent=2))
    if "--md" in sys.argv:
        out = sys.argv[sys.argv.index("--md") + 1]
        Path(out).write_text(to_md(s), encoding="utf-8")
        print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
