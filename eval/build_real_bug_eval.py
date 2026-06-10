#!/usr/bin/env python3
"""Build the real-bug eval manifest from authored gold.

Reads data/eval/oss_screens/real_bug_gold.jsonl, validates every row (schema, within-screen
invariants, provenance rules), runs a PII text-scan, calls the leakage verifier, then writes
data/eval/oss_screens/real_bug_eval_manifest.jsonl. With --upload it pushes the new screens +
manifest to the dataset the harness reads.

Usage:
  python3 scripts/build_real_bug_eval.py                 # validate + write manifest (dry)
  HF_TOKEN=... python3 scripts/build_real_bug_eval.py --upload
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.verify_realbug_leakage import check_denylist

ROOT = Path(__file__).resolve().parents[1]
# `image` fields are DATASET-relative ("screens/x.png"); locally that same relative path
# lives under data/eval/oss_screens/. Resolve every image against LOCAL_BASE, not ROOT.
LOCAL_BASE = ROOT / "data/eval/oss_screens"
GOLD = LOCAL_BASE / "real_bug_gold.jsonl"
MANIFEST = LOCAL_BASE / "real_bug_eval_manifest.jsonl"
SFT_REPO = os.environ.get("SFT_REPO", "gr33r/ux-writing-vision-sft")

RECOGNIZED = ["duplication", "plural", "brand", "casing", "tone", "terminology"]
PROVENANCE = {"natural", "planted"}
REQUIRED = ["id", "image", "surface", "clean", "split", "gold_types", "provenance", "source", "license"]
# Manifest keys the harness reads/uses, plus the slices the scorer needs.
MANIFEST_KEYS = ["id", "image", "surface", "clean", "split", "gold_types", "provenance", "source"]
PII_PATTERNS = [
    re.compile(r"swim_2_birds@icloud\.com", re.I),
    re.compile(r"chris@nottawacottagebookstore\.ca", re.I),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # any bare email in a text field
]
PII_TEXT_FIELDS = ["surface", "notes"]


def validate_rows(rows: list[dict]) -> list[str]:
    errs = []
    seen = set()
    for r in rows:
        rid = r.get("id", "?")
        for k in REQUIRED:
            if k not in r:
                errs.append(f"{rid}: missing required field '{k}'")
        if rid in seen:
            errs.append(f"{rid}: duplicate id")
        seen.add(rid)
        gt = r.get("gold_types", [])
        bad = [t for t in gt if t not in RECOGNIZED]
        if bad:
            errs.append(f"{rid}: gold_types not recognized: {bad}")
        if r.get("provenance") not in PROVENANCE:
            errs.append(f"{rid}: provenance must be one of {PROVENANCE}")
        if r.get("clean"):
            if gt:
                errs.append(f"{rid}: clean row must have empty gold_types")
        else:
            if not gt:
                errs.append(f"{rid}: bug row (clean=false) must have >=1 gold_type")
            if not r.get("gold_issues"):
                errs.append(f"{rid}: bug row must have >=1 gold_issues entry")
            if not r.get("within_screen", False):
                errs.append(f"{rid}: bug row must set within_screen=true (scope rule)")
        if r.get("provenance") == "planted" and not r.get("planted_from"):
            errs.append(f"{rid}: planted row must record planted_from")
        if r.get("provenance") == "natural" and r.get("planted_from"):
            errs.append(f"{rid}: natural row must NOT carry planted_from")
        img = LOCAL_BASE / r.get("image", "")
        if not img.exists():
            errs.append(f"{rid}: image file missing: {r.get('image')}")
    return errs


def scan_pii(rows: list[dict]) -> list[str]:
    hits = []
    for r in rows:
        for f in PII_TEXT_FIELDS:
            v = r.get(f, "") or ""
            for pat in PII_PATTERNS:
                if pat.search(v):
                    hits.append(f"{r.get('id')}: PII in {f}: {pat.pattern}")
    return hits


def build_manifest(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in MANIFEST_KEYS if k in r} for r in rows]


def main() -> int:
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    errs = validate_rows(rows)
    if errs:
        print("VALIDATION FAILED:"); [print("  -", e) for e in errs]; return 1
    pii = scan_pii(rows)
    if pii:
        print("PII SCAN FAILED (redact text fields):"); [print("  -", p) for p in pii]; return 1
    deny = check_denylist(rows)
    if deny:
        print("LEAKAGE (denylist) FAILED:"); [print("  -", d) for d in deny]; return 1
    man = build_manifest(rows)
    MANIFEST.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in man) + "\n", encoding="utf-8")
    n_bug = sum(1 for r in rows if not r["clean"])
    n_nat = sum(1 for r in rows if not r["clean"] and r["provenance"] == "natural")
    print(f"manifest OK: {len(man)} rows ({n_bug} bug / {n_nat} natural) -> {MANIFEST}")

    if "--upload" in sys.argv:
        token = os.environ.get("HF_TOKEN")
        assert token, "HF_TOKEN required for --upload"
        from huggingface_hub import HfApi
        api = HfApi()
        for r in rows:
            api.upload_file(path_or_fileobj=str(LOCAL_BASE / r["image"]),
                            path_in_repo=r["image"], repo_id=SFT_REPO,
                            repo_type="dataset", token=token)
        api.upload_file(path_or_fileobj=str(MANIFEST), path_in_repo="real_bug_eval_manifest.jsonl",
                        repo_id=SFT_REPO, repo_type="dataset", token=token)
        print(f"[info] uploaded {len(rows)} screens + manifest -> {SFT_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
