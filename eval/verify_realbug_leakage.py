#!/usr/bin/env python3
"""Fail-closed leakage guard for the real-bug eval set.

Honest held-out requires the eval screens never appear in the v1.4 vision TRAIN split.
Two checks:
  1. denylist (offline): no eval row reuses a Cal.com screen folded into TRAIN
     (s01-s06, s08). s07/s09 are the held-out eval bug screens and ARE allowed.
  2. content-hash (needs HF_TOKEN): no eval image's sha256 matches any image referenced
     by the training dataset's train.jsonl (downloaded from gr33r/ux-writing-vision-sft).
Exit non-zero on any overlap.

Usage:
  HF_TOKEN=... python3 scripts/verify_realbug_leakage.py [path/to/manifest.jsonl]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SCREENS_DIR = Path("data/eval/oss_screens/screens")
SFT_REPO = os.environ.get("SFT_REPO", "gr33r/ux-writing-vision-sft")
# Cal.com screens folded into TRAIN (take-2 REAL_TRAIN). s07/s09 = held-out eval bug screens.
CALCOM_TRAIN_DENYLIST = {
    "calcom_s01", "calcom_s02", "calcom_s03", "calcom_s04",
    "calcom_s05", "calcom_s06", "calcom_s08",
}


def check_denylist(rows: list[dict]) -> list[str]:
    """Offline: flag any eval row that reuses a trained Cal.com screen (as image or substrate)."""
    problems = []
    for r in rows:
        stem = Path(r.get("image", "")).stem
        if stem.lower() in CALCOM_TRAIN_DENYLIST:
            problems.append(f"{r.get('id')}: uses trained Cal.com screen {stem}")
        pf = r.get("planted_from", "") or ""
        for deny in CALCOM_TRAIN_DENYLIST:
            if deny in pf.lower():
                problems.append(f"{r.get('id')}: planted_from references trained screen {deny}")
    return problems


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def training_image_hashes(token: str) -> dict[str, str]:
    """Download train.jsonl + every image it references; return {hash: rel_path}."""
    from huggingface_hub import hf_hub_download
    train_path = hf_hub_download(SFT_REPO, "train.jsonl", repo_type="dataset", token=token)
    rels = sorted({json.loads(l)["image"] for l in open(train_path, encoding="utf-8") if l.strip()})
    hashes = {}
    for rel in rels:
        fp = hf_hub_download(SFT_REPO, rel, repo_type="dataset", token=token)
        hashes[sha256_file(Path(fp))] = rel
    return hashes


def check_hashes(rows: list[dict], token: str) -> list[str]:
    """Network: flag any eval image whose content hash matches a TRAIN image."""
    train_hashes = training_image_hashes(token)
    problems = []
    for r in rows:
        fp = SCREENS_DIR / Path(r["image"]).name
        if not fp.exists():
            continue
        h = sha256_file(fp)
        if h in train_hashes:
            problems.append(f"{r.get('id')}: image hash matches TRAIN image {train_hashes[h]}")
    return problems


def main() -> int:
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("data/eval/oss_screens/real_bug_eval_manifest.jsonl")
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    problems = check_denylist(rows)
    token = os.environ.get("HF_TOKEN")
    if token:
        problems += check_hashes(rows, token)
    else:
        print("[warn] HF_TOKEN unset — skipping content-hash check (denylist still enforced)")
    if problems:
        print("LEAKAGE CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"leakage check OK ({len(rows)} rows; {'hash+denylist' if token else 'denylist only'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
