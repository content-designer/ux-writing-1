"""Stage the 25 new reference screenshots as real-world CONSISTENCY eval cases.

These images (Cal.com + Ghost captures) are EVAL-ONLY — they strengthen the
underpowered real-bug eval and must never enter training (zero leakage).

This script (no GPU required):
  1. Copies + renames the raw screenshots to stable ids (calcom_n01.., ghost_n01..).
  2. Content-hashes each one and asserts none collide with a synthetic TRAIN screen
     (defense-in-depth; train is synthetic-only anyway).
  3. Emits a gold-manifest SCAFFOLD you fill in by hand (transcribe strings, label the
     relationship issues). An optional model-assisted transcription pass runs on Modal
     later — this scaffold is the human-in-the-loop artifact.

After you fill in the gold, a follow-up appends these rows to the `test` split of
`gr33r/ux-writing-sft` (task=consistency, source=oss/merchant) for eval_consistency.

Usage:
  python data_build/annotate_new_eval.py
  python data_build/annotate_new_eval.py --source /Users/christophergreer/fine-tune-image-references
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_SOURCE = Path("/Users/christophergreer/fine-tune-image-references")
TRAIN_SCREENS = Path(
    "/Users/christophergreer/Documents/Codex/2026-05-30/hugging-face-plugin-hugging-face-openai"
    "/data/eval/oss_screens/synthetic_v14b/screens"
)
# folder name -> (id prefix, source tag, license, url)
SUBDIRS = {
    "Cal.com": ("calcom", "oss", "AGPL-3.0/MIT (Cal.com)", "https://github.com/calcom/cal.com"),
    "Ghost": ("ghost", "oss", "MIT (Ghost)", "https://github.com/TryGhost/Ghost"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/ux-writing-neweval"))
    args = ap.parse_args()

    out_screens = args.out_dir / "screens"
    out_screens.mkdir(parents=True, exist_ok=True)

    train_hashes = {sha256(p) for p in TRAIN_SCREENS.glob("*.png")} if TRAIN_SCREENS.exists() else set()
    print(f"loaded {len(train_hashes)} synthetic train-screen hashes for leakage check")

    manifest: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for folder, (prefix, source, lic, url) in SUBDIRS.items():
        src_dir = args.source / folder
        if not src_dir.exists():
            print(f"WARN: {src_dir} missing, skipping")
            continue
        pngs = sorted(src_dir.glob("*.png"))
        for i, png in enumerate(pngs, 1):
            new_id = f"{prefix}_n{i:02d}"
            h = sha256(png)
            if h in train_hashes:
                raise SystemExit(f"LEAKAGE: {png.name} hash matches a TRAIN screen — refusing to add to eval")
            if h in seen_hashes:
                print(f"WARN: {png.name} is a duplicate of {seen_hashes[h]} (same bytes); skipping")
                continue
            seen_hashes[h] = new_id
            shutil.copy2(png, out_screens / f"{new_id}.png")
            manifest.append({
                "id": new_id,
                "image": f"screens/{new_id}.png",
                "surface": f"{folder} — TODO describe screen",
                "clean": None,                      # TODO: true if no relationship issue
                "split": "test",
                "task": "consistency",
                "source": source,
                "gold_types": [],                   # TODO: e.g. ["terminology","casing"]
                "gold_issues": [],                  # TODO: [{"type","strings":[...],"problem","fix"}]
                "provenance": "natural",
                "license": lic,
                "url": url,
                "sha256": h,
                "notes": "new reference 2026-06; hand-label before use",
            })

    manifest_path = args.out_dir / "new_eval_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in manifest:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"staged {len(manifest)} screens -> {out_screens}")
    print(f"wrote gold SCAFFOLD -> {manifest_path}")
    print("NEXT: fill clean/gold_types/gold_issues by hand (optionally use the Modal "
          "transcription aid), then append these as test/consistency rows to gr33r/ux-writing-sft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
