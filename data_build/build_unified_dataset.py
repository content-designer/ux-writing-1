"""Build the unified, versioned, zero-leakage dataset `gr33r/ux-writing-sft`.

Consolidates the prior two datasets into ONE repo with tagged subsets so both training
runs (text baseline 1a, combined 1b) draw from the same source of truth:

  task=rewrite,   modality=text   : UI string + code context -> {rewrite, reason, risk}
  task=consistency,modality=vision: screenshot -> {inventory, issues}

Every row is tagged (task, modality, source, split) and `messages`/`metadata`/`provenance`
are stored as JSON strings so Arrow can hold both task shapes in one table.

Zero-leakage guarantees enforced here:
  - vision TRAIN/VAL is synthetic-only (real screenshots are added as a `test` split by
    data_build/annotate_new_eval.py — never here).
  - rewrite VAL is carved from train by input_key (uxft.dataset.split_dedup), so no UI
    string spans train and val; rewrite TEST is the separate hand-authored benchmark.

Usage:
  python data_build/build_unified_dataset.py                 # build + validate locally (no push)
  python data_build/build_unified_dataset.py --push          # also push to gr33r/ux-writing-sft (private)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Make the repo root importable regardless of cwd (the source snapshot also ships a
# `uxft/` package, so we must put THIS repo first on sys.path).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from uxft import schema  # noqa: E402
from uxft.dataset import input_key, split_dedup  # noqa: E402

DEFAULT_SOURCE = "/Users/christophergreer/Documents/Codex/2026-05-30/hugging-face-plugin-hugging-face-openai"
DATASET_REPO = "gr33r/ux-writing-sft"
VAL_SIZE = 60  # distinct rewrite input_keys held out for validation

VISION_PROVENANCE = {
    "generation_method": "synthetic HTML -> headless Chromium render (synthetic_v14b)",
    "copyright_posture": "synthetic_owner_generated",
    "license": "Apache-2.0",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def screen_index(source: Path) -> dict[str, Path]:
    """Map screenshot basename -> local path across the synthetic screen dirs."""
    index: dict[str, Path] = {}
    for sub in ("synthetic_v14b/screens", "synthetic_v14/screens"):
        d = source / "data/eval/oss_screens" / sub
        if d.exists():
            for png in d.glob("*.png"):
                index.setdefault(png.name, png)  # prefer v14b (first)
    return index


def tag_rewrite(row: dict, split: str) -> dict:
    return {
        "id": row["id"],
        "task": "rewrite",
        "modality": "text",
        "source": "synthetic",
        "split": split,
        "image": "",
        "messages": row["messages"],
        "metadata": row.get("metadata", {}),
        "provenance": row.get("provenance", {}),
    }


def tag_vision(row: dict, split: str) -> dict:
    return {
        "id": row["id"],
        "task": "consistency",
        "modality": "vision",
        "source": "synthetic",
        "split": split,
        "image": f"screens/{Path(row['image']).name}",
        "messages": row["messages"],
        "metadata": row.get("metadata", {}),
        "provenance": row.get("provenance", VISION_PROVENANCE),
    }


def build(source: Path) -> tuple[dict[str, list[dict]], set[str]]:
    """Return {split: rich_rows} and the set of referenced screen basenames."""
    rewrite_train_raw = read_jsonl(source / "data/processed/train.v3.jsonl")
    rewrite_test_raw = read_jsonl(source / "data/eval/benchmark.jsonl")
    vision_train_raw = read_jsonl(source / "data/processed/vision_sft/train.jsonl")
    vision_val_raw = read_jsonl(source / "data/processed/vision_sft/heldout.jsonl")

    # Carve a zero-leakage rewrite validation slice (held-out input_keys, train uncapped).
    rw_train, rw_val = split_dedup(rewrite_train_raw, eval_size=VAL_SIZE, max_per_input=99)

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    splits["train"] += [tag_rewrite(r, "train") for r in rw_train]
    splits["validation"] += [tag_rewrite(r, "validation") for r in rw_val]
    splits["test"] += [tag_rewrite(r, "test") for r in rewrite_test_raw]
    splits["train"] += [tag_vision(r, "train") for r in vision_train_raw]
    splits["validation"] += [tag_vision(r, "validation") for r in vision_val_raw]

    referenced = {Path(r["image"]).name for r in vision_train_raw + vision_val_raw}
    return splits, referenced


def validate(splits: dict[str, list[dict]]) -> int:
    issues = 0
    for split, rows in splits.items():
        for i, row in enumerate(rows):
            for err in schema.validate_row(row, set(), max_tokens=8192):
                print(f"[{split}:{i}] {row.get('id')}: {err}")
                issues += 1
    return issues


def assert_no_leakage(splits: dict[str, list[dict]]) -> None:
    """No rewrite input_key may span train and {validation,test}."""
    train_keys = {input_key(r) for r in splits["train"] if r["task"] == "rewrite"}
    for split in ("validation", "test"):
        held = {input_key(r) for r in splits[split] if r["task"] == "rewrite"}
        overlap = train_keys & held
        if overlap:
            raise SystemExit(f"LEAKAGE: {len(overlap)} rewrite input_keys span train/{split}: {list(overlap)[:3]}")
    print("leakage check: OK (no rewrite input_key spans train/validation/test)")


def to_storage(row: dict) -> dict:
    """Flatten a rich row to the Arrow-friendly storage schema (all simple types)."""
    return {
        "id": row["id"],
        "task": row["task"],
        "modality": row["modality"],
        "source": row["source"],
        "split": row["split"],
        "image": row["image"],
        "messages_json": json.dumps(row["messages"], ensure_ascii=False),
        "metadata_json": json.dumps(row.get("metadata", {}), ensure_ascii=False),
        "provenance_json": json.dumps(row.get("provenance", {}), ensure_ascii=False),
    }


def write_local(splits: dict[str, list[dict]], referenced: set[str], screens: dict[str, Path],
                out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screens").mkdir(exist_ok=True)
    for split, rows in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(to_storage(row), ensure_ascii=False) + "\n")
    missing = []
    for name in sorted(referenced):
        src = screens.get(name)
        if src is None:
            missing.append(name)
            continue
        shutil.copy2(src, out_dir / "screens" / name)
    if missing:
        raise SystemExit(f"{len(missing)} referenced screens not found locally, e.g. {missing[:5]}")
    print(f"wrote splits + {len(referenced)} screens to {out_dir}")


def push(out_dir: Path, repo: str) -> None:
    from datasets import Dataset, DatasetDict  # lazy: only needed to push
    from huggingface_hub import HfApi

    dd = DatasetDict({
        split: Dataset.from_list([json.loads(l) for l in (out_dir / f"{split}.jsonl").read_text().splitlines() if l.strip()])
        for split in ("train", "validation", "test")
    })
    print({k: len(v) for k, v in dd.items()})
    dd.push_to_hub(repo, private=True)
    api = HfApi()
    api.upload_folder(repo_id=repo, repo_type="dataset",
                      folder_path=str(out_dir / "screens"), path_in_repo="screens")
    # Upload as DATASET_CARD.md (NOT README.md) so we don't clobber the auto-generated
    # dataset_info frontmatter that powers the Hub dataset viewer.
    card = Path(REPO_ROOT) / "docs/DATASET_CARD.md"
    if card.exists():
        api.upload_file(repo_id=repo, repo_type="dataset",
                        path_or_fileobj=str(card), path_in_repo="DATASET_CARD.md")
    print(f"pushed -> https://huggingface.co/datasets/{repo}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/ux-writing-sft-build"))
    ap.add_argument("--repo", default=DATASET_REPO)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    splits, referenced = build(args.source)
    counts = {s: len(r) for s, r in splits.items()}
    by_task = {}
    for s, rows in splits.items():
        for r in rows:
            by_task[(s, r["task"])] = by_task.get((s, r["task"]), 0) + 1
    print("split counts:", counts)
    print("by (split, task):", {f"{k[0]}/{k[1]}": v for k, v in sorted(by_task.items())})

    issues = validate(splits)
    if issues:
        raise SystemExit(f"validation FAILED with {issues} issues")
    print("validation: OK")
    assert_no_leakage(splits)

    write_local(splits, referenced, screen_index(args.source), args.out_dir)

    if args.push:
        push(args.out_dir, args.repo)
    else:
        print("(dry run — not pushed; re-run with --push to upload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
