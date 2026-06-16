"""End-to-end check that the scanner fixes resolve the exact strings that failed.

Reconstructs the seed-20260613 sample of 60 from the real PostHog run artifact, joins it
with the hand labels in eval/failure_sample_60.csv, and asserts — on the real strings —
that the three fixes land:
  - code-breaking non-UI rows are now rejected by the scanner filter (is_ui_copy False)
  - contract-escape rows are flagged (is_contract_escape True)
  - apostrophe-truncated rows are now captured whole by the scanner

Milder non-UI rows (intentionally not caught by the conservative filter) are reported but
not treated as failures.

Run:  python3 eval/verify_scanner_fix.py   (needs HF auth for gr33r/ux-writing-sft)
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.escape_postfilter import is_contract_escape
from uxft.scan import scan_repo
from uxft.ui_filter import is_ui_copy

SEED = 20260613
N = 60
CSV = Path(__file__).parent / "failure_sample_60.csv"


def matched_line(row: dict) -> str:
    """The source line `current_copy` was extracted from, recovered from the numbered context."""
    want = str(row["line"])
    for ln in row.get("context", "").splitlines():
        num, sep, src = ln.partition(": ")
        if sep and num.strip() == want:
            return src
    return ""


def source_from_context(row: dict) -> str:
    return "\n".join(ln.partition(": ")[2] for ln in row.get("context", "").splitlines())


def apostrophe_truncated(row: dict) -> bool:
    cc = " ".join(row["current_copy"].split())
    return bool(re.search(re.escape(cc) + r"'[A-Za-z]", row.get("context", "")))


def captured_whole(row: dict) -> bool:
    frag = " ".join(row["current_copy"].split())
    with tempfile.TemporaryDirectory() as d:
        ext = Path(row["path"]).suffix or ".tsx"
        (Path(d) / f"snippet{ext}").write_text(source_from_context(row), encoding="utf-8")
        vals = {c.current_copy for c in scan_repo(Path(d))}
    return any(v != frag and v.startswith(frag) and len(v) > len(frag) for v in vals)


def main() -> int:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("gr33r/ux-writing-sft", "eval_preds/posthog_review.jsonl", repo_type="dataset")
    changed = [r for r in (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip())
               if r.get("changed")]
    random.seed(SEED)
    sample = random.sample(changed, N)
    labels = list(csv.DictReader(CSV.open(encoding="utf-8")))
    assert len(labels) == N and all(labels[i]["path"] == sample[i]["path"] for i in range(N)), \
        "sample/CSV misaligned — reconstruction drifted"

    counts: dict[str, list[int]] = {"non_ui_breaks": [0, 0], "escape": [0, 0],
                                    "truncation": [0, 0], "non_ui_mild(soft)": [0, 0]}
    hard_fails: list[str] = []

    def record(bucket: str, ok: bool, idx: int, detail: str, hard: bool) -> None:
        counts[bucket][0] += ok
        counts[bucket][1] += 1
        if hard and not ok:
            hard_fails.append(f"#{idx:02d} {bucket}: {detail}")

    for i, (r, lab) in enumerate(zip(sample, labels)):
        disp = lab["disposition"]
        if disp == "error_non_ui_breaks_code":
            record("non_ui_breaks", not is_ui_copy(r["current_copy"], matched_line(r), r["path"]),
                   i, f"still kept {r['current_copy']!r}", hard=True)
        elif disp == "error_non_ui":
            record("non_ui_mild(soft)", not is_ui_copy(r["current_copy"], matched_line(r), r["path"]),
                   i, "", hard=False)
        elif disp == "error_contract_escape":
            record("escape", is_contract_escape(r["suggested_copy"]),
                   i, f"not flagged {r['suggested_copy']!r}", hard=True)
        elif lab["mode"] == "M1" and apostrophe_truncated(r):
            record("truncation", captured_whole(r), i, f"still truncated {r['current_copy']!r}", hard=True)

    for name, (good, tot) in counts.items():
        if tot:
            print(f"  {name:18} {good}/{tot} fixed")
    if hard_fails:
        print("\nFAILURES:")
        for f in hard_fails:
            print("  " + f)
        return 1
    print("\nAll scanner-fix checks passed on the real failure strings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
