"""Curate before/after highlight cards from the PostHog review artifact.

Picks are pinned by (path, line) so the selection is reproducible from
eval_preds/posthog_review.jsonl; any pick missing from the artifact is skipped.

    python3 scripts/pick_highlights.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (path, line) of curated rows — chosen for clear, honest, video-sized wins.
# The page shows hands of four; each click deals the next four, wrapping around.
PICKS: list[tuple[str, int]] = [
    # hand 1
    ("products/ai_observability/frontend/settings/LLMProviderKeysSettings.tsx", 54),   # 'Invalid' -> 'Invalid API key'
    ("frontend/src/scenes/feature-flags/FeatureFlagSettings.tsx", 209),                # 'Done' -> 'Save changes'
    ("products/actions/frontend/components/ActionsTable.tsx", 172),                    # empty-state rewrite
    ("frontend/src/queries/validators.js", 12473),                                     # 'must be string' -> plain language
    # hand 2
    ("products/workflows/frontend/Channels/SlackSetup/SlackSetupModal.tsx", 19),       # 'Connect Slack'
    ("frontend/src/lib/components/AuthorizedUrlList/AuthorizedUrlList.tsx", 179),      # 'NB...' -> direct instruction
    ("products/ai_observability/frontend/clusters/ClusterDetailScatterPlot.tsx", 193), # 'click to view trace' -> 'View trace'
    ("products/conversations/frontend/browserNotificationLogic.ts", 99),               # 'Click to view...' -> 'View...'
    # hand 3
    ("products/workflows/frontend/Workflows/hogflows/panel/testing/hogFlowEditorTestLogic.ts", 571),  # 'Invalid JSON' -> specific
    ("frontend/src/queries/validators.js", 11531),                                     # additionalProperties -> plain language
    ("frontend/src/queries/nodes/InsightViz/TrendsFormula.tsx", 145),                  # consequence-bearing button label
    ("frontend/src/queries/validators.js", 5618),                                      # descriptive key guidance
    # hand 4
    ("frontend/src/scenes/data-warehouse/scene/OverviewTab.tsx", 88),                  # 'Time when this job was created' -> 'Created at'
    ("frontend/src/scenes/trends/mathsLogic.tsx", 380),                                # percentile jargon -> 'Median ...'
    ("frontend/src/scenes/cohorts/CohortEdit.tsx", 283),                               # 'Delete' -> 'Delete cohort'
    ("frontend/src/scenes/billing/BillingHero.tsx", 134),                              # 'Lucky you!' -> "You're on the YC plan"
    # hand 5
    ("products/workflows/frontend/Workflows/WorkflowMetrics.tsx", 42),                 # tooltip rewritten around the user
    ("frontend/src/taxonomy/core-filter-definitions-by-group.json", 4592),             # error message in plain words
    ("frontend/src/lib/components/CyclotronJob/integrations/integrationSetups.tsx", 50),  # 'Configure new...' -> 'Connect Databricks'
    ("frontend/src/scenes/insights/aggregationAxisFormat.ts", 20),                     # adds an example to a label
]

AUTO_LIMIT = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-jsonl", type=Path, default=None,
                    help="local posthog_review.jsonl; default: fetch from the dataset repo")
    ap.add_argument("--auto", action="store_true",
                    help="emit candidate picks (changed, short, plain-language) to choose from")
    ap.add_argument("--out", type=Path, default=Path("docs/demo/posthog_review_highlights.json"))
    args = ap.parse_args()

    if args.review_jsonl:
        path = args.review_jsonl
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("gr33r/ux-writing-sft", "eval_preds/posthog_review.jsonl",
                                    repo_type="dataset", force_download=True))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = {(r["path"], r["line"]): r for r in rows}

    if args.auto:
        cands = [r for r in rows
                 if r["changed"] and r["valid_json"]
                 and 4 <= len(r["current_copy"]) <= 60 and len(r["suggested_copy"]) <= 70]
        for r in cands[:AUTO_LIMIT * 4]:
            print(f'    ("{r["path"]}", {r["line"]}),  # {r["current_copy"][:40]!r} -> {r["suggested_copy"][:40]!r}')
        return 0

    picked = [by_key[k] for k in PICKS if k in by_key]
    out = {
        "source": "eval_preds/posthog_review.jsonl (gr33r/ux-writing-sft)",
        "note": "curated subset; picks pinned by (path, line) in scripts/pick_highlights.py",
        "highlights": [
            {"path": r["path"], "line": r["line"], "kind": r["kind"],
             "before": r["current_copy"], "after": r["suggested_copy"], "reason": r["reason"]}
            for r in picked
        ],
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} highlights to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
