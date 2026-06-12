"""Build docs/demo/posthog_cost_report.json from the Modal run + price snapshot.

Measured cost is real (GPU-seconds x list $/h). Frontier figures are LIST-PRICE
ESTIMATES on OUR measured token counts — clearly labeled, never a quality claim.

    python3 scripts/cost_compare.py            # reads HF artifact + price snapshot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CAVEATS = [
    "Frontier figures are list-price estimates applied to ux-writing-1's measured "
    "prompt/completion token counts for the same workload; they are not measured runs.",
    "Token counts use the Qwen3.6 tokenizer; other providers' tokenizers differ (~±15%).",
    "Estimates assume no hidden reasoning tokens; reasoning-mode APIs bill those as "
    "output, which would raise the frontier cost.",
    "No quality comparison vs frontier models is claimed anywhere; the model's quality "
    "claim is 83% blinded human preference vs its own base model (EVAL_RESULTS.md).",
]


def build_report(run: dict, prices: dict) -> dict:
    ptok, ctok = run["prompt_tokens"], run["completion_tokens"]
    measured = {
        "model": "ux-writing-1 (batched, A100-80GB @ $%.2f/h)" % run["gpu_usd_per_hour"],
        "usd": run["measured_gpu_usd"],
        "strings": run["strings"],
        "strings_per_hour": run["strings_per_hour"],
        "usd_per_1k_strings": run["usd_per_1k_strings"],
        "wall_clock_min": round((run["review_elapsed_s"] + run["model_load_s"]) / 60, 1),
        "valid_json": run["valid_json"], "changed": run["changed"], "kept": run["kept"],
    }
    estimates = []
    for model, p in prices["prices_usd_per_mtok"].items():
        usd = round(ptok / 1e6 * p["input"] + ctok / 1e6 * p["output"], 2)
        estimates.append({
            "model": model, "usd": usd,
            "usd_per_1k_strings": round(usd / run["strings"] * 1000, 2),
            "multiple_vs_measured": round(usd / measured["usd"], 1) if measured["usd"] else None,
            "price_in_per_mtok": p["input"], "price_out_per_mtok": p["output"],
            "source": p["source"],
        })
    return {
        "workload": {"prompt_tokens": ptok, "completion_tokens": ctok,
                     "strings": run["strings"]},
        "measured": measured,
        "estimates_same_tokens_at_list_price": sorted(estimates, key=lambda e: e["usd"]),
        "prices_pulled_at": prices["pulled_at"],
        "caveats": CAVEATS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", type=Path, default=None,
                    help="local posthog_run.json; default: fetch from the dataset repo")
    ap.add_argument("--prices", type=Path, default=Path("docs/demo/llm_api_prices.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/demo/posthog_cost_report.json"))
    args = ap.parse_args()

    if args.run_json:
        run = json.loads(args.run_json.read_text())
    else:
        from huggingface_hub import hf_hub_download
        run = json.loads(Path(hf_hub_download(
            "gr33r/ux-writing-sft", "eval_preds/posthog_run.json",
            repo_type="dataset", force_download=True)).read_text())
    if not run.get("complete"):
        raise SystemExit("run artifact is a partial checkpoint — wait for complete: true")
    report = build_report(run, json.loads(args.prices.read_text()))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
