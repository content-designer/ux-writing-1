import json

from scripts.cost_compare import build_report

RUN = {"strings": 10000, "review_elapsed_s": 4500.0, "model_load_s": 300.0,
       "prompt_tokens": 5_000_000, "completion_tokens": 350_000,
       "strings_per_hour": 8000, "usd_per_1k_strings": 0.31,
       "measured_gpu_usd": 3.33, "gpu_usd_per_hour": 2.50,
       "valid_json": 9990, "changed": 700, "kept": 9300, "complete": True}
PRICES = {"pulled_at": "2026-06-12", "collector": "test",
          "prices_usd_per_mtok": {
              "gpt-5.5": {"input": 5.0, "output": 30.0, "source": "s1"},
              "claude-opus-4.8": {"input": 5.0, "output": 25.0, "source": "s2"}}}


def test_build_report_math():
    rep = build_report(RUN, PRICES)
    est = {e["model"]: e for e in rep["estimates_same_tokens_at_list_price"]}
    # 5M in * $5/M + 0.35M out * $30/M = 25 + 10.5 = 35.5
    assert est["gpt-5.5"]["usd"] == 35.5
    # 5M * 5 + 0.35M * 25 = 25 + 8.75 = 33.75
    assert est["claude-opus-4.8"]["usd"] == 33.75
    assert rep["measured"]["usd"] == 3.33
    assert est["gpt-5.5"]["multiple_vs_measured"] == round(35.5 / 3.33, 1)
    assert rep["caveats"]  # honesty rails always present


def test_report_is_json_serializable_and_sorted():
    rep = build_report(RUN, PRICES)
    json.dumps(rep)
    costs = [e["usd"] for e in rep["estimates_same_tokens_at_list_price"]]
    assert costs == sorted(costs)
