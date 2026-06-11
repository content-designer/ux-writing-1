"""extract_contract_json: thinking-tolerant, and must parse rewrites that keep
{{ variables }} — the system prompt demands they be preserved, so the parser
can never choke on braces inside string values (the old flat {…} regex did,
and placeholder-bearing reviews were mis-flagged as non-JSON output)."""

from uxft.review_repo import extract_contract_json


def test_placeholder_rewrite_parses():
    raw = '{"rewrite": "{{count}} files ready", "reason": "kept the variable", "risk": ""}'
    parsed = extract_contract_json(raw)
    assert parsed is not None
    assert parsed["rewrite"] == "{{count}} files ready"


def test_brace_and_printf_placeholders_parse():
    raw = '{"rewrite": "Saved {0} of %s drafts", "reason": "r", "risk": ""}'
    assert extract_contract_json(raw)["rewrite"] == "Saved {0} of %s drafts"


def test_thinking_prefix_takes_text_after_last_think():
    raw = (
        '<think>draft: {"rewrite": "wrong draft", "reason": "x", "risk": ""}</think>\n'
        '{"rewrite": "Delete project?", "reason": "states consequence", "risk": ""}'
    )
    assert extract_contract_json(raw)["rewrite"] == "Delete project?"


def test_last_parseable_object_wins():
    raw = '{"rewrite": "first", "reason": "a", "risk": ""} then {"rewrite": "second", "reason": "b", "risk": ""}'
    assert extract_contract_json(raw)["rewrite"] == "second"


def test_garbage_and_nonstring_rewrite_return_none():
    assert extract_contract_json("no json here") is None
    assert extract_contract_json('{"rewrite": 42}') is None
    assert extract_contract_json(None) is None


def test_object_with_nested_value_parses():
    raw = 'note {"meta": {"x": 1}} answer {"rewrite": "Try again", "reason": "r", "risk": ""}'
    assert extract_contract_json(raw)["rewrite"] == "Try again"
