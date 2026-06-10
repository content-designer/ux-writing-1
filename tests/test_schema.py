"""Schema validation tests for both task shapes (rewrite + consistency)."""

import json

from uxft import schema as S
from uxft.policy import SYSTEM_PROMPT


def _text_row(**over):
    row = {
        "task": "rewrite", "modality": "text", "source": "synthetic", "split": "train",
        "image": None,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Content type: button\nCurrent copy: OK\n<Button>OK</Button>"},
            {"role": "assistant", "content": json.dumps(
                {"rewrite": "Save payment method", "reason": "Names the action and object.", "risk": ""})},
        ],
        "metadata": {"category": "button", "product_surface": "billing", "example_type": "derived"},
        "provenance": {"source_policy_ids": ["local-ux-writing-skill"]},
    }
    row.update(over)
    return row


def _vision_row(**over):
    row = {
        "task": "consistency", "modality": "vision", "source": "synthetic", "split": "train",
        "image": "screens/scr_0027.png",
        "messages": [
            {"role": "system", "content": "Screen-level CONSISTENCY review."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Review this screen."}]},
            {"role": "assistant", "content": json.dumps({
                "inventory": ["Save changes", "Save Changes"],
                "issues": [{"type": "casing", "strings": ["Save changes", "Save Changes"],
                            "problem": "Inconsistent casing.", "fix": "Use one casing."}]})},
        ],
        "metadata": {"surface": "settings"},
        "provenance": {"license": "MIT", "url": "https://github.com/example"},
    }
    row.update(over)
    return row


def test_valid_text_row():
    assert S.validate_row(_text_row(), set(), 4096) == []


def test_valid_vision_row():
    assert S.validate_row(_vision_row(), set(), 4096) == []


def test_clean_vision_row_empty_issues():
    row = _vision_row()
    row["messages"][2]["content"] = json.dumps({"inventory": ["Save"], "issues": []})
    assert S.validate_row(row, set(), 4096) == []


def test_text_row_missing_rewrite_fails():
    row = _text_row()
    row["messages"][2]["content"] = json.dumps({"reason": "no rewrite"})
    assert any("rewrite" in e for e in S.validate_row(row, set(), 4096))


def test_vision_row_missing_image_fails():
    row = _vision_row(image=None)
    assert any("image" in e for e in S.validate_row(row, set(), 4096))


def test_vision_row_string_user_content_fails():
    row = _vision_row()
    row["messages"][1]["content"] = "should be a list of parts"
    assert any("list of parts" in e for e in S.validate_row(row, set(), 4096))


def test_vision_issue_missing_fix_fails():
    row = _vision_row()
    row["messages"][2]["content"] = json.dumps({
        "inventory": ["A", "B"],
        "issues": [{"type": "casing", "strings": ["A", "B"], "problem": "x"}]})
    assert any("fix" in e for e in S.validate_row(row, set(), 4096))


def test_modality_inferred_when_untagged():
    row = _vision_row()
    row.pop("task"); row.pop("modality")
    assert S.is_vision_row(row, row["messages"]) is True
