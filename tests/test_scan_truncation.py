"""STRING_RE must capture full literals, not truncate at the first apostrophe/quote.

Regression for the scanner bug behind ~18% of the PostHog "changes": `"Don't worry…"`
used to scan as `Don`, so the model "rebuilt" a string that was never really there.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.scan import scan_repo


def _scan(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return {c.current_copy for c in scan_repo(tmp_path)}


def test_contraction_not_truncated(tmp_path):
    vals = _scan(tmp_path, "a.tsx",
                 'const x = { description: "Don\'t worry about backups - we\'ve got it covered." };\n')
    assert any(v.startswith("Don't worry about backups") for v in vals), vals
    assert "Don" not in vals


def test_isnt_not_truncated(tmp_path):
    vals = _scan(tmp_path, "b.tsx",
                 'const m = { skipWarning: "Error tracking isn\'t enabled by default." };\n')
    assert "Error tracking isn't enabled by default." in vals, vals
    assert "Error tracking isn" not in vals


def test_opposite_quote_inside_is_captured(tmp_path):
    vals = _scan(tmp_path, "c.tsx", 'const t = { title: \'He said "hello" to me\' };\n')
    assert any(v.startswith("He said") and "hello" in v for v in vals), vals


def test_escaped_quote_does_not_terminate(tmp_path):
    # JS-escaped apostrophe inside a single-quoted literal
    vals = _scan(tmp_path, "e.tsx", "const t = { label: 'You don\\'t have access here' };\n")
    assert any(v.startswith("You don") and len(v) > len("You don") for v in vals), vals


def test_brace_still_terminates(tmp_path):
    # we must NOT start swallowing JSX/template braces into a copy string
    vals = _scan(tmp_path, "d.tsx", 'const z = { label: "Hello {name} welcome" };\n')
    assert not any("{name}" in v for v in vals), vals


def test_no_catastrophic_backtracking(tmp_path):
    # an unterminated quote followed by a long run must not hang the bounded regex
    (tmp_path / "big.tsx").write_text('const q = { label: "' + "a" * 20_000 + "\n", encoding="utf-8")
    start = time.time()
    scan_repo(tmp_path)
    assert time.time() - start < 2.0
