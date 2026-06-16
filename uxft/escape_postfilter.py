"""Deterministic guard: a "rewrite" that is JSX/template code, not prose, is not a valid edit.

The model occasionally returns the surrounding code expression instead of copy — e.g.
`{category ? 'Update' : 'Create'}` for a button label (~2.4% of changes on the PostHog run,
measured in eval/failure_analysis.py). Mirrors uxft.consistency_postfilter: a small pure
function applied at review time. Callers FLAG the suggestion (set risk, drop it from the
change count) rather than delete the row, so it stays auditable.

Legitimate `{{ variable }}` placeholders (which the contract tells the model to preserve)
must NOT be flagged.
"""
from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{\{\s*[\w.]+\s*\}\}")   # legit handlebars var only: {{ user.name }}, NOT {{ a ? b : c }}
_TERNARY = re.compile(r"\{[^{}]*\?[^{}]*:[^{}]*\}")  # {cond ? 'a' : 'b'}
_OPERATOR = re.compile(r"===|!==|=>")                # js operators / arrow fns
_JSX_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")         # <strong>, </>, <Component …>


def is_contract_escape(suggested: str) -> bool:
    """True if `suggested` is JSX/ternary/template code rather than UI copy."""
    s = (suggested or "").strip()
    if not s:
        return False
    # Allow legit {{ variable }} placeholders: remove them before the structural checks.
    stripped = _PLACEHOLDER.sub("", s)
    if _TERNARY.search(stripped) or _OPERATOR.search(stripped) or _JSX_TAG.search(stripped):
        return True
    # A rewrite that is wholly a single brace expression (and not just a {{ variable }}).
    if s.startswith("{") and s.endswith("}") and not re.fullmatch(r"\{\{[^{}]*\}\}", s):
        return True
    return False
