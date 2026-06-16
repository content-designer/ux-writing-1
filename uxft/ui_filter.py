"""Heuristics for deciding whether an extracted string is real UI copy.

`looks_like_ui_copy` / `NON_UI_PATH` / `CSS_TOKEN` were promoted here from
scripts/sample_posthog_candidates.py, where they only ran at sampling time — *after* the
model had already reviewed everything. Hosting them here lets uxft.scan apply them up
front. `is_probably_non_ui` adds context-aware rejects for the non-UI patterns the PostHog
run surfaced (enum/const values, type annotations, colour-map names, logging/action-type
keys) — it reads the matched source line, so single-word labels like "Save"/"Delete"
survive while "Wizard = 'Wizard'" or a colour constant does not.
"""
from __future__ import annotations

import re

# Path patterns that are never product UI copy (tests, mocks, fixtures, generated code).
NON_UI_PATH = re.compile(
    r"(\.test\.|\.spec\.|\.stories\.|__mocks__|/test/|/tests/|cypress|/e2e/|\.cy\."
    r"|mocks?\.tsx?$|fixtures|\.schemas?\.ts$|generated|schema\.json$)",
    re.I,
)
# css-ish token: lowercase AND contains a structural char (hyphen/digit/colon/bracket/slash/dot)
CSS_TOKEN = re.compile(r"^(?=.*[-0-9:\[\]/.%#])[a-z0-9:\[\]/.%#-]+$")

_PRIMITIVE_TYPES = {
    "string", "number", "boolean", "object", "array", "null", "any", "void",
    "integer", "float", "bigint", "symbol", "undefined", "unknown", "never",
}


def looks_like_ui_copy(value: str) -> bool:
    """Shape-only check (promoted verbatim): reject identifier-ish / CSS-ish strings."""
    if "_" in value:
        return False
    words = value.split()
    if len(words) >= 2:
        css_ish = sum(1 for w in words if CSS_TOKEN.fullmatch(w))
        return css_ish / len(words) < 0.5
    return bool(re.fullmatch(r"[A-Z][a-z]{2,}", words[0]))


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def is_probably_non_ui(value: str, line_text: str, path: str = "") -> bool:
    """True when the matched source line shows this string is not user-facing copy.

    Conservative — fires only on a clear structural signal, so real single-word labels
    survive. `line_text` is the source line the string was extracted from.
    """
    v = _norm(value)
    line = line_text or ""
    low = v.lower()

    # enum / const value: `Wizard = 'Wizard'` (key == value) or a SCREAMING_SNAKE const
    # like `SQL_EDITOR_EMPTY_STATE = 'sql editor empty state'`.
    m = re.search(r"(?:^|[\s,{(\[])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*['\"]"
                  + re.escape(v) + r"['\"]", line)
    if m:
        key = m.group("key")
        if re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", key):  # SCREAMING_SNAKE
            return True
        if key.lower() == low:  # enum member whose value equals its name
            return True

    # type annotation: `"type": "String"`, `type: 'string'`
    if low in _PRIMITIVE_TYPES and re.search(
        r"['\"]?type['\"]?\s*:\s*['\"]" + re.escape(v) + r"['\"]", line
    ):
        return True

    # colour / token map: the line binds a hex code to a `name:` (e.g. `'#6B8E23': { name: 'OliveDrab' }`)
    if re.search(r"#[0-9a-fA-F]{3,8}\b", line) and re.search(r"\bname\b\s*:", line):
        return True

    # logging / action-type key: trailing parenthesised module path
    # (e.g. `load prompts success (scenes.ai-observability.llmPromptsLogic)`).
    if re.search(r"\([\w.\-]+\)\s*$", v):
        return True

    return False


def is_ui_copy(value: str, line_text: str = "", path: str = "") -> bool:
    """Scanner-time gate: keep only shape-plausible copy that isn't a known non-UI form."""
    if path and NON_UI_PATH.search(path):
        return False
    if not looks_like_ui_copy(value):
        return False
    if is_probably_non_ui(value, line_text, path):
        return False
    return True
