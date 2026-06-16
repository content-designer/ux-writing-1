"""Non-UI strings must be rejected at scan time; real copy must survive.

Fixtures are the exact (value, source-line) pairs from the PostHog failure sample —
the non-UI strings the model "improved" (color constants, enums, type strings,
logging keys) and a basket of genuine copy that must not be dropped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.ui_filter import is_probably_non_ui, is_ui_copy

PATH = "frontend/src/x.tsx"

REJECT = [
    ("sql editor empty state", "    SQL_EDITOR_EMPTY_STATE = 'sql editor empty state',"),
    ("Wizard", "    Wizard = 'Wizard',"),
    ("new query started", "const NEW_QUERY_STARTED_ERROR_MESSAGE = 'new query started' as const"),
    ("Green", "    '#6B8E23': { name: 'OliveDrab', group: 'Green' },"),
    ("String", '      "type": "String",'),
    ("load prompts success (scenes.ai-observability.llmPromptsLogic)",
     '    type: "load prompts success (scenes.ai-observability.llmPromptsLogic)";'),
]

KEEP = [
    ("Save changes", "  children: 'Save changes',"),
    ("Delete", "  children: 'Delete',"),
    ("Invalid JSON", "      errors.globals = 'Invalid JSON'"),
    ("MCP is error", '      "label": "MCP is error"'),
    ("Bracket By", "          <div>Bracket By</div>"),
    ("must be string", "        message: 'must be string',"),
]


def test_non_ui_rejected():
    for value, line in REJECT:
        assert is_probably_non_ui(value, line), value
        assert not is_ui_copy(value, line, PATH), value


def test_real_copy_kept():
    for value, line in KEEP:
        assert not is_probably_non_ui(value, line), value
        assert is_ui_copy(value, line, PATH), value


def test_context_gates_ambiguous_word():
    # "Green" is a colour in a colour-map line, but real copy as a standalone label.
    assert is_probably_non_ui("Green", "'#008000': { name: 'Green' }")
    assert not is_probably_non_ui("Green", "<Tag>Green</Tag>")
    assert is_ui_copy("Green", "<Tag>Green</Tag>", PATH)
