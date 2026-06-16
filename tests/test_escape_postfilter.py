"""Contract escapes (JSX/ternary as the 'rewrite') must be flagged; copy must not be."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uxft.escape_postfilter import is_contract_escape


def test_escapes_flagged():
    assert is_contract_escape("{category ? 'Update' : 'Create'}")
    assert is_contract_escape("{tagAction === 'add' ? 'Add tags' : 'Replace tags'}")
    assert is_contract_escape("{isNew ? 'Create' : 'Save'}")
    assert is_contract_escape('"<strong>UK GDPR</strong>" means General Data Protection Regulation')


def test_real_copy_not_flagged():
    for ok in ["Save changes", "Invalid API key", "You're on the YC plan", "Other",
               "Search log message for {{ q }}", "Ready? Set: go"]:
        assert not is_contract_escape(ok), ok


def test_ternary_inside_placeholder_flagged():
    # found by the v2 re-run: a ternary inside {{ }} is still an escape, must be flagged
    assert is_contract_escape("Delete {{ currentProject ? currentProject.name : 'the current project' }}")
    assert is_contract_escape("Move {{ currentProject ? currentProject.name : 'the current project' }}")


def test_simple_placeholder_not_flagged():
    assert not is_contract_escape("Delete {{ project.name }}")
    assert not is_contract_escape("Welcome back, {{ user }}")


def test_empty_or_none():
    assert not is_contract_escape("")
    assert not is_contract_escape(None)
