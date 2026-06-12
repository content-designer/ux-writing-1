"""Guards ported from ux-writing-bench: oversized files skipped, degenerate contexts dropped."""
from pathlib import Path

from uxft.scan import MAX_CONTEXT_CHARS, MAX_FILE_BYTES, scan_repo


def _write(root: Path, name: str, text: str) -> Path:
    p = root / name
    p.write_text(text, encoding="utf-8")
    return p


def test_oversized_file_is_skipped(tmp_path):
    _write(tmp_path, "ok.ts", 'const a = { label: "Save your changes" };\n')
    big = 'const x = { label: "Minified bundle string" };' + ("x" * (MAX_FILE_BYTES + 100))
    _write(tmp_path, "bundle.min.js", big)
    found = scan_repo(tmp_path)
    assert any(c.path == "ok.ts" for c in found)
    assert not any(c.path == "bundle.min.js" for c in found)


def test_degenerate_context_is_dropped(tmp_path):
    # one enormous single line -> context window exceeds MAX_CONTEXT_CHARS
    pad = "y" * (MAX_CONTEXT_CHARS + 500)
    _write(tmp_path, "oneline.ts", f'const q = {{ label: "Delete this workspace" }}; // {pad}\n')
    assert scan_repo(tmp_path) == []
