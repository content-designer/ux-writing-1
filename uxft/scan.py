"""Static extraction of candidate UI strings from frontend codebases."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

UI_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".html", ".json"}
SKIP_DIRS = {
    ".git",
    ".next",
    ".nuxt",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "vendor",
}

STRING_RE = re.compile(
    r"""(?P<prefix>aria-label|title|placeholder|label|alt|text|message|description)?\s*[:=]\s*["'](?P<value>[A-Za-z][^"'\n{}<>]{1,140})["']"""
)
JSX_TEXT_RE = re.compile(r">(?P<value>[A-Za-z][^<>{}\n]{2,140})<")
JSON_VALUE_RE = re.compile(r'"(?P<key>[A-Za-z0-9_.-]+)"\s*:\s*"(?P<value>[A-Za-z][^"\n]{1,140})"')


@dataclass(frozen=True)
class Candidate:
    path: str
    line: int
    kind: str
    current_copy: str
    context: str


def interesting_copy(value: str) -> bool:
    value = " ".join(value.split())
    if len(value) < 3 or len(value) > 140:
        return False
    if not re.search(r"[A-Za-z]", value):
        return False
    if re.fullmatch(r"[A-Z0-9_./:-]+", value):
        return False
    if value.startswith(("http://", "https://", "/", "./", "../")):
        return False
    return True


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in UI_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def context_for(lines: list[str], line_no: int, window: int = 3) -> str:
    start = max(1, line_no - window)
    end = min(len(lines), line_no + window)
    return "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))


def classify(prefix: str | None, value: str, path: Path) -> str:
    lowered = value.lower().strip()
    if prefix in {"aria-label", "alt"}:
        return "accessibility_label"
    if prefix == "placeholder":
        return "form_label"
    if "error" in lowered or "invalid" in lowered or "failed" in lowered:
        return "error"
    if path.suffix.lower() == ".json":
        return "i18n_string"
    if len(value.split()) <= 4:
        return "button_or_label"
    return "body_copy"


def scan_file(path: Path, root: Path) -> list[Candidate]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    candidates: list[Candidate] = []

    patterns = [STRING_RE, JSX_TEXT_RE]
    if path.suffix.lower() == ".json":
        patterns = [JSON_VALUE_RE]

    for pattern in patterns:
        for match in pattern.finditer(text):
            value = " ".join(match.group("value").split())
            if not interesting_copy(value):
                continue
            prefix = match.groupdict().get("prefix") or match.groupdict().get("key")
            line_no = line_number(text, match.start("value"))
            candidates.append(
                Candidate(
                    path=str(path.relative_to(root)),
                    line=line_no,
                    kind=classify(prefix, value, path),
                    current_copy=value,
                    context=context_for(lines, line_no),
                )
            )
    return candidates


def scan_repo(root: Path, limit: int | None = None) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in iter_files(root):
        candidates.extend(scan_file(path, root))
        if limit and len(candidates) >= limit:
            return candidates[:limit]
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract candidate UI strings from a repo.")
    parser.add_argument("repo", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("data/processed/repo_candidates.jsonl"))
    args = parser.parse_args()

    candidates = scan_repo(args.repo.resolve(), args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
    print(f"wrote {len(candidates)} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
