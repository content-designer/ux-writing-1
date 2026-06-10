"""Validation helpers for chat-style UX writing datasets.

Supports two task shapes that share one row container:
  - rewrite (modality=text):       assistant JSON = {rewrite, reason, risk?}
  - consistency (modality=vision): assistant JSON = {inventory:[str], issues:[{type,strings,problem,fix}]}

`validate_row` infers the task from the row's `modality`/`task` tags (or, failing
that, from whether any message content is a list of parts) so it works on both the
legacy text rows and the unified dataset rows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROLE_SEQUENCE = ["system", "user", "assistant"]


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def assistant_json(content: str) -> dict[str, Any]:
    """Validate a rewrite-task assistant payload: {rewrite, reason, risk?}."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content must be JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assistant content must decode to an object")
    if not isinstance(parsed.get("rewrite"), str) or not parsed["rewrite"].strip():
        raise ValueError("assistant JSON requires non-empty string field: rewrite")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        raise ValueError("assistant JSON requires non-empty string field: reason")
    if "risk" in parsed and not isinstance(parsed["risk"], str):
        raise ValueError("assistant JSON field risk must be a string")
    return parsed


def assistant_consistency_json(content: str) -> dict[str, Any]:
    """Validate a consistency-task assistant payload.

    Shape: {"inventory": [str, ...], "issues": [{"type", "strings":[str], "problem", "fix"}, ...]}.
    An empty issues list is valid (a clean screen).
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content must be JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assistant content must decode to an object")
    inventory = parsed.get("inventory")
    if not isinstance(inventory, list) or not all(isinstance(s, str) for s in inventory):
        raise ValueError("consistency JSON requires 'inventory' as a list of strings")
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        raise ValueError("consistency JSON requires 'issues' as a list")
    for i, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise ValueError(f"issues[{i}] must be an object")
        if not isinstance(issue.get("type"), str) or not issue["type"].strip():
            raise ValueError(f"issues[{i}] requires non-empty string 'type'")
        strings = issue.get("strings")
        if not isinstance(strings, list) or not all(isinstance(s, str) for s in strings):
            raise ValueError(f"issues[{i}] requires 'strings' as a list of strings")
        for field in ("problem", "fix"):
            if not isinstance(issue.get(field), str) or not issue[field].strip():
                raise ValueError(f"issues[{i}] requires non-empty string '{field}'")
    return parsed


def raw_source_phrases(raw_dir: Path, phrase_words: int = 14) -> set[str]:
    phrases: set[str] = set()
    if not raw_dir.exists():
        return phrases
    for path in raw_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = payload.get("markdown") or payload.get("content") or ""
        words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
        for index in range(0, max(0, len(words) - phrase_words + 1)):
            phrases.add(" ".join(words[index : index + phrase_words]))
    return phrases


def assert_no_long_source_copy(text: str, phrases: set[str]) -> None:
    if not phrases:
        return
    normalized_words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    for index in range(0, max(0, len(normalized_words) - 14 + 1)):
        phrase = " ".join(normalized_words[index : index + 14])
        if phrase in phrases:
            raise ValueError(f"assistant content appears to copy source phrase: {phrase!r}")


def _text_of(content: Any) -> str:
    """Flatten message content (string, or a list of {type,text} parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p["text"] for p in content
                 if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return " ".join(parts)
    return ""


def is_vision_row(row: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    """Decide whether a row is a vision/consistency row (list content) vs text/rewrite."""
    modality = row.get("modality")
    if modality == "vision" or row.get("task") == "consistency":
        return True
    if modality == "text" or row.get("task") == "rewrite":
        return False
    return any(not isinstance(m.get("content"), str) for m in messages if isinstance(m, dict))


def _validate_vision_messages(messages: list[dict[str, Any]], row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    system_content = messages[0].get("content")
    if not isinstance(system_content, str) or not system_content.strip():
        errors.append("vision row: system message requires non-empty string content")

    user_content = messages[1].get("content")
    if not isinstance(user_content, list):
        errors.append("vision row: user content must be a list of parts (image + text)")
    else:
        has_image = any(isinstance(p, dict) and p.get("type") == "image" for p in user_content)
        has_text = any(
            isinstance(p, dict) and p.get("type") == "text"
            and isinstance(p.get("text"), str) and p["text"].strip()
            for p in user_content
        )
        if not has_image:
            errors.append("vision row: user content must include an {'type':'image'} part")
        if not has_text:
            errors.append("vision row: user content must include a non-empty text part")

    assistant_content = messages[2].get("content")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        errors.append("vision row: assistant message requires non-empty string content")

    if not row.get("image"):
        errors.append("vision row: requires a top-level 'image' path")
    return errors


def validate_row(row: dict[str, Any], raw_phrases: set[str], max_tokens: int) -> list[str]:
    errors: list[str] = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        return ["row must include exactly 3 chat messages"]

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ROLE_SEQUENCE:
        errors.append(f"roles must be {ROLE_SEQUENCE}, got {roles}")

    vision = is_vision_row(row, messages)

    if vision:
        errors.extend(_validate_vision_messages(messages, row))
    else:
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"message {index} must be an object")
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"message {index} requires non-empty string content")

    token_count = sum(
        estimate_tokens(_text_of(message.get("content", "")))
        for message in messages if isinstance(message, dict)
    )
    if token_count > max_tokens:
        errors.append(f"estimated token count {token_count} exceeds max {max_tokens}")

    try:
        if vision:
            payload = assistant_consistency_json(messages[2].get("content", ""))
        else:
            payload = assistant_json(messages[2].get("content", ""))
        assert_no_long_source_copy(json.dumps(payload), raw_phrases)
    except ValueError as exc:
        errors.append(str(exc))

    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("row requires provenance object")
    elif vision:
        # real/synthetic vision rows carry license/url or policy ids
        if not (provenance.get("source_policy_ids") or provenance.get("license") or provenance.get("url")):
            errors.append("provenance requires source_policy_ids or license/url")
    elif not provenance.get("source_policy_ids"):
        errors.append("provenance.source_policy_ids must be non-empty")

    return errors


def validate_file(path: Path, raw_dir: Path, max_tokens: int) -> int:
    raw_phrases = raw_source_phrases(raw_dir)
    rows = iter_jsonl(path)
    issue_count = 0
    for index, row in enumerate(rows, 1):
        errors = validate_row(row, raw_phrases, max_tokens)
        for error in errors:
            print(f"{path}:{index}: {error}")
            issue_count += 1
    print(f"validated {len(rows)} rows from {path}; issues={issue_count}")
    return issue_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate UX writing SFT JSONL data.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    return 1 if validate_file(args.path, args.raw_dir, args.max_tokens) else 0


if __name__ == "__main__":
    raise SystemExit(main())
