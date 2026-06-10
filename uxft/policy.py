"""Derived UX writing policy used to generate and evaluate training rows."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a senior UX writer reviewing interface copy in product code.
Rewrite the UI copy so it is purposeful, concise, conversational, clear, and accessible.
If the current copy is already clear, accurate, and on-brand, keep it unchanged: return it verbatim as the rewrite and say so in the reason.
Preserve product intent. Do not invent actions, facts, or product behavior that are not in the context.
Keep locale-specific terms (for example, "Postal code" for Canadian addresses) and any {{ variables }} exactly as written.
Never weaken safety-critical copy: destructive, payment, privacy, and security messages must keep their consequence and must not be softened.
Return compact JSON with: rewrite, reason, and risk. Use an empty string for risk when none applies."""

QUALITY_STANDARDS = {
    "purposeful": [
        "Support the user's immediate task.",
        "Make the user benefit or next step clear.",
        "Avoid adding copy that compensates for a broken interaction.",
    ],
    "concise": [
        "Remove filler and redundant words.",
        "Front-load the action or problem.",
        "Keep UI strings short enough for compact layouts.",
    ],
    "conversational": [
        "Use natural wording and active voice.",
        "Prefer common words over formal or corporate wording.",
        "Match tone to stakes and user emotion.",
    ],
    "clear": [
        "Use specific verbs such as save, delete, cancel, or retry.",
        "Use consistent terminology for the same concept.",
        "Explain recovery steps in errors.",
    ],
    "accessible": [
        "Use descriptive button and link labels.",
        "Avoid placeholder-only instructions.",
        "Do not rely on color, position, or visual context alone.",
        "Make error text understandable with its field label.",
    ],
}

ISSUE_TAXONOMY = {
    "generic_action": "Button or link text is too generic to stand alone.",
    "vague_error": "Error text does not explain what failed or how to recover.",
    "blame_language": "Copy blames the user or uses system-centered language.",
    "missing_context": "Label or title omits the object or outcome.",
    "inaccessible_link": "Link text depends on surrounding visual context.",
    "high_stakes_ambiguity": "Destructive or financial action is not explicit enough.",
    "wordy_copy": "Copy uses extra words without adding meaning.",
    "terminology_drift": "Copy uses inconsistent names for the same product concept.",
}

CONTENT_TYPES = [
    "button",
    "form_label",
    "inline_error",
    "system_error",
    "empty_state",
    "notification",
    "onboarding",
    "destructive_confirmation",
    "accessibility_label",
]


def compact_policy_text() -> str:
    """Return a compact rubric for prompts and dataset metadata."""
    lines: list[str] = []
    for standard, rules in QUALITY_STANDARDS.items():
        joined = "; ".join(rules)
        lines.append(f"{standard}: {joined}")
    return "\n".join(lines)

