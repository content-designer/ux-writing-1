"""Build derived/synthetic SFT and benchmark data for UX writing rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from uxft.policy import SYSTEM_PROMPT
from uxft.schema import validate_file


@dataclass(frozen=True)
class Scenario:
    category: str
    product_surface: str
    user_state: str
    current: str
    rewrite: str
    reason: str
    risk: str = ""


SCENARIOS = [
    Scenario("button", "billing settings", "ready to save a payment change", "OK", "Save payment method", "Names the action and object so the button works without surrounding context."),
    Scenario("button", "profile editor", "finished editing profile details", "Submit", "Save profile", "Uses a specific action instead of a generic form command."),
    Scenario("button", "file uploader", "choosing a document to upload", "Choose", "Choose file", "Adds the object so the action is clear."),
    Scenario("button", "team invite flow", "reviewing invite details", "Send", "Send invite", "Clarifies what will be sent."),
    Scenario("button", "search filters", "clearing selected filters", "Reset", "Clear filters", "Uses the user's object and a more precise verb."),
    Scenario("form_label", "account signup", "entering contact details", "Your email goes here", "Email address", "Uses a concise noun phrase suitable for a visible field label."),
    Scenario("form_label", "checkout", "entering card details", "Put number", "Card number", "Replaces vague instruction with a clear field label."),
    Scenario("form_label", "workspace creation", "naming a new workspace", "Name it", "Workspace name", "Identifies the exact object being named."),
    Scenario("form_label", "tax profile", "entering a government identifier", "ID", "Tax ID", "Adds necessary context for a high-stakes field."),
    Scenario("form_label", "password setup", "creating credentials", "Secret code", "Password", "Uses familiar terminology that matches user expectations."),
    Scenario("inline_error", "signup form", "recovering from a mistyped email", "Invalid", "Email must include @", "Explains the requirement in a way that works with the field label."),
    Scenario("inline_error", "date picker", "choosing a deadline", "Wrong date", "Choose a future date", "States the fix instead of blaming the input."),
    Scenario("inline_error", "password field", "creating a password", "Bad password", "Password must be at least 8 characters", "Gives a specific requirement and avoids judgmental wording."),
    Scenario("inline_error", "upload form", "attaching a large file", "Not allowed", "File must be under 10 MB", "Provides the constraint needed to recover."),
    Scenario("inline_error", "username field", "choosing an unavailable name", "Error", "Username is already taken", "Explains the problem without dead-end system language."),
    Scenario("system_error", "settings save", "saving changes during a connection issue", "An error occurred while processing your request.", "We couldn't save your changes. Check your connection and try again.", "Names what failed and gives a likely recovery step."),
    Scenario("system_error", "checkout", "payment was declined", "Transaction failed.", "Payment failed. Try a different payment method.", "Keeps the failure specific and action-oriented."),
    Scenario("system_error", "import flow", "CSV schema mismatch", "Upload error.", "Upload failed. Check the column names and try again.", "Tells the user what to inspect next."),
    Scenario("system_error", "session timeout", "returning after inactivity", "Authentication exception.", "Your session expired. Sign in again to continue.", "Uses plain language and a direct next step."),
    Scenario("system_error", "sync status", "cloud sync failed", "Could not complete operation.", "We couldn't sync your changes. Retry when you're back online.", "Connects the failed action to recovery."),
    Scenario("empty_state", "messages inbox", "first time opening an empty inbox", "Nothing to see here.", "No messages yet. Start a conversation to connect with your team.", "Explains why the space is empty and offers a next step."),
    Scenario("empty_state", "saved reports", "no saved reporting views", "Empty", "No saved reports. Save a report to find it here later.", "Makes the state useful instead of merely labeling it."),
    Scenario("empty_state", "search results", "query returned no matches", "No data", "No results found. Try a different keyword or remove filters.", "Adds recovery options for search."),
    Scenario("empty_state", "task list", "new project has no tasks", "You have no stuff.", "No tasks yet. Add a task to start planning work.", "Uses specific product language and a helpful CTA."),
    Scenario("empty_state", "notifications", "no recent notifications", "All clear.", "No new notifications. We'll let you know when something needs attention.", "Sets expectations without adding unnecessary action."),
    Scenario("notification", "security settings", "password changed", "Done", "Password changed", "Confirms the completed action specifically."),
    Scenario("notification", "document sharing", "share email sent", "Success!", "Invitation sent", "Uses a proportional confirmation without vague celebration."),
    Scenario("notification", "billing", "invoice payment completed", "It worked", "Payment processed", "Names the business-critical result."),
    Scenario("notification", "offline mode", "connection dropped", "Network unavailable.", "You're offline. Changes will sync when you're connected.", "Explains impact and what will happen next."),
    Scenario("notification", "autosave", "document saved", "Saved successfully to the system.", "Saved", "Keeps routine confirmation short."),
    Scenario("onboarding", "analytics dashboard", "first visit", "Welcome to our powerful reporting experience.", "Track performance in one place. Connect a data source to get started.", "Focuses on user value and first action."),
    Scenario("onboarding", "bank connection", "cautious about sensitive data", "Give us bank access.", "Connect your bank to see spending insights. We'll guide you through each step.", "Explains the benefit before the permission request."),
    Scenario("onboarding", "team setup", "creating a workspace", "Let's configure everything.", "Invite teammates now or skip this step and add them later.", "Clarifies choice and reduces pressure."),
    Scenario("onboarding", "AI assistant", "trying a new feature", "Start AI", "Ask a question to draft, summarize, or revise content.", "Replaces vague feature language with concrete user actions."),
    Scenario("onboarding", "mobile permissions", "enabling notifications", "Allow notifications.", "Get updates when orders ship. Enable notifications.", "Gives the benefit before asking for permission."),
    Scenario("destructive_confirmation", "account settings", "deleting account", "Are you sure?", "Delete account? You'll lose all data and this can't be undone.", "States the consequence for a high-stakes action."),
    Scenario("destructive_confirmation", "project settings", "removing a project", "Remove?", "Delete project? Team members will lose access immediately.", "Makes the destructive action and impact explicit."),
    Scenario("destructive_confirmation", "billing", "canceling subscription", "Cancel now?", "Cancel subscription? Your team keeps access until the billing period ends.", "Clarifies timing and consequence."),
    Scenario("destructive_confirmation", "files", "deleting a shared file", "Proceed", "Delete shared file? People with access won't be able to open it.", "Avoids vague confirmation copy for an irreversible action."),
    Scenario("destructive_confirmation", "admin panel", "revoking access", "Confirm", "Revoke access? This person will be signed out immediately.", "Names the action and immediate effect."),
    Scenario("accessibility_label", "icon-only search control", "using screen reader navigation", "Go", "Search", "Uses a descriptive accessible name for the icon button."),
    Scenario("accessibility_label", "external link", "navigating a settings page", "Click here", "Read privacy policy", "Makes link purpose clear outside visual context."),
    Scenario("accessibility_label", "close icon", "closing a modal dialog", "X", "Close dialog", "Gives the icon-only control a meaningful label."),
    Scenario("accessibility_label", "table row action", "viewing invoice details", "View", "View invoice details", "Adds the object so repeated row actions are distinguishable."),
    Scenario("accessibility_label", "download icon", "downloading a report", "Download", "Download monthly report", "Adds context for assistive technology users."),
]

SURFACE_VARIANTS = [
    ("consumer mobile app", "general audience", "Keep the rewrite short enough for mobile."),
    ("B2B admin dashboard", "busy professional user", "Use direct professional language."),
    ("developer tool", "technical user", "Keep technical terms only when the context already uses them."),
    ("financial product", "cautious user", "Avoid ambiguity around money or account impact."),
    ("healthcare portal", "stressed patient or caregiver", "Use calm language and avoid blame."),
    ("ecommerce checkout", "shopper completing a purchase", "Keep payment and delivery impact clear."),
    ("education platform", "student or instructor", "Use supportive language without sounding childish."),
    ("public service website", "broad public audience", "Use plain language and avoid idioms."),
]

POLICY_SCENARIOS = {
    "ai_trust_control": [
        Scenario("onboarding", "AI assistant", "deciding whether to use a generated draft", "Generate it", "Generate draft", "Names the AI action without overstating the quality of the result.", "Generated content may need user review."),
        Scenario("notification", "AI summary", "reviewing an AI-created summary", "Summary ready", "AI summary ready to review", "Makes AI involvement visible and sets a review expectation."),
        Scenario("button", "AI rewrite tool", "choosing whether to accept a suggestion", "Use this", "Apply suggestion", "Gives the user a clear control over the generated output."),
    ],
    "brand_voice_adaptation": [
        Scenario("system_error", "billing settings", "failed save during a stressful account task", "Oopsie! We hit a snag.", "We couldn't save your billing changes. Try again.", "Uses calm, direct language for a sensitive moment."),
        Scenario("onboarding", "setup checklist", "new user learning a product", "Configure your operational parameters.", "Set up your workspace in a few steps.", "Adapts tone for learning with plain, helpful wording."),
    ],
    "financial_tax_clarity": [
        Scenario("destructive_confirmation", "tax filing", "removing a tax form", "Remove this?", "Remove tax form? This may change your filing totals.", "Explains the financial consequence before the user confirms."),
        Scenario("system_error", "invoice payment", "payment did not complete", "There was a problem.", "Payment didn't go through. Check the payment method and try again.", "Keeps money-related recovery specific and actionable."),
        Scenario("form_label", "business profile", "entering a government business number", "Number", "Employer identification number", "Uses a precise financial identifier instead of a vague label."),
    ],
    "formatting_minimalism": [
        Scenario("notification", "status banner", "reading an emphasized warning", "IMPORTANT: PLEASE READ THIS NOW!!!", "Review this before you continue", "Uses wording instead of visual shouting to signal importance."),
        Scenario("form_label", "tax profile", "entering an abbreviation-heavy field", "EIN", "Employer identification number (EIN)", "Spells out the term before using the abbreviation."),
    ],
    "help_content": [
        Scenario("empty_state", "help search", "searching support articles with no results", "No articles.", "No help articles found. Try fewer keywords or browse all topics.", "Gives a recovery path for a blocked help task."),
        Scenario("button", "support article", "trying to resolve an issue", "More", "Read troubleshooting steps", "Makes the link destination and user task clear."),
    ],
    "precision_consistency": [
        Scenario("button", "accounting dashboard", "exporting a report", "Get data", "Export report", "Uses the exact object and action instead of a vague synonym."),
        Scenario("inline_error", "business name field", "leaving a required value blank", "Missing", "Business name is required", "Uses the same field term in the error message."),
    ],
    "conversational_plain_language": [
        Scenario("system_error", "settings page", "routine save failed", "The requested operation could not be completed.", "We couldn't save your changes. Try again.", "Replaces system-centered phrasing with direct, conversational copy."),
        Scenario("button", "checkout", "starting purchase flow", "Initiate purchase process", "Buy now", "Uses a familiar action label that fits the user's goal."),
    ],
    "grammar_mechanics": [
        Scenario("button", "settings page", "saving preference changes", "Save Changes", "Save changes", "Uses sentence case for a standard UI action."),
        Scenario("notification", "account page", "routine update completed", "Your Profile Has Been Updated!", "Profile updated", "Keeps capitalization and punctuation quiet for a routine confirmation."),
    ],
    "inclusive_accessible_language": [
        Scenario("accessibility_label", "help center", "opening a support resource", "Click here", "Contact support", "Makes the link purpose clear without relying on surrounding visual text."),
        Scenario("empty_state", "profile setup", "user has not added optional demographic data", "You haven't told us who you are.", "Profile details not added yet. Add them when you're ready.", "Avoids assumptions and pressure around personal information."),
    ],
    "commerce_action_clarity": [
        Scenario("destructive_confirmation", "product admin", "archiving a product", "Archive?", "Archive product? Customers won't be able to buy it.", "Explains the commerce impact before confirming."),
        Scenario("system_error", "checkout", "cart update failed", "Couldn't update.", "We couldn't update the cart. Check item availability and try again.", "Connects the failed action to a likely shopper recovery step."),
        Scenario("button", "order admin", "fulfilling an order", "Do it", "Fulfill order", "Uses an explicit merchant action and object."),
    ],
    "error_recovery": [
        Scenario("system_error", "file export", "export failed after a long-running task", "Something went wrong.", "We couldn't export the file. Check your connection and try again.", "Explains what failed and gives a recovery step."),
        Scenario("inline_error", "phone number field", "entering contact details", "Invalid entry", "Enter a valid phone number", "Replaces blame-oriented copy with a specific correction."),
        Scenario("system_error", "permissions flow", "blocked by missing access", "Access error.", "You don't have access to this workspace. Ask an admin for permission.", "Explains the blocker and the next person to contact."),
    ],
    "onboarding_guidance": [
        Scenario("onboarding", "new workspace setup", "first time creating a workspace", "Set everything up now.", "Start with your workspace name. You can add details later.", "Keeps onboarding focused and reduces setup pressure."),
        Scenario("button", "onboarding checklist", "skipping an optional setup step", "No", "Skip for now", "Makes the deferred action clear and low-stakes."),
        Scenario("empty_state", "first-use dashboard", "no data connected yet", "No dashboard.", "Connect a data source to see your first dashboard.", "Points to the next useful onboarding action."),
    ],
    "visual_context_independence": [
        Scenario("accessibility_label", "image upload preview", "screen reader user reviewing an uploaded image", "Image", "Preview of uploaded profile photo", "Describes the visual's purpose instead of its file type."),
        Scenario("accessibility_label", "status icon", "checking whether sync completed", "Green check", "Sync complete", "Uses the status meaning rather than relying on color."),
        Scenario("button", "icon-only toolbar", "deleting a draft", "Trash", "Delete draft", "Names the action represented by the icon."),
    ],
    "readability": [
        Scenario("system_error", "account recovery", "reading a stressful account message", "Due to an authentication anomaly, your session has been terminated and reauthentication is required.", "Your session expired. Sign in again to continue.", "Shortens dense system language into scannable recovery copy."),
        Scenario("onboarding", "analytics setup", "learning a new dashboard", "In order to begin leveraging insights, configure an initial integration.", "Connect a data source to see insights.", "Front-loads the concrete action and user benefit."),
        Scenario("notification", "bulk import", "import completed with partial results", "The import operation has completed with exceptions.", "Import complete. Some rows need review.", "Makes the result easier to scan and act on."),
    ],
    "ui_text_action_labels": [
        Scenario("button", "issue tracker", "creating a new issue", "New", "Create issue", "Names the action and object without relying on nearby layout."),
        Scenario("button", "merge request", "approving a code change", "OK", "Approve merge request", "Uses the specific action the interface will perform."),
        Scenario("accessibility_label", "sidebar navigation", "opening project settings", "Settings", "Open project settings", "Adds useful context for repeated navigation labels."),
    ],
    "date_time_clarity": [
        Scenario("notification", "scheduled report", "report is scheduled across time zones", "Runs 03/04 at 9.", "Runs April 3 at 9:00 AM ET", "Avoids ambiguous compact date and missing time zone."),
        Scenario("inline_error", "event date field", "choosing an event start time", "Invalid time", "Enter a time after 9:00 AM", "Gives the date/time constraint needed to fix the field."),
        Scenario("system_error", "booking flow", "selected time slot expired", "Time unavailable.", "This time slot is no longer available. Choose another time.", "Explains the scheduling issue and recovery action."),
    ],
    "public_service_plain_language": [
        Scenario("button", "benefits application", "submitting a public-service form", "Proceed with submission", "Submit application", "Uses direct public-service language for the task."),
        Scenario("system_error", "government account", "identity verification failed", "Verification exception.", "We couldn't verify your identity. Check your details and try again.", "Replaces institutional wording with a clear next step."),
        Scenario("empty_state", "public records search", "search returned no matches", "No records.", "No records found. Check the spelling or try fewer keywords.", "Helps a broad public audience recover from no results."),
    ],
    "platform_style_consistency": [
        Scenario("button", "desktop settings", "turning on a system preference", "Activate functionality", "Turn on notifications", "Uses familiar platform wording for a setting action."),
        Scenario("form_label", "developer console", "naming an API key", "Thing name", "API key name", "Uses the product term consistently with the technical object."),
        Scenario("notification", "file sync", "cloud file has updated", "Object synchronization finalized.", "File synced", "Replaces system wording with a platform-consistent status."),
    ],
    "global_audience_localization": [
        Scenario("notification", "international checkout", "delivery estimate crosses locales", "Your order arrives 03/04.", "Your order arrives April 3.", "Avoids compact date ambiguity for a global audience."),
        Scenario("empty_state", "translated app", "first-use dashboard", "Let's get the ball rolling.", "Add your first project to get started.", "Replaces an idiom with literal, translatable language."),
        Scenario("system_error", "address form", "postal code is missing", "Zip is bad.", "Enter a postal code", "Uses locale-neutral wording and avoids blame."),
    ],
}


def stable_id(parts: list[str]) -> str:
    return str(uuid5(NAMESPACE_URL, "::".join(parts)))


def code_context(scenario: Scenario, surface: str) -> str:
    safe_current = scenario.current.replace('"', '\\"')
    return (
        f"// Surface: {surface}\n"
        f"// Product area: {scenario.product_surface}\n"
        f'<Button aria-label="{safe_current}">{safe_current}</Button>'
    )


def user_prompt(scenario: Scenario, surface: str, audience: str, constraint: str) -> str:
    return (
        f"Product surface: {surface}\n"
        f"Audience: {audience}\n"
        f"User state: {scenario.user_state}\n"
        f"Content type: {scenario.category}\n"
        f"Current copy: {scenario.current}\n"
        f"Code/context:\n{code_context(scenario, surface)}\n"
        f"Constraints: {constraint} Preserve the intended product behavior."
    )


def row_for(
    scenario: Scenario,
    variant: tuple[str, str, str],
    source_policy_ids: list[str] | None = None,
    example_type: str = "derived_synthetic_rewrite",
    generation_method: str = "hand-authored derived scenario template",
    copyright_posture: str = "derived_only_no_raw_guide_text",
) -> dict:
    surface, audience, constraint = variant
    assistant = {
        "rewrite": scenario.rewrite,
        "reason": scenario.reason,
        "risk": scenario.risk,
    }
    return {
        "id": stable_id([scenario.category, scenario.current, surface]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(scenario, surface, audience, constraint)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "metadata": {
            "category": scenario.category,
            "product_surface": scenario.product_surface,
            "example_type": example_type,
        },
        "provenance": {
            "source_policy_ids": source_policy_ids
            or [
                "local-ux-writing-skill",
                "intuit-basics",
                "intuit-principles",
                "atlassian-content",
                "microsoft-fluent-content",
                "shopify-polaris-foundations",
            ],
            "generation_method": generation_method,
            "copyright_posture": copyright_posture,
        },
    }


def load_policy_scenarios(policy_path: Path | None) -> list[dict]:
    if not policy_path or not policy_path.exists():
        return []
    artifact = json.loads(policy_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for rule in artifact.get("derived_rules", []):
        theme = rule.get("theme")
        scenarios = POLICY_SCENARIOS.get(theme, [])
        source_policy_ids = ["local-ux-writing-skill", *rule.get("source_policy_ids", [])]
        for scenario in scenarios:
            for variant in SURFACE_VARIANTS:
                rows.append(
                    row_for(
                        scenario,
                        variant,
                        source_policy_ids=source_policy_ids,
                        example_type="public_policy_derived_synthetic_rewrite",
                        generation_method=f"derived from non-verbatim policy theme: {theme}",
                    )
                )
    return rows


def load_repo_candidates(path: Path | None, limit: int) -> list[dict]:
    if not path or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def load_extra_rows(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_from_candidate(candidate: dict, index: int) -> dict:
    current = str(candidate["current_copy"])
    prompt = (
        "Product surface: existing codebase\n"
        "Audience: product user\n"
        f"User state: using the screen that contains {candidate['path']}:{candidate['line']}\n"
        f"Content type: {candidate.get('kind', 'ui_string')}\n"
        f"Current copy: {current}\n"
        f"Code/context:\n{candidate.get('context', '')}\n"
        "Constraints: Suggest a UX writing rewrite only if the context supports it."
    )
    assistant = {
        "rewrite": current,
        "reason": "Needs human review; this seed row preserves the current copy until a reviewed rewrite is added.",
        "risk": "Do not train on unreviewed repo-derived rows without replacing this placeholder rewrite.",
    }
    return {
        "id": stable_id(["repo-candidate", candidate["path"], str(candidate["line"]), str(index)]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "metadata": {
            "category": candidate.get("kind", "ui_string"),
            "product_surface": "existing_codebase",
            "example_type": "repo_candidate_needs_review",
        },
        "provenance": {
            "source_policy_ids": ["local-ux-writing-skill"],
            "generation_method": "static repo scan placeholder",
            "copyright_posture": "private_repo_context_review_required",
        },
    }


def build_rows(
    repo_candidates: Path | None = None,
    policy_path: Path | None = None,
    extra_rows: Path | None = None,
) -> list[dict]:
    from uxft.synthetic_extra import EXTRA_SCENARIOS  # lazy import avoids a circular dependency
    seed_scenarios = SCENARIOS + EXTRA_SCENARIOS
    rows = [row_for(scenario, variant) for scenario in seed_scenarios for variant in SURFACE_VARIANTS]
    rows.extend(load_policy_scenarios(policy_path))
    rows.extend(load_extra_rows(extra_rows))
    rows.extend(row_from_candidate(candidate, i) for i, candidate in enumerate(load_repo_candidates(repo_candidates, 100)))
    return rows


def input_key(row: dict) -> tuple[str, str]:
    """Identity of a training *input*: (category, current copy). Rows that share this
    key are the same UI string under different surface framings — i.e. duplicates of
    the same input that all carry the identical target rewrite."""
    user = row["messages"][1]["content"]
    current = ""
    if "Current copy:" in user:
        current = user.split("Current copy:", 1)[1].split("\n", 1)[0].strip()
    return (row["metadata"].get("category", ""), current)


def split_dedup(rows: list[dict], eval_size: int, max_per_input: int) -> tuple[list[dict], list[dict]]:
    """Build a held-out benchmark and a de-duplicated train set.

    - Groups rows by input_key so the same input never appears in both splits.
    - Reserves `eval_size` distinct inputs (one row each) for the benchmark; those
      inputs are EXCLUDED from train, so eval measures generalization to unseen copy.
    - Caps train to `max_per_input` rows per input, cutting the context-invariant
      surface duplication (was up to 8x) that drives memorization.
    Selection is deterministic (hash of the key), not positional, so the split is
    stable and roughly stratified across categories.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[input_key(row)].append(row)

    # Only inputs with real copy are benchmark-eligible (skips placeholder rows).
    keys = sorted((k for k in groups if k[1]), key=lambda k: hashlib.sha1("::".join(k).encode()).hexdigest())
    held = set(keys[:eval_size])

    eval_rows = [groups[k][0] for k in keys[:eval_size]]
    train_rows: list[dict] = []
    for key, group in groups.items():
        if key in held:
            continue
        train_rows.extend(group[:max_per_input])
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build UX writing SFT and benchmark data.")
    parser.add_argument("--repo-candidates", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=Path("data/processed/policy/public_derived.json"))
    parser.add_argument("--extra-rows", type=Path, default=Path("data/private_sources/processed/private_course_rows.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/eval/benchmark.jsonl"))
    parser.add_argument("--eval-size", type=int, default=60,
                        help="Distinct held-out inputs reserved for the benchmark (excluded from train).")
    parser.add_argument("--max-per-input", type=int, default=3,
                        help="Cap on training rows per input; trims the surface-variant duplication.")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args.repo_candidates, args.policy, args.extra_rows)
    train_rows, eval_rows = split_dedup(rows, args.eval_size, args.max_per_input)

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    print(f"built {len(rows)} candidate rows")
    print(f"wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"wrote {len(eval_rows)} held-out benchmark rows to {args.eval_out}")

    if args.validate:
        return max(
            validate_file(args.train_out, Path("data/raw"), 2048),
            validate_file(args.eval_out, Path("data/raw"), 2048),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
