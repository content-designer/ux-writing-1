"""⛺ Copy Campfire — the UX writing arena.

Two writers answer the same brief by the fire: one is base Qwen3.6-27B, one is
ux-writing-1 (a LoRA fine-tune for UX writing). Sides are anonymized and randomized;
you vote before the reveal. Votes are stored (privately) as preference data that
trains the next version.

Env (Space secrets): BATTLE_URL, AUTH_TOKEN, HF_TOKEN; optional VOTES_DATASET.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import gradio as gr
import requests
from huggingface_hub import HfApi

BATTLE_URL = os.environ.get("BATTLE_URL", "")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
VOTES_DATASET = os.environ.get("VOTES_DATASET", "gr33r/ux-writing-arena-votes")
FINETUNE_NAME = "ux-writing-1 (fine-tune)"
BASE_NAME = "Qwen3.6-27B (base)"

CATEGORIES = [
    "button", "inline_error", "system_error", "empty_state", "notification",
    "onboarding", "destructive_confirmation", "accessibility_label", "form_label",
    "tooltip", "body_copy",
]

# Curated brief deck (131 prompts) shipped alongside the app.
try:
    with open(os.path.join(os.path.dirname(__file__), "battle_corpus.json"), encoding="utf-8") as fh:
        CORPUS = json.load(fh)
except OSError:
    CORPUS = [{"category": "button", "surface": "billing settings", "current": "OK"}]

api = HfApi(token=os.environ.get("HF_TOKEN"))
_vote_lock = threading.Lock()
SESSION_FILE_DIR = "/tmp/campfire_votes"
os.makedirs(SESSION_FILE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------- backend
def call_arm(arm: str, category: str, surface: str, current: str, thinking: bool) -> dict:
    """One writer's answer: {text, thinking, tokens, ms} (or {error})."""
    try:
        resp = requests.post(
            BATTLE_URL,
            json={"category": category, "surface": surface, "current": current,
                  "thinking": thinking, "arm": arm, "_auth": AUTH_TOKEN},
            timeout=420,
        )
        resp.raise_for_status()
        return resp.json()[arm]
    except Exception as exc:  # surfaced on the card, never crashes the battle
        return {"error": str(exc)[:200], "text": "", "thinking": "", "tokens": 0, "ms": 0}


def pretty_card(result: dict | None, label: str) -> str:
    """Render one writer's card as markdown."""
    if result is None:
        return f"### {label}\n\n_🪶 scribbling by the firelight…_"
    if result.get("error"):
        return f"### {label}\n\n⚠️ The fire sputtered: `{result['error']}`"
    text = result.get("text", "")
    rewrite, reason, risk = text, "", ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            rewrite = obj.get("rewrite", text) or "_(kept silent)_"
            reason = obj.get("reason", "")
            risk = obj.get("risk", "")
        except json.JSONDecodeError:
            pass
    secs = result.get("ms", 0) / 1000
    badges = f"`{result.get('tokens', 0)} tokens` · `{secs:.1f}s`"
    card = f"### {label}\n\n# “{rewrite}”\n\n{('_' + reason + '_') if reason else ''}"
    if risk:
        card += f"\n\n⚠️ **Risk:** {risk}"
    card += f"\n\n{badges}"
    thinking = (result.get("thinking") or "").strip()
    if thinking:
        preview = thinking if len(thinking) < 2400 else thinking[:2400] + " …"
        card += f"\n\n<details><summary>🔦 see their thinking</summary>\n\n{preview}\n\n</details>"
    return card


# ------------------------------------------------------------------------------ votes
def log_vote(session_id: str, battle: dict, choice: str) -> None:
    """Append the vote to this session's file and push it to the (private) votes dataset."""
    winner = battle["a_is"] if choice == "A" else (
        ("base" if battle["a_is"] == "finetune" else "finetune") if choice == "B" else choice)
    row = {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "copy_campfire",
        "category": battle["category"],
        "surface": battle["surface"],
        "current": battle["current"],
        "thinking_mode": battle["thinking"],
        "choice": choice,                      # A | B | tie | both_bad
        "winner": winner,                      # finetune | base | tie | both_bad
        "a_is": battle["a_is"],
        "text_a": battle["text_a"],
        "text_b": battle["text_b"],
    }
    path = os.path.join(SESSION_FILE_DIR, f"{session_id}.jsonl")
    with _vote_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=f"votes/{session_id}.jsonl",
                            repo_id=VOTES_DATASET, repo_type="dataset")
        except Exception as exc:
            print(f"[warn] vote upload failed: {exc}")


def load_tally() -> dict:
    tally = {"finetune": 0, "base": 0, "tie": 0, "both_bad": 0}
    try:
        for f in api.list_repo_files(VOTES_DATASET, repo_type="dataset"):
            if f.startswith("votes/") and f.endswith(".jsonl"):
                local = api.hf_hub_download(VOTES_DATASET, f, repo_type="dataset")
                for line in open(local, encoding="utf-8"):
                    if line.strip():
                        w = json.loads(line).get("winner", "")
                        if w in tally:
                            tally[w] += 1
    except Exception as exc:
        print(f"[warn] tally load failed: {exc}")
    return tally


TALLY = load_tally()


def tally_md() -> str:
    total = sum(TALLY.values())
    decisive = TALLY["finetune"] + TALLY["base"]
    rate = f"{TALLY['finetune'] / decisive:.0%}" if decisive else "—"
    return (
        f"## 🏕️ Trail log\n\n"
        f"| | votes |\n|---|---|\n"
        f"| 🔥 {FINETUNE_NAME} | **{TALLY['finetune']}** |\n"
        f"| 🌲 {BASE_NAME} | **{TALLY['base']}** |\n"
        f"| 🤝 Tie | {TALLY['tie']} |\n"
        f"| 🪵 Both need work | {TALLY['both_bad']} |\n\n"
        f"**Fine-tune win rate (decisive votes): {rate}** · {total} campfire votes so far\n\n"
        f"_Before launch, the author's own blinded review put the fine-tune at **83%** "
        f"(65/78 decisive) over the base model._"
    )


# ----------------------------------------------------------------------------- events
def new_brief():
    s = random.choice(CORPUS)
    return s["current"], s.get("category", "button"), s.get("surface", "product interface")


def run_battle(current: str, category: str, surface: str, thinking: bool, session_id: str):
    """Generator: kick both writers off, render each card as it lands."""
    current = (current or "").strip()
    if not current:
        yield (gr.update(), gr.update(), gr.update(), None,
               "⚠️ Toss some copy on the fire first — paste a UI string above.")
        return
    if not BATTLE_URL or not AUTH_TOKEN:
        yield (gr.update(), gr.update(), gr.update(), None,
               "⚠️ The campfire isn't lit: BATTLE_URL / AUTH_TOKEN secrets are missing.")
        return

    a_is = random.choice(["finetune", "base"])
    b_is = "base" if a_is == "finetune" else "finetune"
    status = ("🔦 Lantern mode is on — both writers think out loud (can take a minute or two). "
              "First answer appears as soon as it's ready." if thinking
              else "🔥 Fast mode — answers in ~10–30 seconds each.")
    yield (pretty_card(None, "🅰 Camper A"), pretty_card(None, "🅱 Camper B"),
           gr.update(visible=False), None, status)

    results: dict[str, dict | None] = {a_is: None, b_is: None}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {arm: pool.submit(call_arm, arm, category, surface, current, thinking)
                   for arm in (a_is, b_is)}
        pending = set(futures)
        while pending:
            time.sleep(0.5)
            for arm in list(pending):
                if futures[arm].done():
                    results[arm] = futures[arm].result()
                    pending.discard(arm)
                    yield (pretty_card(results[a_is], "🅰 Camper A"),
                           pretty_card(results[b_is], "🅱 Camper B"),
                           gr.update(visible=False), None,
                           "✍️ One answer in — waiting on the second writer…" if pending else "")

    battle = {
        "category": category, "surface": surface, "current": current, "thinking": thinking,
        "a_is": a_is,
        "text_a": (results[a_is] or {}).get("text", ""),
        "text_b": (results[b_is] or {}).get("text", ""),
    }
    yield (pretty_card(results[a_is], "🅰 Camper A"), pretty_card(results[b_is], "🅱 Camper B"),
           gr.update(visible=True), battle, "🪵 Cast your vote to see who's who.")


def vote(choice: str, battle: dict | None, session_id: str):
    if not battle:
        return gr.update(), gr.update(visible=False), gr.update()
    log_vote(session_id, battle, choice)
    winner = battle["a_is"] if choice == "A" else (
        ("base" if battle["a_is"] == "finetune" else "finetune") if choice == "B" else choice)
    if winner in TALLY:
        TALLY[winner] += 1
    a_name = FINETUNE_NAME if battle["a_is"] == "finetune" else BASE_NAME
    b_name = BASE_NAME if battle["a_is"] == "finetune" else FINETUNE_NAME
    reveal = (f"🔥 **The reveal:** 🅰 was **{a_name}** · 🅱 was **{b_name}**. "
              f"Thanks — your vote feeds the next training run. Toss on another log?")
    return reveal, gr.update(visible=False), tally_md()


# --------------------------------------------------------------------------------- UI
CSS = """
.campfire-header {text-align:center; padding: 0.6rem 0 0.2rem;}
.campfire-header h1 {margin-bottom: 0.1rem;}
.campfire-header p {opacity: 0.75; margin-top: 0;}
.card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px; min-height: 180px;}
footer {visibility: hidden;}
"""

theme = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone")

with gr.Blocks(theme=theme, css=CSS, title="⛺ Copy Campfire") as demo:
    session_id = gr.State(lambda: uuid.uuid4().hex[:12])
    battle_state = gr.State(None)

    gr.HTML('<div class="campfire-header"><h1>⛺ Copy Campfire</h1>'
            "<p>Two writers by the fire — one went to training camp. You judge the copy.</p></div>")

    with gr.Tab("🔥 The fire"):
        with gr.Row():
            current_in = gr.Textbox(label="Your UI copy", scale=3,
                                    placeholder="Paste a button label, error message, empty state…")
            category_in = gr.Dropdown(CATEGORIES, value="button", label="Content type", scale=1)
            surface_in = gr.Textbox(label="Where it lives (optional)", scale=2,
                                    placeholder="e.g. checkout, settings page, mobile signup")
        with gr.Row():
            battle_btn = gr.Button("🔥 Start the battle", variant="primary", scale=2)
            smore_btn = gr.Button("🍫 S'more examples", scale=1)
            thinking_in = gr.Checkbox(value=True, scale=2,
                                      label="🔦 Lantern mode — show their thinking (slower)")
        status_md = gr.Markdown("")
        with gr.Row():
            with gr.Column():
                card_a = gr.Markdown(elem_classes="card")
            with gr.Column():
                card_b = gr.Markdown(elem_classes="card")
        with gr.Row(visible=False) as vote_row:
            vote_a = gr.Button("🅰 wins")
            vote_b = gr.Button("🅱 wins")
            vote_tie = gr.Button("🤝 Tie")
            vote_bad = gr.Button("🪵 Both need work")

    with gr.Tab("🏕️ Trail log"):
        leaderboard_md = gr.Markdown(tally_md())

    with gr.Tab("🧭 About"):
        gr.Markdown(f"""
### What is this?

A blind taste test for UX writing. Each battle sends your copy to **two writers**:

- 🌲 **{BASE_NAME}** — Apache-2.0, vision-capable, ~27.8B parameters
- 🔥 **{FINETUNE_NAME}** — the same model after a QLoRA fine-tune on a hand-built UX-writing dataset

Sides are shuffled every round; you vote before the reveal. Votes are stored privately
as preference data for the next training run (DPO) — **your vote literally trains v2**.

### Why trust the matchup?

Before launch, the author blind-reviewed all 90 held-out benchmark items the same way
(options anonymized, judged, then unblinded): the fine-tune was preferred **65/78 = 83%**
of decisive comparisons. Full methodology, eval code, and training pipeline are open.

### Fair-fight settings

Both writers get the identical prompt, greedy decoding, and the same token budget
(1536 with 🔦 lantern mode, 256 without). Qwen3.6 is a reasoning model — lantern mode
lets both think out loud; watch the token counters to see who needs fewer words.

### Take it home

The fine-tune runs anywhere: scan a whole codebase for copy issues with the CLI, or run
it locally via GGUF. Built for the **HF Build Small hackathon** ("small models, big
adventure") for less than $40 of compute.

*Links: model `gr33r/ux-writing-1` · code `github.com/content-designer/ux-writing-1`*
""")

    battle_btn.click(run_battle, [current_in, category_in, surface_in, thinking_in, session_id],
                     [card_a, card_b, vote_row, battle_state, status_md], concurrency_limit=2)
    current_in.submit(run_battle, [current_in, category_in, surface_in, thinking_in, session_id],
                      [card_a, card_b, vote_row, battle_state, status_md], concurrency_limit=2)
    smore_btn.click(new_brief, [], [current_in, category_in, surface_in])
    for btn, choice in ((vote_a, "A"), (vote_b, "B"), (vote_tie, "tie"), (vote_bad, "both_bad")):
        btn.click(vote, [gr.State(choice), battle_state, session_id],
                  [status_md, vote_row, leaderboard_md])

demo.queue(max_size=30)

if __name__ == "__main__":
    demo.launch()
