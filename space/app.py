"""⛺ Copy Campfire — the UX writing arena.

Two UX writers answer the same brief by the fire: one is base Qwen3.6-27B, one is
ux-writing-1 (a LoRA fine-tune for UX writing). Sides are anonymized and randomized;
you vote before the reveal. Votes are stored (privately) as preference data that
trains the next version.

Design: the Copy Campfire Design System (scout-camp meets editorial) — parchment +
kraft stripes, burnt-orange tactile buttons, espresso bark ink, dashed campsite
frames, Archivo/Newsreader/Nunito/Space Mono, emoji-only icons. One fire per view.

Env (Space secrets): BATTLE_URL, AUTH_TOKEN, HF_TOKEN; optional VOTES_DATASET.
"""

from __future__ import annotations

import html as html_lib
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


# ----------------------------------------------------------------- design-system HTML
def esc(s: str) -> str:
    return html_lib.escape(str(s or ""), quote=False)


def banner(tone: str, body_html: str, icon: str | None = None) -> str:
    """Design-system Banner. tone: plain|info|good|warn|bad|fire."""
    default_icon = {"info": "ℹ️", "good": "🏕️", "warn": "⚠️", "bad": "🪵",
                    "fire": "🔥", "plain": "🪶"}
    glyph = default_icon.get(tone, "🪶") if icon is None else icon
    tone_cls = f" cc-banner--{tone}" if tone != "plain" else ""
    return (f'<div class="cc-banner{tone_cls}" role="status">'
            f'<span class="cc-banner__icon" aria-hidden="true">{glyph}</span>'
            f'<div class="cc-banner__body">{body_html}</div></div>')


def pretty_card(result: dict | None, label: str, reveal: str | None = None,
                winner: bool = False, full: bool = False) -> str:
    """Design-system WriterCard.

    Anti-bias rule: before the vote (full=False) the card shows ONLY the copy. Anything
    that could fingerprint a writer — reasoning length, token counts, latency — sits in
    one uniform collapsed "Field notes" disclosure, identical on both sides. After the
    vote (full=True) the forensics unfurl with the reveal.
    """
    cls = "cc-writer cc-writer--winner" if winner else "cc-writer"
    head = f'<div class="cc-writer__head"><span class="cc-writer__label">{label}</span>'
    body = ""

    if result is None:
        head += "</div>"
        body = ('<div class="cc-writer__loading"><span class="cc-writer__dot"></span>'
                "scribbling by the firelight…</div>")
    elif result.get("error"):
        head += "</div>"
        body = (f'<div class="cc-writer__error">⚠️ The fire sputtered: '
                f'<code>{esc(result["error"])}</code></div>')
    else:
        secs = result.get("ms", 0) / 1000
        if full:  # chips are a tell (base thinks 10x longer) — post-vote only
            head += (f'<span class="cc-writer__meta">'
                     f'<span class="cc-writer__chip">{result.get("tokens", 0)} tokens</span>'
                     f'<span class="cc-writer__chip">{secs:.1f}s</span></span>')
        head += "</div>"
        text = result.get("text", "")
        if not text.strip():
            body = (f'<p class="cc-writer__reason">🪵 This writer spent the whole token '
                    f"budget thinking and never answered. That counts as a quality signal — "
                    f"vote accordingly, or turn off lantern mode for direct answers.</p>")
        else:
            rewrite, reason, risk = text, "", ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    rewrite = obj.get("rewrite", text) or "(kept silent)"
                    reason = obj.get("reason", "")
                    risk = obj.get("risk", "")
                except json.JSONDecodeError:
                    pass
            thinking = (result.get("thinking") or "").strip()
            if len(thinking) > 2400:
                thinking = thinking[:2400] + " …"

            body = f'<p class="cc-writer__quote">{esc(rewrite)}</p>'
            if full:
                if reason:
                    body += f'<p class="cc-writer__reason">{esc(reason)}</p>'
                if risk:
                    body += (f'<div class="cc-writer__risk"><span aria-hidden="true">⚠️</span>'
                             f"<span><b>Risk:</b> {esc(risk)}</span></div>")
                if thinking:
                    body += (f'<details class="cc-writer__think"><summary>see their thinking'
                             f"</summary><p>{esc(thinking)}</p></details>")
            elif reason or risk or thinking:
                notes = ""
                if reason:
                    notes += f'<p class="cc-writer__reason">{esc(reason)}</p>'
                if risk:
                    notes += (f'<div class="cc-writer__risk"><span aria-hidden="true">⚠️</span>'
                              f"<span><b>Risk:</b> {esc(risk)}</span></div>")
                if thinking:
                    notes += f'<p class="cc-writer__thinkraw">{esc(thinking)}</p>'
                body += (f'<details class="cc-writer__notes"><summary>field notes — the '
                         f"why, peeking may bias you</summary>{notes}</details>")

    reveal_html = f'<div class="cc-writer__reveal">{reveal}</div>' if reveal else ""
    return f'<div class="{cls}">{head}{body}{reveal_html}</div>'


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
    """Design-system Trail log panel."""
    total = sum(TALLY.values())
    decisive = TALLY["finetune"] + TALLY["base"]
    if decisive:
        hero = (f'<span class="trail__ratenum">{TALLY["finetune"] / decisive:.0%}</span>'
                '<span class="trail__ratelbl">fine-tune win rate, decisive campfire votes</span>')
    else:
        hero = ('<span class="trail__ratelbl">No decisive votes yet — the first 🅰/🅱 '
                "winner lights this number up.</span>")
    max_votes = max(1, TALLY["finetune"], TALLY["base"], TALLY["tie"], TALLY["both_bad"])

    def bar(n: int, color: str) -> str:
        pct = max(4, round(100 * n / max_votes))
        return (f'<td class="trail__bar"><span class="trail__fill" '
                f'style="width:{pct}%;background:{color}"></span></td>')

    rows = [
        ("🪶", FINETUNE_NAME, "went to training camp", TALLY["finetune"], "var(--fire-500)"),
        ("🌲", BASE_NAME, "the wild stock model", TALLY["base"], "var(--pine-500)"),
        ("🤝", "Tie", "both copies would ship", TALLY["tie"], "var(--gold-400)"),
        ("🪵", "Both need work", "back to the drawing board", TALLY["both_bad"], "var(--bark-400)"),
    ]
    trs = "".join(
        f'<tr><td class="trail__emoji">{e}</td>'
        f'<td class="trail__name"><b>{esc(n)}</b><span>{esc(d)}</span></td>'
        f'{bar(v, c)}<td class="trail__wins cc-meta">{v}</td></tr>'
        for e, n, d, v, c in rows
    )
    return f"""
<div class="cc-kicker cc-kicker--fire"><span class="cc-kicker__rule"></span><span>Trail log</span><span class="cc-kicker__rule"></span></div>
<div class="trail__hero">
  <div class="trail__rate">{hero}</div>
</div>
<table class="trail__table">{trs}</table>
<p class="trail__foot">{total} campfire votes so far. Before launch, the author's own blinded
review put the fine-tune at <b>83%</b> (65/78 decisive) over the base model.</p>"""


ABOUT_HTML = f"""
<div class="about">
  <div class="cc-kicker"><span class="cc-kicker__star">✳</span><span>What is this</span><span class="cc-kicker__star">✳</span></div>
  <h2 class="about__h">A blind taste test for UX writing</h2>
  <p class="about__lede">Each battle sends your copy to <b>two UX writers</b>: 🌲 <b>{BASE_NAME}</b>
  (Apache-2.0, ≈27.8B parameters) and 🪶 <b>{FINETUNE_NAME}</b> — the same model after a QLoRA
  fine-tune on a hand-built UX writing dataset. Sides are shuffled every round; you vote before
  the reveal. <b>Your vote literally trains v2</b> — votes are stored privately as preference
  data for the next training run.</p>

  <div class="cc-card cc-card--dashed cc-card--stripes about__panel">
    <h3 class="about__panelh">🎯 Why trust the matchup?</h3>
    <p>Before launch, the author blind-reviewed all 90 held-out benchmark items the same way
    (options anonymized, judged, then unblinded): the fine-tune was preferred
    <b>65/78 = 83%</b> of decisive comparisons. Methodology, eval code, and the training
    pipeline are open.</p>
  </div>
  <div class="cc-card cc-card--dashed about__panel">
    <h3 class="about__panelh">🔦 Fair-fight settings</h3>
    <p>Both writers get the identical prompt, greedy decoding, and the same token budget
    (1536 with lantern mode, 256 without). And because anything that fingerprints a writer
    would bias your vote — one of them reasons at much greater length — explanations, token
    counts, and timings stay tucked into identical "field notes" until <i>after</i> you vote.
    Judge the copy; the forensics come with the reveal.</p>
  </div>
  <div class="cc-card cc-card--dashed about__panel">
    <h3 class="about__panelh">🏕️ Take it home</h3>
    <p>The fine-tune runs anywhere: scan a whole codebase for copy issues with the CLI, run it
    on your laptop via GGUF, or teach it your own style guide in an afternoon. Built for the
    HF <b>Build Small</b> hackathon (<i>small models, big adventure</i>) on ≈$40 of compute.</p>
  </div>
  <p class="foot">⛺ model: gr33r/ux-writing-1 · code: github.com/content-designer/ux-writing-1</p>
</div>"""


# ----------------------------------------------------------------------------- events
def new_brief():
    s = random.choice(CORPUS)
    return s["current"], s.get("category", "button"), s.get("surface", "product interface")


def run_battle(current: str, category: str, surface: str, thinking: bool, session_id: str):
    """Generator: kick both writers off, render each card as it lands."""
    current = (current or "").strip()
    if not current:
        yield (gr.update(), gr.update(), gr.update(), None,
               banner("warn", "Toss some copy on the fire first — paste a UI string above."))
        return
    if not BATTLE_URL or not AUTH_TOKEN:
        yield (gr.update(), gr.update(), gr.update(), None,
               banner("bad", "The campfire isn't lit: BATTLE_URL / AUTH_TOKEN secrets are missing."))
        return

    a_is = random.choice(["finetune", "base"])
    b_is = "base" if a_is == "finetune" else "finetune"
    status = ("🔦 Lantern mode is on — both writers think out loud (can take a minute or two). "
              "First answer appears as soon as it's ready." if thinking
              else "Fast mode — answers in ≈10–30 seconds each.")
    yield (pretty_card(None, "🅰 Camper A"), pretty_card(None, "🅱 Camper B"),
           gr.update(visible=False), None, banner("fire", status))

    # Heartbeat-yield every ~2s: long silent gaps between yields get the SSE stream
    # killed by the proxy (surfaces as a bare Gradio "Error"), and the elapsed counter
    # doubles as honest UX during cold starts.
    try:
        results: dict[str, dict | None] = {a_is: None, b_is: None}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {arm: pool.submit(call_arm, arm, category, surface, current, thinking)
                       for arm in (a_is, b_is)}
            pending = set(futures)
            while pending:
                time.sleep(2)
                done_now = []
                for arm in list(pending):
                    if futures[arm].done():
                        results[arm] = futures[arm].result()
                        pending.discard(arm)
                        done_now.append(arm)
                elapsed = int(time.time() - t0)
                if pending:
                    # ONE message at a time — each state replaces the last.
                    if any(results.values()):
                        msg, icon = "✍️ One answer in — waiting on the second writer…", "🪶"
                    elif elapsed > 30:
                        msg, icon = ("⛺ Waking the campfire GPU — the first battle after a "
                                     "quiet spell takes up to ≈3 minutes."), "⛺"
                    else:
                        msg, icon = "Roasting marshmallows… both campers are writing.", "🔥"
                    # Only re-render a card when its writer just finished (less DOM churn).
                    yield (pretty_card(results[a_is], "🅰 Camper A") if a_is in done_now else gr.update(),
                           pretty_card(results[b_is], "🅱 Camper B") if b_is in done_now else gr.update(),
                           gr.update(visible=False), None,
                           banner("info", f"{msg} <span class='cc-meta'>{elapsed}s</span>", icon=icon))

        battle = {
            "category": category, "surface": surface, "current": current, "thinking": thinking,
            "a_is": a_is,
            "result_a": results[a_is], "result_b": results[b_is],
            "text_a": (results[a_is] or {}).get("text", ""),
            "text_b": (results[b_is] or {}).get("text", ""),
        }
        yield (pretty_card(results[a_is], "🅰 Camper A"), pretty_card(results[b_is], "🅱 Camper B"),
               gr.update(visible=True), battle,
               banner("plain", "Read both, then cast your vote. The campers stay anonymous until you do."))
    except Exception as exc:  # never surface a bare Gradio "Error" chip
        yield (gr.update(), gr.update(), gr.update(visible=False), None,
               banner("bad", f"The fire sputtered: <code>{esc(str(exc)[:160])}</code> — try another battle."))


def vote(choice: str, battle: dict | None, session_id: str):
    if not battle:
        return gr.update(), gr.update(), gr.update(), gr.update(visible=False), gr.update()
    log_vote(session_id, battle, choice)
    winner = battle["a_is"] if choice == "A" else (
        ("base" if battle["a_is"] == "finetune" else "finetune") if choice == "B" else choice)
    if winner in TALLY:
        TALLY[winner] += 1

    def reveal_for(side_is: str) -> str:
        return ("🪶 ux-writing-1 — the fine-tune" if side_is == "finetune"
                else "🌲 Qwen3.6-27B — the base model")

    b_is = "base" if battle["a_is"] == "finetune" else "finetune"
    card_a = pretty_card(battle.get("result_a"), "🅰 Camper A", full=True,
                         reveal=reveal_for(battle["a_is"]), winner=(winner == battle["a_is"] and choice == "A"))
    card_b = pretty_card(battle.get("result_b"), "🅱 Camper B", full=True,
                         reveal=reveal_for(b_is), winner=(winner == b_is and choice == "B"))
    fine_slot = "🅰" if battle["a_is"] == "finetune" else "🅱"
    status = banner("good", f"<b>The reveal:</b> {fine_slot} was <b>ux-writing-1</b> (the "
                            f"fine-tune). Thanks — your vote feeds the next training run. "
                            f"Toss on another log?")
    return card_a, card_b, status, gr.update(visible=False), tally_md()


# --------------------------------------------------------------------------------- UI
FONTS_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;800;900&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400;1,6..72,500&family=Nunito:ital,wght@0,400;0,600;0,700;0,800;1,400&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
"""

CSS = """
/* ════════ Copy Campfire Design System — tokens ════════ */
:root {
  --fire-50:#FFF2E6; --fire-100:#FFE1C6; --fire-200:#FCC596; --fire-300:#FBA968;
  --fire-400:#F7842F; --fire-500:#F2641B; --fire-600:#DA530F; --fire-700:#B23F08;
  --ember-500:#EE8E1E; --gold-400:#D8A94B; --gold-500:#C0892C; --gold-600:#9A6B1E;
  --parch-50:#FCF8EF; --parch-100:#F8F1DF; --parch-200:#F1E7CF; --parch-300:#E6D6B4; --parch-400:#D6C09A;
  --bark-400:#9B7B55; --bark-500:#7A5C3C; --bark-600:#5E4429; --bark-700:#4A3420;
  --bark-800:#362618; --bark-900:#271A0F;
  --pine-500:#6E7B3E; --pine-600:#5C6E33; --pine-700:#45531F;
  --ink-blue:#2F6CB0; --bad-500:#C24A24;
  --text-strong:var(--bark-900); --text-body:var(--bark-800); --text-muted:#6E5B43;
  --text-faint:#927B5E; --text-accent:var(--pine-600);
  --surface-raised:var(--parch-50); --surface-sunken:var(--parch-100);
  --border-hairline:rgba(54,38,24,0.16); --border-strong:rgba(54,38,24,0.32);
  --border-camp:var(--bark-500);
  --primary:var(--fire-500); --primary-hover:var(--fire-600); --primary-edge:var(--fire-700);
  --tint-good:rgba(110,123,62,0.14); --tint-warn:rgba(192,137,44,0.16);
  --tint-bad:rgba(194,74,36,0.14); --tint-info:rgba(47,108,176,0.12);
  --font-display:'Archivo',system-ui,sans-serif;
  --font-serif:'Newsreader',Georgia,serif;
  --font-sans:'Nunito',system-ui,sans-serif;
  --font-mono:'Space Mono',ui-monospace,Menlo,monospace;
  --radius-sm:6px; --radius-md:10px; --radius-card:14px; --radius-lg:18px; --radius-pill:999px;
  --shadow-xs:0 1px 2px rgba(39,26,15,0.08);
  --shadow-card:0 1px 2px rgba(39,26,15,0.06),0 8px 22px rgba(39,26,15,0.07);
  --tactile-depth:5px;
  --ease-out:cubic-bezier(0.22,1,0.36,1); --dur-fast:120ms; --dur-base:200ms; --dur-slow:320ms;
  --texture-stripes:repeating-linear-gradient(-45deg,transparent 0,transparent 9px,
      rgba(122,92,60,0.06) 9px,rgba(122,92,60,0.06) 10px);
}

/* ════════ page ════════ */
html { scrollbar-gutter: stable; } /* stop layout width-flap when the scrollbar appears */
body, .gradio-container {
  background: var(--parch-200) !important;
  background-image: var(--texture-stripes) !important;
  font-family: var(--font-sans) !important;
  color: var(--text-body) !important;
}
/* ONE width, owned by the design system: clamp every Gradio wrapper to the 1080px
   content token so responsive breakpoints can't snap the layout narrow/wide. */
.gradio-container, .gradio-container > .main, .gradio-container .fillable,
.gradio-container main, gradio-app > div {
  max-width: 1080px !important; width: 100% !important; margin-inline: auto !important;
}
.gradio-container {
  /* Re-point Gradio's own theme tokens at the design system (fixes blue checkbox,
     off-brand tab underline, and focus rings at the source). */
  --color-accent: var(--fire-500) !important;
  --color-accent-soft: var(--fire-100) !important;
  --checkbox-background-color-selected: var(--fire-500) !important;
  --checkbox-border-color-selected: var(--fire-500) !important;
  --checkbox-border-color-focus: var(--fire-400) !important;
  --checkbox-border-color: var(--bark-600) !important;
  --input-border-color-focus: var(--fire-500) !important;
  --body-text-color: var(--bark-800) !important;
  --block-label-text-color: var(--fire-600) !important;
  --loader-color: var(--fire-500) !important;
}
footer { visibility: hidden; }
.cc-meta { font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: .04em; color: var(--text-muted); }

/* ════════ topbar ════════ */
.campfire-header { display:flex; align-items:center; gap:13px; padding: 22px 4px 6px; background: transparent; }
.campfire-header .brandmark { font-size:38px; line-height:1; filter: drop-shadow(0 2px 3px rgba(39,26,15,.25)); }
.campfire-header .brandname { font-family:var(--font-display); font-weight:900; font-size:30px;
  color:var(--bark-900); letter-spacing:-0.01em; line-height:1.05; }
.campfire-header .brandtag { font-family:var(--font-serif); font-style:italic; font-size:1.125rem; color:var(--text-muted); }

/* ════════ tabs ════════ */
.tab-nav, .tab-wrapper .tab-container { border-bottom: 2px solid var(--border-hairline) !important; background: transparent !important; }
button[role="tab"] {
  font-family: var(--font-sans) !important; font-weight: 700 !important; font-size: 1.125rem !important;
  color: var(--text-muted) !important; background: none !important; border: none !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important; padding: 10px 14px !important;
}
button[role="tab"]:hover { color: var(--text-strong) !important; background: rgba(110,123,62,0.12) !important; }
button[role="tab"][aria-selected="true"] { color: var(--fire-600) !important;
  border-bottom: 3px solid var(--fire-500) !important; }

/* ════════ the brief (double-bark frame) ════════ */
.cc-brief {
  background: var(--surface-raised) !important;
  border: 3px solid var(--bark-700) !important; border-radius: var(--radius-lg) !important;
  box-shadow: inset 0 0 0 2px var(--parch-50), var(--shadow-card) !important;
  padding: 18px !important; gap: 12px !important;
}
.cc-brief .block, .cc-brief .form { background: transparent !important; border: none !important; box-shadow: none !important; }
/* Field labels only (Gradio marks them with block-info) — NOT checkbox/body labels */
.cc-brief span[data-testid="block-info"] {
  font-family: var(--font-sans) !important; font-weight: 700 !important;
  font-size: 0.875rem !important; color: var(--fire-600) !important;
}
.cc-lantern label, .cc-lantern label span {
  font-family: var(--font-sans) !important; font-weight: 700 !important;
  font-size: 1rem !important; color: var(--bark-800) !important;
}
.cc-lantern span[data-testid="block-info"], .cc-lantern .info {
  font-family: var(--font-serif) !important; font-style: italic !important;
  font-weight: 400 !important; font-size: 0.9rem !important;
  color: var(--text-muted) !important; margin-top: 2px !important;
}

/* Gradio's built-in loaders are redundant with our status banner + scribbling cards */
.progress-bar, .progress-text, .eta-bar, .meta-text, .meta-text-center { display: none !important; }
.generating { border: none !important; animation: none !important; }
.cc-brief textarea, .cc-brief input[type="text"], .cc-brief .dropdown input, .cc-brief .wrap-inner {
  font-family: var(--font-sans) !important; font-size: 1rem !important; color: var(--text-strong) !important;
  background: var(--surface-raised) !important;
  border: 2px solid var(--border-strong) !important; border-radius: var(--radius-md) !important;
}
.cc-brief textarea:focus, .cc-brief input:focus {
  border-color: var(--fire-500) !important; box-shadow: 0 0 0 3px var(--fire-100) !important;
}
.cc-brief input[type="checkbox"] { accent-color: var(--fire-500) !important; width: 20px; height: 20px; }

/* ════════ tactile buttons ════════ */
button.cc-btn {
  font-family: var(--font-sans) !important; font-weight: 800 !important; font-size: 1rem !important;
  line-height: 1 !important; color: #fff !important;
  background: var(--primary) !important; border: none !important; border-radius: var(--radius-md) !important;
  padding: 14px 24px !important; white-space: nowrap !important;
  box-shadow: 0 var(--tactile-depth) 0 var(--primary-edge), 0 9px 10px rgba(39,26,15,.20) !important;
  transform: translateY(0); transition: transform var(--dur-fast) var(--ease-out),
    box-shadow var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out);
}
button.cc-btn-secondary { white-space: nowrap !important; }
button.cc-btn:hover { background: var(--primary-hover) !important; }
button.cc-btn:active { transform: translateY(4px);
  box-shadow: 0 1px 0 var(--primary-edge), 0 2px 4px rgba(39,26,15,.18) !important; }
button.cc-btn-secondary {
  background: var(--surface-raised) !important; color: var(--bark-900) !important;
  border: 2px solid var(--bark-700) !important;
  box-shadow: 0 var(--tactile-depth) 0 var(--parch-400), 0 9px 10px rgba(39,26,15,.12) !important;
}
button.cc-btn-secondary:hover { background: var(--parch-100) !important; }
button.cc-btn-secondary:active { transform: translateY(4px); box-shadow: 0 1px 0 var(--parch-400) !important; }

/* ════════ banner ════════ */
.cc-banner { display:flex; align-items:flex-start; gap:11px; font-family:var(--font-sans);
  font-size:1rem; color:var(--text-body); background:var(--surface-raised);
  border:1px solid var(--border-hairline); border-left:4px solid var(--bark-500);
  border-radius:var(--radius-md); padding:13px 16px; box-shadow:var(--shadow-xs); }
.cc-banner__icon { font-size:18px; line-height:1.35; flex:none; }
.cc-banner__body { flex:1; } .cc-banner__body b { color:var(--text-strong); }
.cc-banner--info { border-left-color:var(--ink-blue); background:#f3ede0; }
.cc-banner--good { border-left-color:var(--pine-500); background:#ece9d6; }
.cc-banner--warn { border-left-color:var(--gold-500); background:#f5ead2; }
.cc-banner--bad  { border-left-color:var(--bad-500); background:#f3e0d4; }
.cc-banner--fire { border-left-color:var(--fire-500); background:var(--fire-50); }

/* ════════ writer cards ════════ */
.cc-writer { position:relative; background:var(--surface-raised);
  border:1px solid var(--border-hairline); border-radius:var(--radius-card);
  padding:20px 22px 18px; box-shadow:var(--shadow-card);
  display:flex; flex-direction:column; gap:12px; min-height:180px;
  transition:border-color var(--dur-base) var(--ease-out), box-shadow var(--dur-base) var(--ease-out); }
.cc-writer::before { content:""; position:absolute; left:18px; right:18px; top:0; height:4px;
  background:var(--bark-400); border-radius:0 0 4px 4px; opacity:.5; }
.cc-writer--winner { border-color:var(--fire-400);
  box-shadow:var(--shadow-card), 0 0 0 3px var(--fire-100); }
.cc-writer--winner::before { background:var(--fire-500); opacity:1; }
.cc-writer__head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
.cc-writer__label { font-family:var(--font-sans); font-weight:800; font-size:1.125rem; color:var(--text-strong); }
.cc-writer__meta { display:inline-flex; gap:6px; }
.cc-writer__chip { font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);
  background:var(--surface-sunken); border:1px solid var(--border-hairline);
  border-radius:var(--radius-sm); padding:3px 7px; white-space:nowrap; }
.cc-writer__quote { font-family:var(--font-serif); font-weight:500; font-size:1.75rem;
  line-height:1.18; color:var(--text-strong); margin:2px 0; }
.cc-writer__quote::before { content:"\\201C"; color:var(--fire-400); }
.cc-writer__quote::after { content:"\\201D"; color:var(--fire-400); }
.cc-writer__reason { font-family:var(--font-serif); font-style:italic; font-size:1.125rem; color:var(--text-muted); }
.cc-writer__risk { display:flex; gap:8px; align-items:flex-start; font-family:var(--font-sans);
  font-size:0.875rem; color:var(--bark-800); background:var(--tint-warn);
  border-radius:var(--radius-sm); padding:8px 10px; }
.cc-writer__risk b { color:var(--gold-600); }
.cc-writer__loading { font-family:var(--font-serif); font-style:italic; font-size:1.125rem;
  color:var(--text-faint); display:flex; align-items:center; gap:8px; }
.cc-writer__dot { width:7px; height:7px; border-radius:50%; background:var(--fire-400);
  animation:cc-pulse 1s var(--ease-out) infinite; }
@keyframes cc-pulse { 0%,100%{opacity:.3; transform:scale(.8);} 50%{opacity:1; transform:scale(1.1);} }
.cc-writer__error { font-family:var(--font-sans); font-size:0.875rem; color:var(--bad-500); }
.cc-writer__think summary { font-family:var(--font-sans); font-weight:600; font-size:0.875rem;
  color:var(--pine-600); cursor:pointer; list-style:none; }
.cc-writer__think summary::-webkit-details-marker { display:none; }
.cc-writer__think summary::before { content:"🔦 "; }
.cc-writer__think p { font-family:var(--font-mono); font-size:0.75rem; line-height:1.65;
  color:var(--text-muted); background:var(--surface-sunken); border-radius:var(--radius-sm);
  padding:10px 12px; margin-top:8px; max-height:160px; overflow:auto; }
/* Pre-vote uniform disclosure: identical on both cards so nothing fingerprints a writer */
.cc-writer__notes summary { font-family:var(--font-serif); font-style:italic; font-size:0.9rem;
  color:var(--text-faint); cursor:pointer; list-style:none; }
.cc-writer__notes summary::-webkit-details-marker { display:none; }
.cc-writer__notes summary::before { content:"🗒️ "; }
.cc-writer__notes[open] summary { color:var(--text-muted); }
.cc-writer__notes .cc-writer__reason { margin-top:8px; }
.cc-writer__thinkraw { font-family:var(--font-mono); font-size:0.75rem; line-height:1.65;
  color:var(--text-muted); background:var(--surface-sunken); border-radius:var(--radius-sm);
  padding:10px 12px; margin-top:8px; max-height:160px; overflow:auto; }
.cc-writer__reveal { font-family:var(--font-mono); font-size:0.75rem; letter-spacing:.04em;
  color:var(--fire-700); border-top:1px dashed var(--border-camp); padding-top:10px; margin-top:auto; }

/* ════════ kicker + vote hint ════════ */
.cc-kicker { font-family:var(--font-serif); font-weight:600; font-size:0.875rem;
  letter-spacing:0.18em; text-transform:uppercase; color:var(--text-accent);
  display:flex; align-items:center; gap:12px; justify-content:center; margin:6px 0 2px; }
.cc-kicker--fire { color:var(--fire-600); }
.cc-kicker__star { color:var(--ember-500); font-size:0.85em; }
.cc-kicker__rule { flex:1; height:0; border-top:1px dashed var(--border-camp); min-width:24px; max-width:160px; }
.cc-votehint { text-align:center; font-family:var(--font-serif); font-style:italic;
  color:var(--text-muted); font-size:1.125rem; padding-top:4px; }

/* ════════ trail log ════════ */
.trail__hero { display:flex; align-items:center; gap:20px; margin:12px 0 18px; }
.trail__ratenum { font-family:var(--font-display); font-weight:900; font-size:64px;
  line-height:1; color:var(--fire-600); display:block; }
.trail__ratelbl { font-family:var(--font-serif); font-style:italic; font-size:1.125rem; color:var(--text-muted); }
.trail__table { width:100%; border-collapse:collapse; border:none !important; }
.trail__table tr { border:none !important; background:transparent !important; }
.trail__table td { padding:11px 8px; border:none !important;
  border-bottom:1px dashed var(--border-camp) !important; vertical-align:middle; }
.trail__emoji { font-size:24px; width:36px; text-align:center; }
.trail__name { width:220px; }
.trail__name b { display:block; font-family:var(--font-sans); font-weight:800;
  color:var(--text-strong); font-size:1rem; }
.trail__name span { font-family:var(--font-serif); font-style:italic; font-size:0.875rem; color:var(--text-muted); }
.trail__fill { display:block; height:12px; border-radius:var(--radius-pill); min-width:4px;
  transition:width var(--dur-slow) var(--ease-out); }
.trail__wins { width:52px; text-align:right; }
.trail__foot { font-family:var(--font-serif); font-size:1.125rem; color:var(--text-muted); margin:16px 0 0; }

/* ════════ about ════════ */
.about__h { font-family:var(--font-display); font-weight:900; font-size:3rem;
  color:var(--bark-900); margin:10px 0 12px; line-height:1.04; }
.about__lede { font-family:var(--font-serif); font-size:1.375rem; line-height:1.65;
  color:var(--text-body); max-width:64ch; margin:0 0 18px; }
.about__panel { margin-top:14px; }
.cc-card { background:var(--surface-raised); border-radius:var(--radius-card);
  border:1px solid var(--border-hairline); box-shadow:var(--shadow-card); padding:20px; }
.cc-card--dashed { border:2px dashed var(--border-camp); box-shadow:none; }
.cc-card--stripes { background-image:var(--texture-stripes); }
.about__panelh { font-family:var(--font-sans); font-weight:800; font-size:1.125rem;
  color:var(--text-strong); margin:0 0 8px; }
.about__panel p { font-family:var(--font-serif); font-size:1.125rem; line-height:1.65;
  color:var(--text-muted); margin:0; }
.foot { display:flex; gap:10px; justify-content:center; align-items:center; padding-top:28px;
  font-family:var(--font-mono); font-size:0.75rem; color:var(--text-faint); }

/* Gradio chrome cleanup inside tabs */
.tabitem, .gap, .block { background: transparent !important; border: none !important; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=CSS, head=FONTS_HEAD, title="⛺ Copy Campfire",
               fill_width=True) as demo:
    session_id = gr.State(lambda: uuid.uuid4().hex[:12])
    battle_state = gr.State(None)

    gr.HTML(
        '<div class="campfire-header"><span class="brandmark">⛺</span>'
        '<div><div class="brandname">Copy Campfire</div>'
        '<div class="brandtag">Two UX writers by the fire — one went to training camp. '
        "You judge the copy.</div></div></div>"
    )

    with gr.Tab("🔥 The fire"):
        with gr.Column(elem_classes="cc-brief"):
            gr.HTML('<div class="cc-kicker cc-kicker--fire">'
                    '<span class="cc-kicker__rule"></span><span>The brief</span>'
                    '<span class="cc-kicker__rule"></span></div>')
            with gr.Row(equal_height=False):
                current_in = gr.Textbox(label="UI copy on trial", scale=3, lines=5,
                                        placeholder="Paste a button label, error message, empty state…")
                with gr.Column(scale=2):
                    category_in = gr.Dropdown(CATEGORIES, value="button", label="Content type")
                    surface_in = gr.Textbox(label="Where it lives (optional)",
                                            placeholder="e.g. checkout, settings page")
                    thinking_in = gr.Checkbox(value=True, elem_classes="cc-lantern",
                                              label="🔦 Lantern mode",
                                              info="Show their thinking — slower, but you see how each writer reasons")
            with gr.Row():
                battle_btn = gr.Button("🔥 Light the fire", elem_classes="cc-btn", scale=0, min_width=220)
                smore_btn = gr.Button("🍫 S'more examples", elem_classes="cc-btn-secondary",
                                      scale=0, min_width=200)
        status_md = gr.HTML("")
        with gr.Row():
            card_a = gr.HTML()
            card_b = gr.HTML()
        with gr.Column(visible=False) as vote_row:
            gr.HTML('<div class="cc-votehint">🪵 Whose copy would you ship?</div>')
            with gr.Row():
                vote_a = gr.Button("🅰 wins", elem_classes="cc-btn")
                vote_b = gr.Button("🅱 wins", elem_classes="cc-btn")
                vote_tie = gr.Button("🤝 Tie", elem_classes="cc-btn-secondary")
                vote_bad = gr.Button("🪵 Both need work", elem_classes="cc-btn-secondary")

    with gr.Tab("🏕️ Trail log"):
        leaderboard_md = gr.HTML(tally_md())

    with gr.Tab("🧭 About"):
        gr.HTML(ABOUT_HTML)

    # show_progress="hidden": our status banner + scribbling cards ARE the loading UI
    battle_btn.click(run_battle, [current_in, category_in, surface_in, thinking_in, session_id],
                     [card_a, card_b, vote_row, battle_state, status_md],
                     concurrency_limit=2, show_progress="hidden")
    current_in.submit(run_battle, [current_in, category_in, surface_in, thinking_in, session_id],
                      [card_a, card_b, vote_row, battle_state, status_md],
                      concurrency_limit=2, show_progress="hidden")
    smore_btn.click(new_brief, [], [current_in, category_in, surface_in], show_progress="hidden")
    for btn, choice in ((vote_a, "A"), (vote_b, "B"), (vote_tie, "tie"), (vote_bad, "both_bad")):
        btn.click(vote, [gr.State(choice), battle_state, session_id],
                  [card_a, card_b, status_md, vote_row, leaderboard_md], show_progress="hidden")

demo.queue(max_size=30)

if __name__ == "__main__":
    demo.launch()
