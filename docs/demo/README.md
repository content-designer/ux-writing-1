# Demo artifact: reviewing Cal.com's UI copy with ux-writing-1

A real run against [Cal.com](https://github.com/calcom/cal.com) (open-source scheduling
app, ≈7,700 files), June 10 2026 — the workflow from the video, unedited:

```bash
python -m uxft.scan /path/to/cal.com --limit 200 --out candidates.jsonl     # seconds
python -m uxft.review_repo /path/to/cal.com --limit 200 --workers 8 \
  --endpoint $UXW1_ENDPOINT --api-key $TOKEN --out review.jsonl             # ≈10 min, ≈$0.40
```

Two passes, 200 strings each: the monorepo root ([calcom_review.jsonl](calcom_review.jsonl))
and the UI-dense `apps/web` tree ([calcom_web_review.jsonl](calcom_web_review.jsonl)).
**Zero request errors; the model kept 365 of 400 strings as-is** — restraint is trained
behavior, not luck.

## Highlights (verbatim from the JSONL)

| current copy | suggested | why it's good |
|---|---|---|
| `Internal Server Error` | "Something went wrong on our end. Try again in a moment." | blame-free, recoverable |
| `SÃ£o Paulo` | "São Paulo" | caught a mojibake/encoding bug |
| `week_view` | "Week view" | i18n key surfaced as readable label |
| `update-eventtype` | "Update event type" | same |
| `ApiKey for cronjobs` | "API key for scheduled tasks" | terminology + plain language |
| `Sendgrid api key. Used for email reminde…` | "SendGrid API key. Used for email reminders in workflows" | brand casing fixed |
| `Configures the global SMTP server port` | "Port for the global SMTP server" | label, not sentence |
| `Set the following value to true if you w…` | "Enable team impersonation" | 9 words → 3 |

## Honest caveats (left in the artifact on purpose)

- The **scanner is regex-based** and over-collects on monorepos: it grabbed some CSS
  class strings and config values (`text-blue-500`, `mt-2 inline-block…`), which the
  model then tried to rewrite from surrounding context. Filter candidates by `kind`, or
  scan UI-dense subtrees. (AST-aware scanning is a welcome contribution.)
- This is a **human-in-the-loop review feed**, not a codemod. Nothing here was applied
  to Cal.com; suggestions on config/infra strings in particular need an engineer's eyes.

Throughput/cost context for this run: [../COST_NOTES.md](../COST_NOTES.md).
