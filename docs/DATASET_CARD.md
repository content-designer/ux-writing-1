---
license: apache-2.0
language:
  - en
pretty_name: UX Writing SFT (rewrite + consistency)
tags:
  - ux-writing
  - microcopy
  - content-design
  - multimodal
  - sft
size_categories:
  - 1K<n<10K
---

# UX Writing SFT — `gr33r/ux-writing-sft`

A unified, versioned, **zero-leakage** supervised fine-tuning dataset for reviewing UX writing
across codebases and screenshots. It consolidates two prior datasets into one repo with tagged
subsets so a single base model (`Qwen/Qwen3.6-27B`) can be trained on both tasks and evaluated
with full attribution. Built for the [Build Small hackathon](https://huggingface.co/build-small-hackathon).

## Tasks (one row container, two shapes)

| `task` | `modality` | Input → Output |
|---|---|---|
| `rewrite` | `text` | UI string + code context → `{"rewrite","reason","risk"}` |
| `consistency` | `vision` | screenshot → `{"inventory":[...],"issues":[{type,strings,problem,fix}]}` |

## Schema (storage)

All columns are simple types so Arrow can hold both task shapes in one table:

| column | type | notes |
|---|---|---|
| `id` | string | stable unique id |
| `task` | string | `rewrite` \| `consistency` |
| `modality` | string | `text` \| `vision` |
| `source` | string | `synthetic` \| `real` \| `oss` \| `merchant` |
| `split` | string | `train` \| `validation` \| `test` |
| `image` | string | dataset-relative path (`screens/...`) or `""` for text rows |
| `messages_json` | string | JSON-encoded chat messages (string content for text; list-of-parts for vision) |
| `metadata_json` | string | JSON-encoded metadata |
| `provenance_json` | string | JSON-encoded provenance (source ids / license / url / posture) |

Parse `messages_json` back into the chat list in your collator; image bytes live under `screens/`.

## Splits

| split | rewrite | consistency | notes |
|---|---:|---:|---|
| train | ~1,392 | 156 | rewrite synthetic-derived; consistency synthetic screens |
| validation | ~60 | 24 | rewrite held out by `input_key`; consistency synthetic held-out |
| test | 90 | (added separately) | rewrite hand-authored gold benchmark; real-screenshot consistency test is appended by the eval-annotation step |

## Zero-leakage guarantees

- **Consistency train/val are synthetic-only.** Real screenshots (Cal.com, Ghost, Wealthsimple,
  and newly annotated references) are added **only** to the `test` split — never to train.
- **Rewrite val/test never share a UI string with train**: rewrite rows are split by
  `input_key = (category, current_copy)`; no input_key spans train and validation/test.
- The builder fails closed if any leakage or schema violation is detected.

## Provenance & licensing

Apache-2.0. Training rows are owner-authored or derived from public guidance **without verbatim
copying** (derived-only posture); synthetic screens are owner-generated. No proprietary or
course-licensed material is included. Real screenshots used for evaluation are from
permissively-licensed open-source products (MIT) or the owner's own redesigns; see `NOTICE`.

## Build

```bash
python data_build/build_unified_dataset.py            # build + validate locally
python data_build/build_unified_dataset.py --push     # push to the Hub (private)
```

Built from [content-designer/ux-writing-fine-tune](https://github.com/content-designer/ux-writing-fine-tune).
