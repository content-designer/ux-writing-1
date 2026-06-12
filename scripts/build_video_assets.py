"""Inject run artifacts into the video asset templates. No hand-typed numbers.

Each built page is fully self-contained: the shared Campfire CSS and the data
JSON are inlined, so the HTML can be opened/recorded from anywhere.

    python3 scripts/build_video_assets.py                  # build all three
    python3 scripts/build_video_assets.py weights_heatmap  # build a subset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ASSETS = Path("docs/video_assets")

JOBS = {
    "cost_compare.html": "docs/demo/posthog_cost_report.json",
    "weights_heatmap.html": "docs/video_assets/adapter_deltas.json",
    "stats_banner.html": "docs/demo/posthog_cost_report.json",
}


def build(name: str, data_path: str) -> None:
    template = (ASSETS / "templates" / name).read_text(encoding="utf-8")
    css = (ASSETS / "campfire.css").read_text(encoding="utf-8")
    data = json.dumps(json.loads(Path(data_path).read_text(encoding="utf-8")))
    out = template.replace("/*__CAMPFIRE_CSS__*/", css).replace("/*__DATA__*/", data)
    if out == template:
        raise SystemExit(f"{name}: no injection points found")
    (ASSETS / name).write_text(out, encoding="utf-8")
    print(f"built {ASSETS / name}")


def main() -> int:
    only = {f"{a.removesuffix('.html')}.html" for a in sys.argv[1:]} or set(JOBS)
    for name, data_path in JOBS.items():
        if name not in only:
            continue
        if not Path(data_path).exists():
            print(f"skip {name}: {data_path} not present yet")
            continue
        build(name, data_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
