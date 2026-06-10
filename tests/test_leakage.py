# tests/test_leakage.py
"""Asserts for the real-bug leakage verifier (offline denylist portion)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.verify_realbug_leakage import check_denylist, CALCOM_TRAIN_DENYLIST


def test_realbug_leakage():
    # s07/s09 are the held-out eval bug screens — they must never be on the TRAIN denylist.
    assert "calcom_s07" not in CALCOM_TRAIN_DENYLIST and "calcom_s09" not in CALCOM_TRAIN_DENYLIST, \
        CALCOM_TRAIN_DENYLIST

    # s05/s08 were folded into TRAIN; using them in eval must be flagged.
    bad = [
        {"id": "x1", "image": "screens/calcom_s05.png", "planted_from": ""},
        {"id": "x2", "image": "screens/gitea_team.png", "planted_from": "calcom_s08 (edit ...)"},
    ]
    problems = check_denylist(bad)
    assert any("calcom_s05" in p for p in problems), problems
    assert any("calcom_s08" in p for p in problems), problems

    # s07/s09 are the held-out eval bug screens — allowed.
    ok = [
        {"id": "a", "image": "screens/calcom_s07.png", "planted_from": ""},
        {"id": "b", "image": "screens/calcom_s09.png", "planted_from": ""},
        {"id": "c", "image": "screens/gitea_team_dup.png", "planted_from": "gitea_team_members (edit)"},
    ]
    assert check_denylist(ok) == [], check_denylist(ok)
    print("check_realbug_leakage OK")
