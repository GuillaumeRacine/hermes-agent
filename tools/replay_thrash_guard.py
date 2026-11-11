#!/usr/bin/env python3
"""Replay recorded compression passes through the old and new guards (#64).

Reads the real ``compression started`` / ``compression done`` pairs out of
``~/.hermes/logs/agent.log*`` and asks, per session: how many passes would each
guard have allowed?

The old guard needed two *consecutive* ineffective passes and reset on any
effective one; the new one backs off when 2 of the last 3 were ineffective and
scores against the following pass's observed context rather than an estimate.

    python3 tools/replay_thrash_guard.py [--min-passes 3]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.context_compressor import (  # noqa: E402
    _MIN_SAVINGS_PCT, _OUTCOME_TRIGGER, _OUTCOME_WINDOW,
)

LOGS = sorted((Path.home() / ".hermes" / "logs").glob("agent.log*"))
START = re.compile(r"compression started: session=(\S+) messages=(\d+) tokens=~([\d,]+)")
DONE = re.compile(r"compression done: session=(\S+) messages=[\d]+->[\d]+ rough_tokens=~([\d,]+)")


def _n(s: str) -> int:
    return int(s.replace(",", ""))


def load() -> dict[str, list[tuple[int, int]]]:
    """session -> [(tokens_before, tokens_after), ...] in log order."""
    starts: dict[str, list[int]] = defaultdict(list)
    dones: dict[str, list[int]] = defaultdict(list)
    for lf in LOGS:
        try:
            text = lf.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            m = START.search(line)
            if m:
                starts[m.group(1)].append(_n(m.group(3)))
                continue
            m = DONE.search(line)
            if m:
                dones[m.group(1)].append(_n(m.group(2)))
    return {s: list(zip(v, dones.get(s, []))) for s, v in starts.items() if dones.get(s)}


def simulate(passes: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (allowed_old, allowed_new) for one session's recorded passes."""
    old_consecutive = 0
    old_allowed = 0
    window: list[bool] = []
    new_allowed = 0
    for before, after in passes:
        pct = (before - after) / before * 100 if before else 0.0
        ineffective = pct < _MIN_SAVINGS_PCT

        if old_consecutive < 2:
            old_allowed += 1
            old_consecutive = old_consecutive + 1 if ineffective else 0

        if sum(window) < _OUTCOME_TRIGGER:
            new_allowed += 1
            window.append(ineffective)
            del window[:-_OUTCOME_WINDOW]
    return old_allowed, new_allowed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-passes", type=int, default=3)
    a = ap.parse_args(argv)

    sessions = load()
    if not sessions:
        print("no recorded compression passes found", file=sys.stderr)
        return 2

    rows = []
    for sid, passes in sessions.items():
        if len(passes) < a.min_passes:
            continue
        rows.append((sid, len(passes), *simulate(passes)))
    rows.sort(key=lambda r: -r[1])

    tot_r = sum(r[1] for r in rows)
    tot_o = sum(r[2] for r in rows)
    tot_n = sum(r[3] for r in rows)
    print(f"sessions with >={a.min_passes} recorded passes: {len(rows)}\n")
    print(f"  {'session':32s} {'recorded':>9s} {'old guard':>10s} {'new guard':>10s}")
    for sid, rec, old, new in rows[:12]:
        print(f"  {sid:32s} {rec:9d} {old:10d} {new:10d}")
    print(f"\n  {'TOTAL':32s} {tot_r:9d} {tot_o:10d} {tot_n:10d}")
    if tot_o:
        print(f"\n  passes avoided vs old guard: {tot_o - tot_n} ({(1 - tot_n / tot_o) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
