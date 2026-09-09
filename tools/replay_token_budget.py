#!/usr/bin/env python3
"""Replay recorded route telemetry through the token budget (hermes-agent#62).

Answers one question: how many turns would the configured ``per_turn`` limit
have stopped, under raw provider totals versus cost-weighted accounting?

Usage:
    python3 tools/replay_token_budget.py [TELEMETRY.jsonl] [--per-turn N]
                                         [--weight W] [--show N]

Defaults to ``~/.hermes/logs/adaptive-routing.jsonl``.  Reads only
``route_outcome`` events; each carries one turn's ``tokens`` breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.token_budget import TokenBudget  # noqa: E402

DEFAULT_LOG = os.path.expanduser("~/.hermes/logs/adaptive-routing.jsonl")


def load_turns(path):
    turns = []
    with open(path) as fh:
        for line in fh:
            if '"route_outcome"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "route_outcome":
                continue
            tokens = row.get("tokens")
            if isinstance(tokens, dict) and tokens:
                turns.append(row)
    return turns


def charge(tokens, weight, per_turn):
    """Charge one turn to a fresh budget; return (weighted, raw, breached)."""
    budget = TokenBudget(per_turn=per_turn, cache_read_weight=weight)
    budget.record(tokens)
    return budget.turn_tokens, budget.turn_raw_tokens, budget.breach() is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=DEFAULT_LOG)
    ap.add_argument("--per-turn", type=int, default=1_500_000)
    ap.add_argument("--weight", type=float, default=0.1)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"no telemetry at {args.log}", file=sys.stderr)
        return 2

    turns = load_turns(args.log)
    if not turns:
        print(f"no route_outcome events in {args.log}", file=sys.stderr)
        return 2

    raw_hits, new_hits, rows = [], [], []
    for row in turns:
        tokens = row["tokens"]
        # weight=1.0 reproduces the pre-fix accounting exactly.
        old_charge, raw, old_breach = charge(tokens, 1.0, args.per_turn)
        new_charge, _, new_breach = charge(tokens, args.weight, args.per_turn)
        if old_breach:
            raw_hits.append(row)
        if new_breach:
            new_hits.append(row)
        rows.append((row, old_charge, new_charge, old_breach, new_breach))

    total = len(turns)
    print(f"telemetry:      {args.log}")
    print(f"turns replayed: {total}")
    print(f"per_turn limit: {args.per_turn:,}   cache_read_weight: {args.weight:g}")
    print()
    print(f"  breaches, raw totals (weight 1.0): {len(raw_hits):4d}  "
          f"({len(raw_hits) / total * 100:.1f}%)")
    print(f"  breaches, cost-weighted:           {len(new_hits):4d}  "
          f"({len(new_hits) / total * 100:.1f}%)")
    print(f"  false positives removed:           {len(raw_hits) - len(new_hits):4d}")

    still = [r for r in rows if r[4]]
    if still:
        print(f"\nstill stopping ({len(still)}) — these are real:")
        for row, old_c, new_c, _, _ in still[: args.show]:
            print(f"  {row['created_at'][:16]}  class={row.get('class')}  "
                  f"calls={row.get('api_calls')}  charged={new_c:,}  raw={old_c:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
