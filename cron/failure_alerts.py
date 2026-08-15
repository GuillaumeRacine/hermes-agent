"""Consecutive-failure suppression for cron alert delivery.

Before this, ``run_one_job`` delivered an alert on *every* failed run. A job
that failed identically every 15 minutes emitted 96 messages/day forever and
nothing ever paused it — see issue #32, where one broken job posted 52
byte-identical Slack alerts over 13 hours.

This tracks a per-job consecutive-failure streak keyed on a stable signature of
the error, and delivers on an exponential schedule (failures 1, 2, 4, 8, 16, …)
instead of every run. After ``AUTO_PAUSE_AFTER`` consecutive *identical*
failures the job is paused with one final message, so a permanently broken job
stops billing the channel.

A *changed* signature resets the streak: a job whose failure mode changes is
news, and says so immediately.

Recovery announcements are deliberately not handled here — the cron error
watchdog already reports those in its "Recovered" section.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Deliver when the streak hits one of these counts; suppress otherwise.
# Powers of two keep early failures visible while collapsing a long outage:
# 52 identical failures become 6 messages instead of 52.
_NOTIFY_AT = frozenset((1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024))

# Consecutive identical failures before the job is paused outright.
# At */15 this is ~5 hours of a job failing the same way every time.
AUTO_PAUSE_AFTER = 20

# Bound the signature input so a multi-KB provider blob can't bloat the state
# file or slow the regex.
_SIGNATURE_INPUT_LIMIT = 500


def _state_path() -> Path:
    """Resolve the streak file next to the other cron state.

    Read at call time, not import time, so tests and a relocated HERMES_HOME
    both land in the right place.
    """
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "cron" / "failure_streaks.json"


def failure_signature(job: dict, error: str | None) -> str:
    """A signature that is stable across repeated runs of the same failure.

    Deliberately excludes timestamps, PIDs and run counters so that "the same
    thing broke again" collapses, while a genuinely different error does not.
    """
    text = (error or "unknown error").strip()[:_SIGNATURE_INPUT_LIMIT]
    text = re.sub(r"\s+", " ", text)
    # Wall-clock stamps and bare PIDs differ every run but mean nothing here.
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "<ts>", text)
    text = re.sub(r"\bpid[= ]\d+", "pid=<pid>", text, flags=re.IGNORECASE)
    parts = [str(job.get("name") or ""), str(job.get("script") or ""), text]
    return "\x1f".join(parts)


@dataclass(frozen=True)
class FailureDecision:
    """What the caller should do about this failure."""

    deliver: bool
    streak: int
    first_seen: str
    should_pause: bool
    suppressed: int  # failures silently swallowed since the last delivery

    def summary_suffix(self) -> str:
        """A short, actionable tail for the delivered alert.

        The streak and its start time are the part an operator can act on; a
        truncated stack trace repeated 52 times is not.
        """
        if self.streak <= 1:
            return ""
        extra = ""
        if self.suppressed > 0:
            extra = f", {self.suppressed} identical alert{'s' if self.suppressed != 1 else ''} suppressed"
        return f" (failed {self.streak}x consecutively since {self.first_seen}{extra})"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        # A corrupt or absent streak file must never block delivery — losing
        # the streak means we alert too often, which is the safe direction.
        return {}


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _mutate(fn):
    """Run ``fn(state) -> result`` under an exclusive lock.

    The gateway ticks cron from multiple threads, so a bare read-modify-write
    would drop concurrent updates. The lock file is separate from the state
    file because the atomic ``os.replace`` swaps the inode out from under any
    descriptor held on the state file itself.
    """
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            state = _load(path)
            result, changed = fn(state)
            if changed:
                _atomic_write(path, state)
            return result
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def record_failure(job: dict, error: str | None, *, now: datetime | None = None) -> FailureDecision:
    """Record one failed run and decide whether to alert about it."""
    job_id = str(job.get("id") or job.get("name") or "unknown")
    signature = failure_signature(job, error)
    stamp = (now or datetime.now()).isoformat(timespec="seconds")

    def apply(state: dict):
        entry = state.get(job_id)
        if not isinstance(entry, dict) or entry.get("signature") != signature:
            # New job, or the failure mode changed — restart the streak and
            # always speak up.
            entry = {
                "signature": signature,
                "streak": 0,
                "first_seen": stamp,
                "last_notified_streak": 0,
            }

        entry["streak"] = int(entry.get("streak", 0)) + 1
        streak = entry["streak"]
        # Exactly ``==``, not ``>=``: if the pause itself fails the job keeps
        # running, and ``>=`` would then alert on every single subsequent run —
        # reintroducing the storm this module exists to stop. Past the
        # threshold we fall back to the ordinary backoff schedule.
        should_pause = streak == AUTO_PAUSE_AFTER
        deliver = should_pause or streak in _NOTIFY_AT

        suppressed = 0
        if deliver:
            suppressed = max(0, streak - int(entry.get("last_notified_streak", 0)) - 1)
            entry["last_notified_streak"] = streak

        state[job_id] = entry
        return (
            FailureDecision(
                deliver=deliver,
                streak=streak,
                first_seen=str(entry.get("first_seen") or stamp),
                should_pause=should_pause,
                suppressed=suppressed,
            ),
            True,
        )

    return _mutate(apply)


def clear_failure(job: dict) -> int:
    """Forget a job's streak after a successful run.

    Returns the streak that was cleared (0 if there was none), so the caller
    can tell a first-time success from a recovery.
    """
    job_id = str(job.get("id") or job.get("name") or "unknown")

    def apply(state: dict):
        entry = state.pop(job_id, None)
        if not isinstance(entry, dict):
            return 0, False
        return int(entry.get("streak", 0)), True

    return _mutate(apply)
