"""Tests for consecutive-failure alert suppression (#32)."""

import json

import pytest

from cron.failure_alerts import (
    AUTO_PAUSE_AFTER,
    clear_failure,
    failure_signature,
    record_failure,
)

JOB = {"id": "591570e2c4af", "name": "EnvoyMusic order notifications", "script": "envoy_order_notify.sh"}
ERROR = "Script exited with code 1\nstdout:\nError: Cannot find module 'order-notify.mjs'"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "cron").mkdir()
    return tmp_path


def test_first_failure_always_delivers():
    decision = record_failure(JOB, ERROR)
    assert decision.deliver is True
    assert decision.streak == 1
    assert decision.suppressed == 0
    # Nothing useful to append on a first failure.
    assert decision.summary_suffix() == ""


def test_identical_failures_back_off_exponentially():
    alerted = [d.streak for d in (record_failure(JOB, ERROR) for _ in range(16)) if d.deliver]
    assert alerted == [1, 2, 4, 8, 16]


def test_the_incident_52_failures_collapse_to_a_handful():
    """The exact shape of #32: 52 identical failures every 15 minutes."""
    decisions = [record_failure(JOB, ERROR) for _ in range(52)]
    alerted = [d.streak for d in decisions if d.deliver]
    # 1, 2, 4, 8, 16 by backoff, plus the auto-pause run at AUTO_PAUSE_AFTER.
    assert alerted == [1, 2, 4, 8, 16, AUTO_PAUSE_AFTER, 32]
    assert len(alerted) < 8, "52 alerts must collapse to single digits"


def test_auto_pause_triggers_once_the_streak_is_hopeless():
    decisions = [record_failure(JOB, ERROR) for _ in range(AUTO_PAUSE_AFTER)]
    assert not any(d.should_pause for d in decisions[:-1])
    final = decisions[-1]
    assert final.should_pause is True
    assert final.deliver is True, "the pause message must always be delivered"


def test_suppressed_count_is_reported_on_the_next_alert():
    for _ in range(3):
        record_failure(JOB, ERROR)
    fourth = record_failure(JOB, ERROR)
    assert fourth.deliver is True and fourth.streak == 4
    # Failures 3 was swallowed between the alert at 2 and this one at 4.
    assert fourth.suppressed == 1
    assert "failed 4x consecutively" in fourth.summary_suffix()
    assert "1 identical alert suppressed" in fourth.summary_suffix()


def test_a_changed_failure_mode_resets_the_streak_and_speaks_up():
    for _ in range(5):
        record_failure(JOB, ERROR)
    different = record_failure(JOB, "Script timed out after 300s")
    assert different.streak == 1
    assert different.deliver is True, "a new failure mode is news"


def test_success_clears_the_streak():
    for _ in range(5):
        record_failure(JOB, ERROR)
    assert clear_failure(JOB) == 5
    assert clear_failure(JOB) == 0
    # Next failure starts over and alerts immediately.
    assert record_failure(JOB, ERROR).streak == 1


def test_jobs_are_tracked_independently():
    other = {"id": "other", "name": "Other job", "script": "other.py"}
    for _ in range(4):
        record_failure(JOB, ERROR)
    assert record_failure(other, ERROR).streak == 1


def test_signature_ignores_timestamps_and_pids():
    a = failure_signature(JOB, "boom at 2026-08-15T00:15:00-04:00 pid=123")
    b = failure_signature(JOB, "boom at 2026-08-15T13:45:00-04:00 pid=987")
    assert a == b, "same failure at a different time is the same failure"


def test_signature_distinguishes_real_differences():
    assert failure_signature(JOB, "module not found") != failure_signature(JOB, "timeout")


def test_state_survives_a_corrupt_file(isolated_home):
    (isolated_home / "cron" / "failure_streaks.json").write_text("{not json")
    decision = record_failure(JOB, ERROR)
    assert decision.deliver is True, "a corrupt streak file must fail open"
    assert decision.streak == 1


def test_state_file_is_valid_json_after_writes(isolated_home):
    for _ in range(3):
        record_failure(JOB, ERROR)
    payload = json.loads((isolated_home / "cron" / "failure_streaks.json").read_text())
    assert payload[JOB["id"]]["streak"] == 3


def test_concurrent_writers_do_not_lose_updates(isolated_home):
    """Two processes failing the same job must not clobber each other's count.

    Threads share the interpreter, so this exercises the flock path across
    real processes instead.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[2])
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from cron.failure_alerts import record_failure;"
        "[record_failure({'id':'race','name':'r','script':'s'}, 'boom') for _ in range(25)]"
    ) % repo
    procs = [
        subprocess.Popen([sys.executable, "-c", code], env={**__import__("os").environ})
        for _ in range(4)
    ]
    for p in procs:
        assert p.wait(timeout=60) == 0

    payload = json.loads((isolated_home / "cron" / "failure_streaks.json").read_text())
    assert payload["race"]["streak"] == 100, "lost updates: flock is not holding"
