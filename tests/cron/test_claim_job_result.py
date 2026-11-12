"""Regression tests for the structured atomic fire-claim result API.

``claim_job_for_fire_result`` returns exactly one of "claimed" / "missing" /
"paused" / "disabled" / "already_claimed". Real-store tests against an
isolated tmp HERMES_HOME; no mocks of the API under test.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cron import jobs


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolated HERMES_HOME plus module-cached paths (real store, no real home)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", directory)
    monkeypatch.setattr(jobs, "JOBS_FILE", directory / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", directory / "output")
    return tmp_path


def create():
    """Fresh recurring job per case so no claim state leaks between tests."""
    return jobs.create_job(prompt="synthetic offline test", schedule="every 5m", name="claim-result")


def test_paused_job_returns_paused_and_preserves_bytes(store):
    job = create()
    jobs.pause_job(job["id"])
    before = jobs.JOBS_FILE.read_bytes()
    result = jobs.claim_job_for_fire_result(job["id"])
    assert result == "paused"
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_disabled_job_returns_disabled_and_preserves_bytes(store):
    job = create()
    jobs.update_job(job["id"], {"enabled": False, "state": "scheduled"})
    before = jobs.JOBS_FILE.read_bytes()
    result = jobs.claim_job_for_fire_result(job["id"])
    assert result == "disabled"
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_paused_wins_over_disabled(store):
    """pause_job sets paused and disabled; paused precedence must win."""
    job = create()
    jobs.pause_job(job["id"])
    stored = jobs.get_job(job["id"])
    assert stored.get("state") == "paused"
    assert stored.get("enabled") is False
    before = jobs.JOBS_FILE.read_bytes()
    assert jobs.claim_job_for_fire_result(job["id"]) == "paused"
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_missing_returns_missing_and_preserves_bytes(store):
    job = create()
    jobs.save_jobs([])
    before = jobs.JOBS_FILE.read_bytes()
    assert jobs.claim_job_for_fire_result("nope-does-not-exist") == "missing"
    assert jobs.JOBS_FILE.read_bytes() == before


@pytest.mark.parametrize("setup,expected", [
    (lambda jid: jobs.pause_job(jid), "paused"),
    (lambda jid: jobs.update_job(jid, {"enabled": False, "state": "scheduled"}), "disabled"),
])
def test_blocked_states_return_reason_and_preserve_bytes(store, setup, expected):
    job = create()
    setup(job["id"])
    before = jobs.JOBS_FILE.read_bytes()
    assert jobs.claim_job_for_fire_result(job["id"]) == expected
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_fresh_claim_then_already_claimed(store):
    """An active normal job claims exactly once; second attempt reports
    already_claimed and claim bytes are preserved by the losing call."""
    job = create()
    jid = job["id"]
    assert jobs.claim_job_for_fire_result(jid) == "claimed"
    assert jobs.get_job(jid).get("fire_claim") is not None
    after_claim = jobs.JOBS_FILE.read_bytes()
    assert jobs.claim_job_for_fire_result(jid) == "already_claimed"
    assert jobs.JOBS_FILE.read_bytes() == after_claim


def test_stale_claim_with_ttl0_is_reclaimable(store):
    job = create()
    jid = job["id"]
    assert jobs.claim_job_for_fire_result(jid) == "claimed"
    assert jobs.claim_job_for_fire_result(jid, claim_ttl_seconds=0) == "claimed"


def test_bool_wrapper_literal_booleans(store):
    """claim_job_for_fire is a bool wrapper: exact True only for claimed."""
    job = create()
    jid = job["id"]
    assert jobs.claim_job_for_fire(jid) is True
    assert jobs.claim_job_for_fire(jid) is False

    blocked = create()
    jobs.update_job(blocked["id"], {"enabled": False, "state": "scheduled"})
    assert jobs.claim_job_for_fire(blocked["id"]) is False

    assert jobs.claim_job_for_fire("nope-does-not-exist") is False


def test_concurrent_claims_exactly_one_wins(store):
    """Two threads racing the same fresh job: one claimed, one already_claimed."""
    job = create()
    jid = job["id"]
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait(timeout=5)
        return jobs.claim_job_for_fire_result(jid)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        results = sorted(f.result(timeout=10) for f in futures)

    assert results == ["already_claimed", "claimed"]
    assert jobs.get_job(jid).get("fire_claim") is not None
