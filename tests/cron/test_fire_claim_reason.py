"""Frozen eligibility contract; the caller owns locking and claim persistence."""
import copy
from datetime import datetime, timedelta, timezone

import pytest

from cron.fire_claim import fire_claim_blocker


NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def blocker(job, **kwargs):
    return fire_claim_blocker(job, now=NOW, ensure_aware=aware, **kwargs)


@pytest.mark.parametrize("job, expected", [
    (None, "missing"),
    ({}, None),
    ({"enabled": True}, None),
    ({"enabled": False}, "disabled"),
    ({"state": "paused", "enabled": False}, "paused"),
    ({"state": "paused", "enabled": True}, "paused"),
    ({"enabled": False, "fire_claim": {"at": NOW.isoformat()}}, "disabled"),
    ({"state": "paused", "fire_claim": {"at": NOW.isoformat()}}, "paused"),
])
def test_eligibility_and_reason_precedence(job, expected):
    assert blocker(job) == expected


@pytest.mark.parametrize("age, expected", [
    (-60, "already_claimed"),
    (0, "already_claimed"),
    (299.999999, "already_claimed"),
    (300, None),
    (301, None),
])
def test_default_ttl_boundary_and_future_claim(age, expected):
    job = {"fire_claim": {"at": (NOW - timedelta(seconds=age)).isoformat()}}
    assert blocker(job) == expected


def test_custom_ttl_and_zero_ttl():
    job = {"fire_claim": {"at": (NOW - timedelta(seconds=10)).isoformat()}}
    assert blocker(job, claim_ttl_seconds=11) == "already_claimed"
    assert blocker(job, claim_ttl_seconds=10) is None
    assert blocker({"fire_claim": {"at": NOW.isoformat()}}, claim_ttl_seconds=0) is None


@pytest.mark.parametrize("claim", [None, {}, False, "bad", ["bad"],
                                  {"at": None}, {"at": 123}, {"at": "not-a-date"}])
def test_malformed_claim_does_not_block(claim):
    assert blocker({"fire_claim": claim}) is None


def test_normalizer_receives_parsed_naive_time_and_controls_comparison():
    seen = []

    def normalize(value):
        seen.append(value)
        # Legacy system-local time is UTC-5 in this fixture, not UTC.
        return value.replace(tzinfo=timezone(timedelta(hours=-5)))

    result = fire_claim_blocker(
        {"fire_claim": {"at": "2026-11-12T06:59:59"}},
        now=NOW, ensure_aware=normalize,
    )
    assert result == "already_claimed"
    assert seen == [datetime(2026, 11, 12, 6, 59, 59)]


def test_offset_claim_compares_instants():
    assert blocker({"fire_claim": {"at": "2026-11-12T07:00:00-05:00"}}) == "already_claimed"


def test_normalizer_exception_preserves_malformed_claim_fallback():
    def broken(_value):
        raise RuntimeError("synthetic normalization failure")

    assert fire_claim_blocker(
        {"fire_claim": {"at": NOW.isoformat()}}, now=NOW, ensure_aware=broken,
    ) is None


@pytest.mark.parametrize("age", [0, 600])
def test_decision_never_mutates_job_or_nested_claim(age):
    job = {
        "id": "synthetic-job", "enabled": True, "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "unchanged", "last_run_at": None,
        "fire_claim": {"at": (NOW - timedelta(seconds=age)).isoformat(), "by": "synthetic"},
    }
    original = copy.deepcopy(job)
    claim = job["fire_claim"]
    blocker(job)
    assert job == original
    assert job["fire_claim"] is claim
