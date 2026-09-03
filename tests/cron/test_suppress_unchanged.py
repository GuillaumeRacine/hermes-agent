"""Tests for the opt-in ``suppress_unchanged`` cron delivery mode.

Covers the pure helpers (content hashing + suppression decision) and the
persistence wiring in ``create_job`` / ``mark_job_run``. Delivery-path
integration lives in the scheduler tests; here we lock down the building
blocks so the "post only on state change" contract can't silently regress.
"""

import pytest

import cron.scheduler as s
from cron.jobs import (
    content_delivery_hash,
    should_suppress_unchanged_delivery,
    create_job,
    load_jobs,
    mark_job_run,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# =========================================================================
# content_delivery_hash
# =========================================================================

class TestContentDeliveryHash:
    def test_stable_for_identical_text(self):
        assert content_delivery_hash("recovery partial (7 left)") == \
            content_delivery_hash("recovery partial (7 left)")

    def test_differs_on_change(self):
        assert content_delivery_hash("recovery partial (7 left)") != \
            content_delivery_hash("recovery partial (6 left)")

    def test_is_sha256_hexdigest(self):
        digest = content_delivery_hash("x")
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_none_and_empty_are_equivalent(self):
        # Defensive: helper coerces None to "" so callers never crash.
        assert content_delivery_hash(None) == content_delivery_hash("")


# =========================================================================
# should_suppress_unchanged_delivery
# =========================================================================

class TestShouldSuppressUnchangedDelivery:
    CONTENT = "recovery partial (7 left)"

    def test_false_when_flag_off(self):
        job = {"last_delivered_hash": content_delivery_hash(self.CONTENT)}
        assert should_suppress_unchanged_delivery(job, self.CONTENT) is False

    def test_false_when_flag_off_explicit(self):
        job = {
            "suppress_unchanged": False,
            "last_delivered_hash": content_delivery_hash(self.CONTENT),
        }
        assert should_suppress_unchanged_delivery(job, self.CONTENT) is False

    def test_false_on_changed_content(self):
        job = {
            "suppress_unchanged": True,
            "last_delivered_hash": content_delivery_hash(self.CONTENT),
        }
        assert should_suppress_unchanged_delivery(job, "recovery partial (6 left)") is False

    def test_true_on_identical_content_with_flag_on(self):
        job = {
            "suppress_unchanged": True,
            "last_delivered_hash": content_delivery_hash(self.CONTENT),
        }
        assert should_suppress_unchanged_delivery(job, self.CONTENT) is True

    def test_false_on_empty_content(self):
        # Empty content is never suppressed (and hashing "" must not match a
        # real prior body).
        job = {"suppress_unchanged": True, "last_delivered_hash": content_delivery_hash("")}
        assert should_suppress_unchanged_delivery(job, "") is False

    def test_false_when_no_prior_hash(self):
        # First run: nothing delivered yet, so nothing to suppress.
        job = {"suppress_unchanged": True, "last_delivered_hash": None}
        assert should_suppress_unchanged_delivery(job, self.CONTENT) is False

    def test_legacy_job_without_keys_returns_false(self):
        # Pre-existing job record missing both new keys must not crash.
        assert should_suppress_unchanged_delivery({}, self.CONTENT) is False


# =========================================================================
# create_job defaults
# =========================================================================

class TestCreateJobDefaults:
    def test_defaults_off(self, tmp_cron_dir):
        job = create_job(prompt="status", schedule="every 5m")
        assert job["suppress_unchanged"] is False
        assert job["last_delivered_hash"] is None

    def test_opt_in(self, tmp_cron_dir):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=True)
        assert job["suppress_unchanged"] is True
        assert job["last_delivered_hash"] is None

    def test_loose_truthy_normalized_to_bool(self, tmp_cron_dir):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=1)
        assert job["suppress_unchanged"] is True


# =========================================================================
# mark_job_run delivered_hash persistence
# =========================================================================

class TestMarkJobRunDeliveredHash:
    def _get(self, job_id):
        return next(j for j in load_jobs() if j["id"] == job_id)

    def test_persists_hash_when_passed(self, tmp_cron_dir):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=True)
        digest = content_delivery_hash("recovery partial (7 left)")
        mark_job_run(job["id"], True, delivered_hash=digest)
        assert self._get(job["id"])["last_delivered_hash"] == digest

    def test_leaves_hash_unchanged_when_none(self, tmp_cron_dir):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=True)
        digest = content_delivery_hash("recovery partial (7 left)")
        mark_job_run(job["id"], True, delivered_hash=digest)
        # A subsequent suppressed tick passes delivered_hash=None → unchanged.
        mark_job_run(job["id"], True, delivered_hash=None)
        assert self._get(job["id"])["last_delivered_hash"] == digest

    def test_default_call_does_not_touch_hash(self, tmp_cron_dir):
        # Jobs that never opted in keep last_delivered_hash=None across runs.
        job = create_job(prompt="status", schedule="every 5m")
        mark_job_run(job["id"], True)
        assert self._get(job["id"])["last_delivered_hash"] is None


# =========================================================================
# Scheduler wiring (run_one_job end-to-end)
# =========================================================================

class TestSchedulerSuppressWiring:
    """Drive run_one_job through the real jobs store to prove the delivery
    path skips identical reposts for opted-in jobs and never touches
    default-off jobs."""

    def _patch(self, monkeypatch, body, deliver_calls, *, success=True):
        monkeypatch.setattr(
            s, "run_job", lambda job: (success, "# out", body, None)
        )
        monkeypatch.setattr(s, "save_job_output", lambda jid, out: "/tmp/o.md")

        def fake_deliver(job, content, adapters=None, loop=None):
            deliver_calls.append(content)
            return None

        monkeypatch.setattr(s, "_deliver_result", fake_deliver)

    def _get(self, job_id):
        return next(j for j in load_jobs() if j["id"] == job_id)

    def test_second_identical_tick_is_suppressed(self, tmp_cron_dir, monkeypatch):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=True)
        deliver_calls = []
        self._patch(monkeypatch, "recovery partial (7 left)", deliver_calls)

        # First tick delivers and records the hash.
        s.run_one_job(self._get(job["id"]))
        assert deliver_calls == ["recovery partial (7 left)"]
        assert self._get(job["id"])["last_delivered_hash"] == \
            content_delivery_hash("recovery partial (7 left)")

        # Second tick, identical body → suppressed (no new delivery).
        s.run_one_job(self._get(job["id"]))
        assert deliver_calls == ["recovery partial (7 left)"]  # unchanged

    def test_changed_body_delivers_and_updates_hash(self, tmp_cron_dir, monkeypatch):
        job = create_job(prompt="status", schedule="every 5m", suppress_unchanged=True)
        deliver_calls = []
        self._patch(monkeypatch, "recovery partial (7 left)", deliver_calls)
        s.run_one_job(self._get(job["id"]))

        # Now the body changes → must deliver and re-record.
        self._patch(monkeypatch, "recovery partial (6 left)", deliver_calls)
        s.run_one_job(self._get(job["id"]))
        assert deliver_calls == [
            "recovery partial (7 left)",
            "recovery partial (6 left)",
        ]
        assert self._get(job["id"])["last_delivered_hash"] == \
            content_delivery_hash("recovery partial (6 left)")

    def test_default_off_job_never_records_hash_or_suppresses(self, tmp_cron_dir, monkeypatch):
        job = create_job(prompt="status", schedule="every 5m")  # flag off
        deliver_calls = []
        self._patch(monkeypatch, "recovery partial (7 left)", deliver_calls)

        s.run_one_job(self._get(job["id"]))
        s.run_one_job(self._get(job["id"]))

        # Delivered both times; hash never recorded (no churn for opted-out jobs).
        assert deliver_calls == [
            "recovery partial (7 left)",
            "recovery partial (7 left)",
        ]
        assert self._get(job["id"])["last_delivered_hash"] is None
