"""Claim infrastructure failures are errors, never successful runs or benign skips."""
import json
from unittest.mock import Mock

import pytest

from cron import jobs, scheduler
from tools import cronjob_tools


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    run = Mock()
    mark = Mock()
    monkeypatch.setattr(scheduler, "run_one_job", run)
    monkeypatch.setattr(cronjob_tools, "mark_job_run", mark)
    yield jobs.create_job(prompt="offline failure probe", schedule="every 5m")
    run.assert_not_called()
    mark.assert_not_called()


def fail_claim(*args, **kwargs):
    raise OSError("synthetic store failure")


def test_claim_failure_is_error_without_recording_a_run(store, monkeypatch):
    assert jobs.claim_job_for_fire(store["id"]) is True
    before = jobs.JOBS_FILE.read_bytes()
    monkeypatch.setattr(cronjob_tools, "claim_job_for_fire_result", fail_claim)
    result = json.loads(cronjob_tools.cronjob(action="run", job_id=store["id"]))
    assert result["success"] is False
    assert result["error"] == "synthetic store failure"
    assert result["job"]["execution_error"] == result["error"]
    assert result["job"]["executed"] is False
    assert "execution_skipped" not in result["job"]
    assert jobs.JOBS_FILE.read_bytes() == before


def test_cli_claim_error_exits_nonzero(store, monkeypatch, capsys):
    from hermes_cli import cron as cli
    monkeypatch.setattr(cronjob_tools, "claim_job_for_fire_result", fail_claim)
    monkeypatch.setattr(cli, "_cron_api", lambda **kw: json.loads(cronjob_tools.cronjob(**kw)))
    assert cli._job_action("run", store["id"], "Triggered") == 1
    output = capsys.readouterr().out
    assert "Failed to run job: synthetic store failure" in output
    assert "Ran now" not in output


def test_post_persistence_failure_keeps_claim_for_ttl_recovery(store, monkeypatch):
    original = jobs.save_jobs
    # Pin the schedule advance; it remains persisted on an uncertain outcome.
    monkeypatch.setattr(jobs, "compute_next_run", lambda *args: "2099-01-01T00:00:00+00:00")

    def persist_then_fail(value):
        original(value)
        raise OSError("synthetic post-write failure")

    monkeypatch.setattr(jobs, "save_jobs", persist_then_fail)
    result = json.loads(cronjob_tools.cronjob(action="run", job_id=store["id"]))
    assert result["success"] is False
    assert result["job"]["executed"] is False
    assert "post-write failure" in result["error"]
    saved = jobs.get_job(store["id"])
    assert saved["fire_claim"]
    assert saved["next_run_at"] == "2099-01-01T00:00:00+00:00"
    assert saved.get("last_run_at") is None
    # Ownership is uncertain after failure: preserve the claim instead of clearing
    # another caller's work. Existing TTL recovery still works once storage recovers.
    monkeypatch.setattr(jobs, "save_jobs", original)
    assert jobs.claim_job_for_fire_result(store["id"]) == "already_claimed"
    assert jobs.claim_job_for_fire_result(store["id"], claim_ttl_seconds=0) == "claimed"


def test_save_failure_before_replace_preserves_store(store, monkeypatch):
    before = jobs.JOBS_FILE.read_bytes()
    def fail_replace(*args, **kwargs):
        raise OSError("synthetic replace failure")
    monkeypatch.setattr(jobs, "atomic_replace", fail_replace)
    result = json.loads(cronjob_tools.cronjob(action="run", job_id=store["id"]))
    assert result["success"] is False
    assert result["job"]["executed"] is False
    assert result["error"] == "synthetic replace failure"
    assert "execution_skipped" not in result["job"]
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(store["id"]).get("fire_claim") is None
