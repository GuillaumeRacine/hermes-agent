"""Truthful manual-run skips against a real, isolated cron store."""
import json
from unittest.mock import Mock

import pytest

from cron import jobs
from tools import cronjob_tools


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", directory)
    monkeypatch.setattr(jobs, "JOBS_FILE", directory / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", directory / "output")
    from cron import scheduler
    run = Mock(side_effect=AssertionError("a skipped job must never execute"))
    monkeypatch.setattr(scheduler, "run_one_job", run)
    yield run
    run.assert_not_called()


def create():
    return jobs.create_job(prompt="synthetic offline test", schedule="every 5m", name="skip-test")


def run_skip(job_id, action="run"):
    result = json.loads(cronjob_tools.cronjob(action=action, job_id=job_id))
    assert result["success"] is True
    assert result["job"]["executed"] is False
    assert result["job"]["execution_success"] is False
    assert result["job"]["execution_skipped"]
    return result["job"]["execution_skipped"].lower()


@pytest.mark.parametrize("action", ["run", "run_now", "trigger"])
def test_paused_aliases_leave_entire_store_unchanged(store, action):
    job = create()
    jobs.pause_job(job["id"])
    before = jobs.JOBS_FILE.read_bytes()
    message = run_skip(job["id"], action)
    assert "paused" in message
    assert "already" not in message
    assert jobs.JOBS_FILE.read_bytes() == before
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_disabled_job_is_not_reported_as_running(store):
    job = create()
    jobs.update_job(job["id"], {"enabled": False, "state": "scheduled"})
    before = jobs.JOBS_FILE.read_bytes()
    message = run_skip(job["id"])
    assert "disabled" in message
    assert "already" not in message
    assert jobs.JOBS_FILE.read_bytes() == before


def test_fresh_claim_is_preserved_and_reported(store):
    job = create()
    assert jobs.claim_job_for_fire(job["id"]) is True
    before = jobs.JOBS_FILE.read_bytes()
    message = run_skip(job["id"])
    assert "active fire claim" in message or "already" in message
    assert jobs.JOBS_FILE.read_bytes() == before


def test_deletion_after_initial_resolution_reports_missing(store, monkeypatch):
    job = create()
    original = cronjob_tools.resolve_job_ref

    def resolve_then_delete(ref):
        found = original(ref)
        jobs.save_jobs([])
        return found

    monkeypatch.setattr(cronjob_tools, "resolve_job_ref", resolve_then_delete)
    message = run_skip(job["id"])
    assert any(reason in message for reason in ("not found", "missing", "no longer exists"))
    assert "already" not in message
    assert jobs.load_jobs() == []


def test_claim_exception_cannot_mark_run_or_clear_another_claim(store, monkeypatch):
    job = create()
    assert jobs.claim_job_for_fire(job["id"]) is True
    before = jobs.JOBS_FILE.read_bytes()
    mark = Mock(side_effect=AssertionError("unowned claim must not be marked"))

    def fail_claim(*args, **kwargs):
        raise RuntimeError("synthetic claim failure")

    monkeypatch.setattr(cronjob_tools, "claim_job_for_fire_result", fail_claim)
    monkeypatch.setattr(cronjob_tools, "mark_job_run", mark)
    result = json.loads(cronjob_tools.cronjob(action="run", job_id=job["id"]))
    assert result["success"] is False
    assert result["job"]["executed"] is False
    assert result["job"]["execution_error"] == "synthetic claim failure"
    assert "execution_skipped" not in result["job"]
    mark.assert_not_called()
    assert jobs.JOBS_FILE.read_bytes() == before


def test_cli_prints_actual_paused_reason(store, monkeypatch, capsys):
    from hermes_cli import cron as cli
    job = create()
    jobs.pause_job(job["id"])
    before = jobs.JOBS_FILE.read_bytes()
    monkeypatch.setattr(cli, "_cron_api", lambda **kw: json.loads(cronjob_tools.cronjob(**kw)))
    assert cli._job_action("run", job["id"], "Triggered") == 0
    output = capsys.readouterr().out.lower()
    assert "job is paused; not run." in output
    assert "already being fired" not in output
    assert "ran now" not in output
    assert "triggered" not in output
    assert "skipped job:" in output
    assert jobs.JOBS_FILE.read_bytes() == before
