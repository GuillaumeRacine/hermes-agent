"""New tool imports beside a cached old store without enabling old claims."""
import importlib
import json
from unittest.mock import Mock

import pytest

from cron import jobs


@pytest.fixture(autouse=True)
def restore_tool_bindings(monkeypatch):
    yield
    monkeypatch.undo()
    import sys
    tool = sys.modules.get("tools.cronjob_tools")
    if tool is not None:
        importlib.reload(tool)


@pytest.mark.parametrize("alias", ["run", "run_now", "trigger"])
def test_old_cached_jobs_allows_import_and_list_but_run_fails_closed(tmp_path, monkeypatch, alias):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", directory)
    monkeypatch.setattr(jobs, "JOBS_FILE", directory / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", directory / "output")
    job = jobs.create_job(prompt="offline synthetic compatibility test", schedule="every 5m", name="compat")
    before = jobs.JOBS_FILE.read_bytes()
    from cron import scheduler
    run = Mock(side_effect=AssertionError("must not execute"))
    legacy = Mock(side_effect=AssertionError("must not claim via legacy API"))
    mark = Mock(side_effect=AssertionError("must not record or clear claims"))
    monkeypatch.setattr(scheduler, "run_one_job", run)
    monkeypatch.setattr(jobs, "claim_job_for_fire", legacy)
    monkeypatch.setattr(jobs, "mark_job_run", mark)
    monkeypatch.delattr(jobs, "claim_job_for_fire_result")
    # Simulate first load of new tool source with old jobs already cached.
    from tools import cronjob_tools
    importlib.reload(cronjob_tools)
    listed = json.loads(cronjob_tools.cronjob(action="list"))
    assert listed["success"] is True
    assert job["id"] in json.dumps(listed)
    response = json.loads(cronjob_tools.cronjob(action=alias, job_id=job["id"]))
    assert response["success"] is False
    assert "restart" in response["error"].lower()
    assert response["job"]["executed"] is False
    assert "execution_skipped" not in response["job"]
    assert jobs.JOBS_FILE.read_bytes() == before
    run.assert_not_called()
    legacy.assert_not_called()
    mark.assert_not_called()
