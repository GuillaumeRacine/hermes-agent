"""Per-job max_iterations is honored and capped by config (Context#84 fix 3)."""
from cron.scheduler import _resolve_job_max_iterations


def test_defaults_to_config_when_job_has_no_budget():
    assert _resolve_job_max_iterations({}, {"agent": {"max_turns": 60}}) == 60
    assert _resolve_job_max_iterations({"max_iterations": None}, {}) == 90


def test_job_budget_wins_but_never_exceeds_config_cap():
    cfg = {"agent": {"max_turns": 60}}
    assert _resolve_job_max_iterations({"max_iterations": 12}, cfg) == 12
    assert _resolve_job_max_iterations({"max_iterations": "25"}, cfg) == 25
    assert _resolve_job_max_iterations({"max_iterations": 500}, cfg) == 60


def test_garbage_budget_falls_back_to_config():
    assert _resolve_job_max_iterations({"max_iterations": "lots"}, {"max_turns": 40}) == 40
    assert _resolve_job_max_iterations({"max_iterations": 0}, {"max_turns": 40}) == 40
