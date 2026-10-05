"""Cron-test fixtures.

Provides a default ``HERMES_MODEL`` for cron run_job tests so each one
doesn't have to spell out a model. The global conftest blanks
HERMES_MODEL hermetically; without this autouse fixture every cron test
that exercises ``run_job`` would hit the fail-fast guard added in
``cron/scheduler.py`` (see issue #23979) and have to be rewritten.

Tests that specifically need ``HERMES_MODEL`` unset — model-resolution
edge cases — call ``monkeypatch.delenv("HERMES_MODEL", raising=False)``
inside the test, which overrides this fixture's value for that scope.
"""

import pytest


@pytest.fixture(autouse=True)
def _default_cron_test_model(monkeypatch):
    """Pin a default HERMES_MODEL so cron run_job tests have a resolvable model."""
    monkeypatch.setenv("HERMES_MODEL", "test-cron-default-model")
    yield


@pytest.fixture(autouse=True)
def _isolate_cron_store(tmp_path, monkeypatch):
    """Point every cron/jobs.py path constant at a per-test tempdir.

    cron/jobs.py resolves HERMES_DIR/JOBS_FILE at import time, so a test that
    only sets HERMES_HOME after the module is imported still writes the REAL
    ~/.hermes/cron/jobs.json. That leaked live "w" / "echo hi" every-5m agent
    jobs into the running gateway twice (2026-09-09: 6 jobs, 2026-10-05: 2 jobs,
    ~1.5M tokens/day). Tests that need specific paths still override these.
    """
    import cron.jobs as jobs_mod

    home = tmp_path / "_cron_home"
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "TICKER_HEARTBEAT_FILE", cron_dir / "ticker_heartbeat")
    monkeypatch.setattr(jobs_mod, "TICKER_SUCCESS_FILE", cron_dir / "ticker_last_success")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", cron_dir / "output")
    yield
