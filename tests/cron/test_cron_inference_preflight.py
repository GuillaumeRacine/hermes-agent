"""run_job() must refuse a job BEFORE its pre-run script when the agent step cannot run.

hermes-home #241: the drift guard used to fire after the pre-run script, so
``briefs_collect.py`` advanced its cursor for a digest that was then skipped.
"""

import time

import pytest

from cron import scheduler
from hermes_cli.provider_circuits import record_failure


def _cfg(tmp_path, **extra):
    cfg = {
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "provider_circuits": {"enabled": True, "state_path": str(tmp_path / "circuits.json")},
    }
    cfg.update(extra)
    return cfg


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(scheduler, "_load_cron_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {"provider": (requested or cfg["model"]["provider"]), "api_key": "k"},
    )
    return cfg


def test_no_agent_jobs_are_never_preflighted(runtime):
    assert scheduler._inference_preflight_error({"id": "j", "no_agent": True, "script": "x.sh"}) is None


def test_pinned_job_on_healthy_provider_passes(runtime):
    job = {"id": "j", "provider": "openai-codex", "model": "gpt-5.6-sol"}
    assert scheduler._inference_preflight_error(job) is None


def test_unpinned_drift_is_refused_before_script(runtime):
    job = {"id": "j", "provider_snapshot": "openai-codex", "model_snapshot": "gpt-5.5"}
    err = scheduler._inference_preflight_error(job)
    assert err and "drifted" in err and "did not run" in err
    assert "model 'gpt-5.5' -> 'gpt-5.6-sol'" in err


def test_unpinned_without_drift_passes(runtime):
    job = {"id": "j", "provider_snapshot": "openai-codex", "model_snapshot": "gpt-5.6-sol"}
    assert scheduler._inference_preflight_error(job) is None


def test_open_primary_with_no_fallback_is_refused(runtime, tmp_path):
    record_failure("openai-codex", "gpt-5.6-sol", "rate_limit",
                   path=tmp_path / "circuits.json", retry_after_seconds=90_000)
    job = {"id": "j", "provider": "openai-codex", "model": "gpt-5.6-sol"}
    err = scheduler._inference_preflight_error(job)
    assert err and "circuit open" in err and "did not run" in err


def test_open_primary_with_closed_fallback_passes(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, fallback_providers=[{"provider": "zai", "model": "glm-5.2"}])
    monkeypatch.setattr(scheduler, "_load_cron_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {"provider": requested or "openai-codex"},
    )
    record_failure("openai-codex", "*", "billing", path=tmp_path / "circuits.json")
    job = {"id": "j", "provider": "openai-codex", "model": "gpt-5.6-sol"}
    assert scheduler._inference_preflight_error(job) is None


def test_open_primary_and_open_fallbacks_are_refused(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, fallback_providers=[{"provider": "zai", "model": "glm-5.2"}])
    monkeypatch.setattr(scheduler, "_load_cron_config", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {"provider": requested or "openai-codex"},
    )
    path = tmp_path / "circuits.json"
    record_failure("openai-codex", "gpt-5.6-sol", "rate_limit", path=path, retry_after_seconds=90_000)
    record_failure("zai", "glm-5.2", "rate_limit", path=path, retry_after_seconds=90_000)
    job = {"id": "j", "provider": "openai-codex", "model": "gpt-5.6-sol"}
    err = scheduler._inference_preflight_error(job)
    assert err and "every fallback provider's circuit is open" in err


def test_run_job_refuses_before_running_prerun_script(runtime, monkeypatch):
    def _script_must_not_run(path):
        raise AssertionError("pre-run script executed despite preflight refusal")

    monkeypatch.setattr(scheduler, "_run_job_script", _script_must_not_run)
    job = {
        "id": "ee88d8e8e151",
        "name": "Daily Briefs Digest",
        "prompt": "digest",
        "script": "briefs_collect.py",
        "provider_snapshot": "openai-codex",
        "model_snapshot": "gpt-5.5",
    }
    success, doc, final, error = scheduler.run_job(job)
    assert success is False
    assert final == ""
    assert error.startswith("RuntimeError: Skipped to prevent unintended spend")
    assert "global inference config drifted" in error
    assert "(FAILED)" in doc and "did not run" in doc


def test_first_closed_fallback_runtime_skips_open_entries(tmp_path):
    cfg = _cfg(
        tmp_path,
        fallback_providers=[
            {"provider": "zai", "model": "glm-5.2"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
        ],
    )
    record_failure("zai", "glm-5.2", "rate_limit", path=tmp_path / "circuits.json", retry_after_seconds=90_000)
    resolved = []

    def _resolve(requested=None, **kw):
        resolved.append(requested)
        return {"provider": requested}

    runtime, model = scheduler._first_closed_fallback_runtime(cfg, _resolve, skip_provider="openai-codex")
    assert runtime == {"provider": "anthropic"} and model == "claude-sonnet-4-6"
    assert resolved == ["anthropic"]
