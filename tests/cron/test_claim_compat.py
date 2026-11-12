"""Owner-frozen contract for a fail-closed, late-bound claim adapter."""
import importlib
from unittest.mock import Mock

import pytest

from cron import jobs


@pytest.mark.parametrize("value", ["claimed", "missing", "paused", "disabled", "already_claimed"])
def test_callable_result_is_delegated_unchanged(monkeypatch, value):
    from cron.claim_compat import claim_job_for_fire_result
    call = Mock(return_value=value)
    monkeypatch.setattr(jobs, "claim_job_for_fire_result", call)
    assert claim_job_for_fire_result("synthetic-id") == value
    call.assert_called_once_with("synthetic-id")


def test_dispatch_uses_current_api_each_call(monkeypatch):
    from cron.claim_compat import claim_job_for_fire_result
    first = Mock(return_value="paused")
    second = Mock(return_value="disabled")
    monkeypatch.setattr(jobs, "claim_job_for_fire_result", first)
    assert claim_job_for_fire_result("one") == "paused"
    monkeypatch.setattr(jobs, "claim_job_for_fire_result", second)
    assert claim_job_for_fire_result("two") == "disabled"
    first.assert_called_once_with("one")
    second.assert_called_once_with("two")


def test_api_exception_identity_is_preserved(monkeypatch):
    from cron.claim_compat import claim_job_for_fire_result
    error = ValueError("synthetic store failure")
    monkeypatch.setattr(jobs, "claim_job_for_fire_result", Mock(side_effect=error))
    with pytest.raises(ValueError) as caught:
        claim_job_for_fire_result("synthetic-id")
    assert caught.value is error


@pytest.mark.parametrize("value", [None, False, "claimed", 1])
def test_noncallable_requires_restart_without_legacy_claim(monkeypatch, value):
    from cron.claim_compat import claim_job_for_fire_result
    old = Mock(side_effect=AssertionError("legacy claim is forbidden"))
    monkeypatch.setattr(jobs, "claim_job_for_fire", old)
    monkeypatch.setattr(jobs, "claim_job_for_fire_result", value)
    with pytest.raises(RuntimeError, match="(?i)restart"):
        claim_job_for_fire_result("synthetic-id")
    old.assert_not_called()


def test_absent_api_requires_restart_without_legacy_claim(monkeypatch):
    from cron.claim_compat import claim_job_for_fire_result
    old = Mock(side_effect=AssertionError("legacy claim is forbidden"))
    monkeypatch.setattr(jobs, "claim_job_for_fire", old)
    monkeypatch.delattr(jobs, "claim_job_for_fire_result")
    with pytest.raises(RuntimeError, match="(?i)restart"):
        claim_job_for_fire_result("synthetic-id")
    old.assert_not_called()
