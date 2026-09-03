"""The provider's own reset time (429 body / headers) must drive the circuit cooldown.

Incident 2026-09-01/03 (hermes-home #233): Codex returned ``usage_limit_reached``
with ``resets_at`` ~4 days out; the circuit reopened after the flat 3600 s
``cooldown_seconds.rate_limit`` and every session re-dialled an exhausted
provider. ``record_provider_failure`` now forwards the parsed reset into
``record_failure(retry_after_seconds=...)``.
"""

import time
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.chat_completion_helpers import (
    record_provider_failure,
    reset_seconds_from_error,
    try_activate_fallback,
)
from hermes_cli.provider_circuits import circuit_status


class _QuotaError(Exception):
    """Mimics openai.RateLimitError: .status_code + .body with the Codex payload."""

    def __init__(self, resets_at: float):
        super().__init__("Error code: 429 - usage_limit_reached")
        self.status_code = 429
        self.body = {
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "resets_at": resets_at,
            }
        }
        self.response = None


@pytest.fixture
def circuits_config(tmp_path, monkeypatch):
    cfg = {"provider_circuits": {"enabled": True, "state_path": str(tmp_path / "circuits.json")}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    return cfg


def _agent(provider="openai-codex", model="gpt-5.5"):
    return SimpleNamespace(
        provider=provider,
        model=model,
        _circuit_provider=None,
        _rate_limit_state=None,
        _fallback_activated=False,
        _primary_runtime={"provider": provider},
        _fallback_chain=[],
        _fallback_index=0,
        _rate_limited_until=0.0,
    )


def test_reset_seconds_from_error_reads_codex_body():
    err = _QuotaError(resets_at=time.time() + 300_000)
    seconds = reset_seconds_from_error(err)
    assert seconds is not None and 299_000 < seconds <= 300_000


def test_reset_seconds_from_error_none_without_hint():
    assert reset_seconds_from_error(Exception("plain failure")) is None
    assert reset_seconds_from_error(None) is None


def test_record_provider_failure_opens_until_reset(circuits_config):
    reset_at = time.time() + 300_000
    agent = _agent()
    seconds = record_provider_failure(agent, FailoverReason.rate_limit, error=_QuotaError(reset_at))
    assert seconds and seconds > 3600
    path = circuits_config["provider_circuits"]["state_path"]
    # Still open long after the flat one-hour cooldown would have expired.
    status = circuit_status("openai-codex", "gpt-5.5", path=path, now=time.time() + 100_000)
    assert status["status"] == "open"
    # Closed once the provider's reset has passed.
    status = circuit_status("openai-codex", "gpt-5.5", path=path, now=reset_at + 5)
    assert status["status"] != "open"


def test_record_provider_failure_without_reset_uses_flat_cooldown(circuits_config):
    agent = _agent()
    assert record_provider_failure(agent, FailoverReason.rate_limit, error=Exception("429")) is None
    path = circuits_config["provider_circuits"]["state_path"]
    assert circuit_status("openai-codex", "gpt-5.5", path=path, now=time.time() + 3_500)["status"] == "open"
    assert circuit_status("openai-codex", "gpt-5.5", path=path, now=time.time() + 3_700)["status"] != "open"


def test_try_activate_fallback_holds_primary_until_reset(circuits_config):
    agent = _agent()
    before = time.monotonic()
    assert try_activate_fallback(agent, FailoverReason.rate_limit, error=_QuotaError(time.time() + 50_000)) is False
    assert agent._rate_limited_until - before > 49_000


def test_try_activate_fallback_default_probe_interval_without_reset(circuits_config):
    agent = _agent()
    before = time.monotonic()
    assert try_activate_fallback(agent, FailoverReason.rate_limit, error=Exception("429")) is False
    assert 55 <= agent._rate_limited_until - before <= 65
