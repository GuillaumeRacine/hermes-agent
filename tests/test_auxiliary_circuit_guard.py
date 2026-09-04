"""Auxiliary fallback chains (vision, compression, MoA aggregator …) honour provider circuits."""

import time

import pytest

from agent import auxiliary_client as aux
from hermes_cli.provider_circuits import circuit_status, record_failure


class _RateLimitError(Exception):
    def __init__(self, resets_at=None, message="429 rate limit"):
        super().__init__(message)
        self.status_code = 429
        self.body = {"error": {"type": "usage_limit_reached", "resets_at": resets_at}} if resets_at else None
        self.response = None


@pytest.fixture
def circuits(tmp_path, monkeypatch):
    path = tmp_path / "circuits.json"
    cfg = {"provider_circuits": {"enabled": True, "state_path": str(path)}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    return path


def test_main_fallback_chain_skips_open_circuit(circuits, monkeypatch):
    record_failure("zai", "glm-5.2", "rate_limit", path=circuits, retry_after_seconds=90_000)
    monkeypatch.setattr(
        "hermes_cli.fallback_config.get_fallback_chain",
        lambda cfg: [{"provider": "zai", "model": "glm-5.2"}],
    )
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "openai-codex")
    monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda p: False)

    def _boom(entry):
        raise AssertionError(f"must not resolve open-circuit entry {entry}")

    monkeypatch.setattr(aux, "_resolve_fallback_entry", _boom)
    client, model, label = aux._try_main_fallback_chain("vision", "openai-codex")
    assert (client, model, label) == (None, None, "")


def test_configured_fallback_chain_skips_open_circuit(circuits, monkeypatch):
    record_failure("openrouter", "anthropic/claude-opus-4.8", "billing", path=circuits)
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "openrouter", "model": "anthropic/claude-opus-4.8"}]},
    )

    def _boom(entry):
        raise AssertionError("must not resolve open-circuit entry")

    monkeypatch.setattr(aux, "_resolve_fallback_entry", _boom)
    assert aux._try_configured_fallback_chain("vision", "openai-codex") == (None, None, "")


def test_circuit_open_note_fails_open_on_bad_state(circuits, monkeypatch):
    circuits.write_text("{not json")
    assert aux._circuit_open_note("zai", "glm-5.2") is None


def test_record_circuit_failure_uses_reset(circuits):
    reset_at = time.time() + 250_000
    aux._record_circuit_failure("openai-codex", "gpt-5.5", _RateLimitError(reset_at))
    assert circuit_status("openai-codex", "gpt-5.5", path=circuits, now=time.time() + 100_000)["status"] == "open"
    assert circuit_status("openai-codex", "gpt-5.5", path=circuits, now=reset_at + 5)["status"] != "open"


def test_record_circuit_failure_ignores_non_capacity_errors(circuits):
    aux._record_circuit_failure("openai-codex", "gpt-5.5", RuntimeError("timeout"))
    assert circuit_status("openai-codex", "gpt-5.5", path=circuits)["status"] != "open"
    aux._record_circuit_failure("auto", "x", _RateLimitError())
    assert circuit_status("auto", "x", path=circuits)["status"] != "open"
