"""MoA reference slots must honour the persistent provider circuit.

Before: a reference whose provider was out of quota was re-dialled on every
MoA turn (76 identical Codex failures on 2026-09-03). Now an open circuit
skips the slot with a labelled note, and a quota failure from a reference
opens the circuit with the provider's reset time.
"""

import time

import pytest

from agent import moa_loop
from hermes_cli.provider_circuits import circuit_status, record_failure


class _RateLimitError(Exception):
    def __init__(self, resets_at):
        super().__init__("429 rate limit")
        self.status_code = 429
        self.body = {"error": {"type": "usage_limit_reached", "resets_at": resets_at}}
        self.response = None


@pytest.fixture
def circuits(tmp_path, monkeypatch):
    path = tmp_path / "circuits.json"
    cfg = {"provider_circuits": {"enabled": True, "state_path": str(path)}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    return path


def test_open_circuit_skips_reference_without_calling(circuits, monkeypatch):
    record_failure("openai-codex", "gpt-5.5", "rate_limit", path=circuits, retry_after_seconds=90_000)

    def _boom(**kwargs):
        raise AssertionError("call_llm must not be dialled for an open circuit")

    monkeypatch.setattr(moa_loop, "call_llm", _boom)
    out = moa_loop._run_references_parallel(
        [{"provider": "openai-codex", "model": "gpt-5.5"}],
        [{"role": "user", "content": "hi"}],
    )
    assert len(out) == 1
    label, text = out[0]
    assert label == "openai-codex:gpt-5.5"
    assert text.startswith("[skipped: provider circuit open")


def test_closed_circuit_calls_reference(circuits, monkeypatch):
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kwargs: "unused")
    monkeypatch.setattr(moa_loop, "_extract_text", lambda response: "ref answer")
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": slot["provider"], "model": slot["model"]})
    out = moa_loop._run_references_parallel(
        [{"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}],
        [{"role": "user", "content": "hi"}],
    )
    assert out == [("openrouter:deepseek/deepseek-v4-pro", "ref answer")]


def test_reference_quota_failure_opens_circuit_until_reset(circuits, monkeypatch):
    reset_at = time.time() + 200_000

    def _fail(**kwargs):
        raise _RateLimitError(reset_at)

    monkeypatch.setattr(moa_loop, "call_llm", _fail)
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": slot["provider"], "model": slot["model"]})
    label, text = moa_loop._run_reference(
        {"provider": "openai-codex", "model": "gpt-5.5"}, [{"role": "user", "content": "hi"}]
    )
    assert text.startswith("[failed:")
    assert circuit_status("openai-codex", "gpt-5.5", path=circuits, now=time.time() + 100_000)["status"] == "open"
    assert circuit_status("openai-codex", "gpt-5.5", path=circuits, now=reset_at + 5)["status"] != "open"


def test_transient_reference_failure_does_not_open_circuit(circuits, monkeypatch):
    def _fail(**kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(moa_loop, "call_llm", _fail)
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": slot["provider"], "model": slot["model"]})
    moa_loop._run_reference({"provider": "openrouter", "model": "x"}, [{"role": "user", "content": "hi"}])
    assert circuit_status("openrouter", "x", path=circuits)["status"] != "open"
