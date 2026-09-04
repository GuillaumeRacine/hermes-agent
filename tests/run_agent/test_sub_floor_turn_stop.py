"""A turn whose fallback lands on a sub-floor runtime must stop, not answer.

hermes-home #233 P0-1: on 2026-09-01 the chain bottomed out on
``local-ollama/llama3.2:3b`` and the 3B model impersonated the agent. With
the degraded-mode gate the conversation loop ends the turn with reason
``degraded_sub_floor`` and the fixed notice, without asking the floor model
for an answer.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


FLOOR_CONFIG = {
    "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"},
    "fallback_providers": [{"provider": "local-ollama", "model": "llama3.2:3b"}],
    "provider_circuits": {"enabled": False},
    "degraded_mode": {
        "enabled": True,
        "floor": {"providers": ["local-ollama", "ollama"], "models": []},
        "notify_template": "DEGRADED: {detail} (eta {eta})",
    },
}


@pytest.fixture(autouse=True)
def _floor_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: dict(FLOOR_CONFIG))


def _make_agent(fallback_model):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            api_key="fake",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        agent._persist_session = lambda *a, **k: None
        agent._save_trajectory = lambda *a, **k: None
        return agent


class _RateLimitError(Exception):
    status_code = 429

    def __str__(self):
        return "Error code: 429 - Rate limit exceeded."


def _ollama_client():
    client = MagicMock()
    client.base_url = "http://localhost:11434/v1"
    client.api_key = "ollama"
    return client


def _mock_response(content="I am llama"):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None,
                          reasoning_content=None, reasoning_details=None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           model="llama3.2:3b", usage=None)


def test_fallback_to_sub_floor_ends_turn_with_notice():
    agent = _make_agent({"provider": "local-ollama", "model": "llama3.2:3b"})
    calls = []

    def _fake_api_call(api_kwargs):
        calls.append(agent.model)
        if len(calls) == 1:
            raise _RateLimitError()
        # Any further call would be the sub-floor model answering — forbidden.
        return _mock_response()

    agent._interruptible_api_call = _fake_api_call

    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_ollama_client(), "llama3.2:3b")),
        patch("run_agent.time.sleep", return_value=None),
    ):
        result = agent.run_conversation("please run the monthly report")

    # The chain switched to the floor model...
    assert agent._fallback_activated is True
    assert agent.provider == "local-ollama"
    assert agent._degraded_sub_floor is True
    # ...but it was never asked for the answer.
    assert calls == ["anthropic/claude-sonnet-4"]
    assert result["turn_exit_reason"] == "degraded_sub_floor"
    assert result["final_response"].startswith("DEGRADED:")
    assert "local-ollama/llama3.2:3b" in result["final_response"]
    # Transcript stays role-alternating: user → assistant(notice).
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == result["final_response"]


def test_capable_fallback_is_not_gated():
    agent = _make_agent({"provider": "zai", "model": "glm-5.2"})
    calls = []

    def _fake_api_call(api_kwargs):
        calls.append(agent.model)
        if len(calls) == 1:
            raise _RateLimitError()
        return _mock_response("Recovered on glm")

    agent._interruptible_api_call = _fake_api_call
    zai = MagicMock()
    zai.base_url = "https://api.z.ai/api/paas/v4"
    zai.api_key = "zai-key"

    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(zai, "glm-5.2")),
        patch("run_agent.time.sleep", return_value=None),
    ):
        result = agent.run_conversation("hello")

    assert result["final_response"] == "Recovered on glm"
    assert result["turn_exit_reason"].startswith("text_response(")
    assert not getattr(agent, "_degraded_sub_floor", False)
    assert calls == ["anthropic/claude-sonnet-4", "glm-5.2"]


def test_stale_flag_clears_when_primary_restored():
    """A cached agent that returned to its primary must answer normally."""
    agent = _make_agent({"provider": "local-ollama", "model": "llama3.2:3b"})
    agent._degraded_sub_floor = True  # left over from a previous degraded turn
    agent._degraded_notice = "stale"
    agent._fallback_activated = False  # primary already restored

    agent._interruptible_api_call = lambda api_kwargs: _mock_response("normal answer")
    with patch("run_agent.time.sleep", return_value=None):
        result = agent.run_conversation("hello again")

    assert result["final_response"] == "normal answer"
    assert agent._degraded_sub_floor is False
