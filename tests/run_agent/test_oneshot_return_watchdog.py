"""Regression coverage for bounded one-shot post-tool return paths."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _response(*, content="", tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=None,
        model="test/model",
    )


def test_deadline_arms_only_after_tool_results_and_never_during_tools():
    from agent.conversation_loop import _arm_oneshot_return_deadline

    agent = SimpleNamespace(
        _oneshot_return_timeout_seconds=30,
        _oneshot_return_deadline=None,
        _executing_tools=False,
    )

    _arm_oneshot_return_deadline(agent, [{"role": "user", "content": "go"}])
    assert agent._oneshot_return_deadline is None

    before = time.monotonic()
    _arm_oneshot_return_deadline(agent, [{"role": "tool", "content": "done"}])
    assert before + 29 <= agent._oneshot_return_deadline <= before + 31

    agent._executing_tools = True
    _arm_oneshot_return_deadline(agent, [{"role": "tool", "content": "done"}])
    assert agent._oneshot_return_deadline is None


def test_nonstream_codex_return_path_exits_with_resumable_timeout():
    """The exact helper observed in #189 must not multiply retry windows."""
    from agent.chat_completion_helpers import interruptible_api_call
    from agent.errors import OneShotReturnTimeoutError

    release = threading.Event()

    class _Completions:
        def create(self, **_kwargs):
            release.wait(timeout=5)
            return object()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
    )

    class _Agent:
        api_mode = "chat_completions"
        session_id = "completed-side-effect-session"
        _interrupt_requested = False
        _oneshot_return_timeout_seconds = 0.05
        _oneshot_return_deadline = time.monotonic() + 0.05

        @staticmethod
        def _compute_non_stream_stale_timeout(_kwargs):
            return float("inf")

        @staticmethod
        def _create_request_openai_client(**_kwargs):
            return client

        @staticmethod
        def _abort_request_openai_client(_client, *, reason):
            assert reason == "oneshot_return_timeout"
            release.set()

        @staticmethod
        def _close_request_openai_client(_client, *, reason):
            release.set()

        @staticmethod
        def _touch_activity(_desc):
            return None

        @staticmethod
        def _buffer_status(_desc):
            return None

    started = time.monotonic()
    with pytest.raises(OneShotReturnTimeoutError) as exc:
        interruptible_api_call(_Agent(), {"model": "test/model"})

    assert time.monotonic() - started < 1.0
    assert exc.value.session_id == "completed-side-effect-session"
    assert "hermes --resume completed-side-effect-session" in str(exc.value)


def test_completed_tool_side_effect_survives_hung_final_response():
    """End-to-end loop fixture: tool completes, then return path fails closed."""
    from agent.errors import OneShotReturnTimeoutError
    from run_agent import AIAgent

    tool_call = SimpleNamespace(
        id="call-proof",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"q":"proof"}'),
    )
    responses = [
        _response(tool_calls=[tool_call], finish_reason="tool_calls"),
        OneShotReturnTimeoutError(
            session_id="fixture-session",
            timeout_seconds=30,
        ),
    ]
    completed = {"tool": False, "deadline_was_clear": False}

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        enabled_toolsets=["web"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    agent._oneshot_return_timeout_seconds = 30
    agent.valid_tool_names.add("web_search")
    agent._persist_session = lambda *_args, **_kwargs: None
    agent._save_trajectory = lambda *_args, **_kwargs: None
    agent._cleanup_task_resources = lambda *_args, **_kwargs: None

    def _api_call(_kwargs):
        value = responses.pop(0)
        if isinstance(value, BaseException):
            # The post-tool iteration must have armed the watchdog before the
            # provider call that ultimately stalls.
            assert agent._oneshot_return_deadline is not None
            raise value
        return value

    def _complete_tool(*_args, **_kwargs):
        completed["tool"] = True
        completed["deadline_was_clear"] = agent._oneshot_return_deadline is None
        return "durable side effect complete"

    agent._interruptible_api_call = _api_call
    with patch("run_agent.handle_function_call", side_effect=_complete_tool):
        with pytest.raises(OneShotReturnTimeoutError) as exc:
            agent.run_conversation("perform the bounded task")

    assert completed == {"tool": True, "deadline_was_clear": True}
    assert exc.value.session_id == "fixture-session"
    assert responses == []
