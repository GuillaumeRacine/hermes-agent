"""Regression coverage for bounded one-shot post-tool return paths."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


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
