"""Integration tests: token budget enforcement inside run_conversation().

hermes-home #233 P1-4.  Builds a real AIAgent with a fake OpenAI client
(pattern from tests/run_agent/test_dict_tool_call_args.py and
test_provider_fallback.py) and feeds it responses carrying ``usage`` so the
loop's usage-accounting block drives ``agent._token_budget``.
"""

import json
from types import SimpleNamespace

import pytest

from agent.token_budget import TOKEN_BUDGET_EXCEEDED, TokenBudget


def _tool_call(call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": "x"})),
    )


def _usage(prompt, completion):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
    )


def _tool_response(prompt=500, completion=100, call_id="call_1"):
    msg = SimpleNamespace(content=None, reasoning=None, tool_calls=[_tool_call(call_id)])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=_usage(prompt, completion),
    )


def _text_response(text="done", prompt=500, completion=100):
    msg = SimpleNamespace(content=text, reasoning=None, tool_calls=[])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=_usage(prompt, completion),
    )


class _FakeChatCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("fake client ran out of responses")
        item = self.responses.pop(0)
        return item() if callable(item) else item


class _FakeClient:
    def __init__(self, responses):
        self.completions = _FakeChatCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_agent(monkeypatch, responses, *, platform="cli", max_iterations=10):
    """Agent whose client returns ``responses`` in order.

    ``responses`` may hold response objects or zero-arg callables; a
    callable is invoked lazily so tests can append more after construction.
    """
    from run_agent import AIAgent

    client = _FakeClient(responses)
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
    )
    agent = AIAgent(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        platform=platform,
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )
    agent._disable_streaming = True
    agent._cleanup_task_resources = lambda *a, **k: None
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    agent._client_calls = client.completions
    return agent


def _set_budget(agent, **kwargs):
    agent._token_budget = TokenBudget(**kwargs)
    agent._token_budget_exceeded = False
    return agent._token_budget


# ── wiring ────────────────────────────────────────────────────────────


def test_agent_gets_budget_from_config_for_its_platform(monkeypatch):
    cfg = {"agent": {"token_budget": {
        "per_session": 600_000,
        "platforms": {"slack": {"per_turn": 150_000}},
    }}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    agent = _make_agent(monkeypatch, [_text_response()], platform="slack")
    assert agent._token_budget.per_session == 600_000
    assert agent._token_budget.per_turn == 150_000
    assert agent._token_budget.action == "stop"
    assert agent._token_budget_exceeded is False


def test_unlimited_default_never_stops(monkeypatch):
    agent = _make_agent(monkeypatch, [_tool_response(), _tool_response(call_id="call_2"), _text_response()])
    assert not agent._token_budget.enabled
    result = agent.run_conversation("go")
    assert result["final_response"] == "done"
    assert result["turn_exit_reason"].startswith("text_response")
    assert agent._token_budget.session_tokens == 1_800
    assert agent.session_total_tokens == 1_800


# ── per_turn / stop ───────────────────────────────────────────────────


def test_per_turn_stop_ends_turn_with_reason_and_message(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_tool_response(500, 100), _tool_response(500, 100, call_id="call_2"), _text_response()],
    )
    tb = _set_budget(agent, per_turn=1_000)

    result = agent.run_conversation("go")

    assert agent._client_calls.calls == 2          # third response never requested
    assert result["turn_exit_reason"] == TOKEN_BUDGET_EXCEEDED
    assert result["final_response"] == (
        "Stopped: this session has used 1,200 tokens of its unlimited budget "
        "(turn: 1,200/1,000). Reply `continue` to allow one more turn, or start a new session."
    )
    assert result["completed"] is True
    assert result["messages"][-1] == {"role": "assistant", "content": result["final_response"]}
    assert result["messages"][-2]["role"] == "tool"  # tool results kept -> valid history
    assert tb.turn_tokens == 1_200 and tb.session_tokens == 1_200
    # per_turn breach is not sticky: the next turn runs normally
    assert agent._token_budget_exceeded is False
    agent._client_calls.responses = [_text_response("again")]
    result2 = agent.run_conversation("next")
    assert result2["final_response"] == "again"
    assert tb.turn_tokens == 600 and tb.session_tokens == 1_800


def test_per_turn_warn_does_not_stop(monkeypatch, caplog):
    agent = _make_agent(
        monkeypatch,
        [_tool_response(500, 100), _tool_response(500, 100, call_id="call_2"), _text_response()],
    )
    tb = _set_budget(agent, per_turn=1_000, action="warn")
    statuses = []
    agent._buffer_status = lambda msg: statuses.append(msg)

    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        result = agent.run_conversation("go")

    assert agent._client_calls.calls == 3
    assert result["final_response"] == "done"
    assert result["turn_exit_reason"].startswith("text_response")
    assert tb.turn_tokens == 1_800
    assert agent._token_budget_exceeded is False
    warn_lines = [r for r in caplog.records if "Token budget (per_turn)" in r.getMessage()]
    assert len(warn_lines) == 1                    # once per breach, not per call
    assert len([s for s in statuses if "Token budget warning" in s]) == 1


# ── per_session across turns + continue override ──────────────────────


def test_per_session_accumulates_across_turns_and_refuses_next_turn(monkeypatch):
    agent = _make_agent(monkeypatch, [_text_response(prompt=500, completion=100)])
    tb = _set_budget(agent, per_session=1_000, per_turn=0)

    r1 = agent.run_conversation("one")            # 600 tokens, under budget
    assert r1["final_response"] == "done"
    assert tb.session_tokens == 600 and not agent._token_budget_exceeded

    agent._client_calls.responses = [_text_response(prompt=500, completion=100)]
    r2 = agent.run_conversation("two")            # 1200 > 1000, final text so turn ends normally
    assert r2["final_response"] == "done"
    assert tb.session_tokens == 1_200

    # Third turn: refused before any API call.
    calls_before = agent._client_calls.calls
    agent._client_calls.responses = [_text_response("should not run")]
    r3 = agent.run_conversation("three")
    assert agent._client_calls.calls == calls_before
    assert r3["turn_exit_reason"] == TOKEN_BUDGET_EXCEEDED
    assert r3["api_calls"] == 0
    assert r3["final_response"].startswith("Stopped: this session has used 1,200 tokens of its 1,000 budget")
    assert agent._token_budget_exceeded is True
    assert r3["messages"][-1]["role"] == "assistant"


def test_continue_override_allows_exactly_one_more_turn(monkeypatch):
    agent = _make_agent(monkeypatch, [_text_response(prompt=1_000, completion=200)])
    tb = _set_budget(agent, per_session=1_000, per_turn=500)

    # Turn 1: the single text response overshoots the session limit.
    r1 = agent.run_conversation("one")
    assert r1["final_response"] == "done"
    assert tb.session_tokens == 1_200

    # Turn 2 (non-continue): refused, exceeded state set.
    agent._client_calls.responses = [_text_response("nope")]
    r2 = agent.run_conversation("two")
    assert r2["turn_exit_reason"] == TOKEN_BUDGET_EXCEEDED
    assert agent._token_budget_exceeded is True

    # Turn 3: `continue` raises the limit by per_turn and runs one turn.
    agent._client_calls.responses = [_text_response("resumed", prompt=300, completion=50)]
    r3 = agent.run_conversation("  Continue ")
    assert r3["final_response"] == "resumed"
    assert tb.per_session == 1_200 + 500          # max(limit, spent) + per_turn
    assert tb.session_tokens == 1_550
    assert tb.extensions_granted == 1
    assert agent._token_budget_exceeded is False

    # Turn 4: 1550 <= 1700, still allowed; spend past the limit again.
    agent._client_calls.responses = [_text_response("more", prompt=300, completion=50)]
    r4 = agent.run_conversation("four")
    assert r4["final_response"] == "more"
    assert tb.session_tokens == 1_900

    # Turn 5: refused again -> each `continue` buys exactly one turn.
    agent._client_calls.responses = [_text_response("nope")]
    r5 = agent.run_conversation("five")
    assert r5["turn_exit_reason"] == TOKEN_BUDGET_EXCEEDED

    # `continue` while NOT exceeded is an ordinary message (no extension).
    agent._client_calls.responses = [_text_response("ok", prompt=1, completion=1)]
    agent._token_budget_exceeded = False
    tb.grant_extension()
    granted_before = tb.extensions_granted
    agent.run_conversation("continue")
    assert tb.extensions_granted == granted_before


def test_per_session_stop_mid_turn_sets_exceeded_state(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_tool_response(500, 100), _tool_response(500, 100, call_id="call_2"), _text_response()],
    )
    tb = _set_budget(agent, per_session=1_000)
    result = agent.run_conversation("go")
    assert agent._client_calls.calls == 2
    assert result["turn_exit_reason"] == TOKEN_BUDGET_EXCEEDED
    assert agent._token_budget_exceeded is True
    assert tb.exceeded is True
    assert "1,200 tokens of its 1,000 budget" in result["final_response"]


def test_reset_session_state_clears_budget_counters(monkeypatch):
    agent = _make_agent(monkeypatch, [_text_response(prompt=900, completion=200)])
    tb = _set_budget(agent, per_session=1_000)
    agent.run_conversation("one")
    assert tb.session_tokens == 1_100
    agent._token_budget_exceeded = True
    agent.reset_session_state()
    assert tb.session_tokens == 0 and tb.turn_tokens == 0
    assert tb.per_session == 1_000                # limits survive
    assert agent._token_budget_exceeded is False


# ── context_soft_limit ────────────────────────────────────────────────


def test_context_soft_limit_forces_compression(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_tool_response(prompt=5_000, completion=100), _text_response(prompt=200, completion=10)],
    )
    _set_budget(agent, context_soft_limit=4_000)
    agent.compression_enabled = True
    # Threshold check alone would not fire (5k prompt vs a large model window).
    assert not agent.context_compressor.should_compress(5_000)

    compress_calls = []

    def _fake_compress(messages, system_message, **kwargs):
        compress_calls.append(kwargs)
        return messages[:-1], system_message     # "made progress"

    agent._compress_context = _fake_compress
    result = agent.run_conversation("go")

    assert result["final_response"] == "done"
    assert len(compress_calls) == 1
    assert compress_calls[0]["approx_tokens"] == 5_000
    assert agent._token_budget.compression_requested() is False


def test_context_soft_limit_below_threshold_does_not_compress(monkeypatch):
    agent = _make_agent(
        monkeypatch,
        [_tool_response(prompt=3_000, completion=100), _text_response()],
    )
    _set_budget(agent, context_soft_limit=4_000)
    agent.compression_enabled = True
    compress_calls = []
    agent._compress_context = lambda m, s, **kw: (compress_calls.append(kw) or (m, s))
    agent.run_conversation("go")
    assert compress_calls == []


def test_context_soft_limit_warns_once_when_compression_unavailable(monkeypatch, caplog):
    agent = _make_agent(
        monkeypatch,
        [
            _tool_response(prompt=5_000, completion=100),
            _tool_response(prompt=5_500, completion=100, call_id="call_2"),
            _text_response(),
        ],
    )
    _set_budget(agent, context_soft_limit=4_000)
    agent.compression_enabled = False
    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        result = agent.run_conversation("go")
    assert result["final_response"] == "done"
    msgs = [r.getMessage() for r in caplog.records
            if "context soft limit exceeded and compression unavailable" in r.getMessage()]
    assert len(msgs) == 1


def test_turn_ended_log_line_reports_tokens(monkeypatch, caplog):
    agent = _make_agent(monkeypatch, [_text_response(prompt=500, completion=100)])
    _set_budget(agent, per_session=10_000)
    with caplog.at_level("INFO", logger="agent.conversation_loop"):
        agent.run_conversation("go")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Turn ended:")]
    assert lines and "tokens=600/600/10000" in lines[-1]


@pytest.mark.parametrize("reason,needle", [
    ("token_budget_exceeded", "token budget"),
    ("text_response(finish_reason=stop)", ""),
])
def test_completion_explainer_knows_token_budget(reason, needle):
    from run_agent import AIAgent

    text = AIAgent._format_turn_completion_explanation(reason)
    if needle:
        assert needle in text
    else:
        assert text == ""
