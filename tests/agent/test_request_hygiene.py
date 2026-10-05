"""hermes-home#330: never send tool_choice / parallel_tool_calls without tools."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.request_hygiene import drop_orphan_tool_fields


def test_drops_tool_choice_when_tools_missing():
    kw = {"model": "grok-4", "tool_choice": "auto", "parallel_tool_calls": True}
    assert drop_orphan_tool_fields(kw) == {"model": "grok-4"}


def test_drops_tool_choice_and_empty_tools_list():
    kw = {"model": "m", "tools": [], "tool_choice": "auto"}
    assert drop_orphan_tool_fields(kw) == {"model": "m"}


def test_keeps_fields_when_tools_present():
    kw = {"model": "m", "tools": [{"type": "function", "name": "x"}],
          "tool_choice": "auto", "parallel_tool_calls": True}
    assert drop_orphan_tool_fields(dict(kw)) == kw


def test_non_dict_passthrough():
    assert drop_orphan_tool_fields(None) is None


def test_run_codex_stream_strips_orphan_tool_choice():
    """The Responses send point is the last line of defence for every caller."""
    from agent.codex_runtime import run_codex_stream

    final = SimpleNamespace(output=[], status="completed")
    client = MagicMock()
    client.responses.create.return_value = final  # concrete response, no __iter__
    agent = SimpleNamespace(
        _ensure_primary_openai_client=lambda reason=None: client,
        _interrupt_requested=False,
        _fire_stream_delta=lambda t: None,
        _fire_reasoning_delta=lambda t: None,
        _touch_activity=lambda m: None,
    )
    run_codex_stream(agent, {"model": "grok-4", "input": [], "tool_choice": "auto",
                             "parallel_tool_calls": True})
    sent = client.responses.create.call_args.kwargs
    assert "tool_choice" not in sent
    assert "parallel_tool_calls" not in sent
    assert sent["stream"] is True


def test_codex_transport_override_cannot_inject_orphan_tool_choice():
    from agent.transports.codex import ResponsesApiTransport

    t = ResponsesApiTransport()
    kw = t.build_kwargs(
        model="grok-4",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        tools=None,
        request_overrides={"tool_choice": "auto"},
    )
    assert "tool_choice" not in kw
    assert "tools" not in kw
