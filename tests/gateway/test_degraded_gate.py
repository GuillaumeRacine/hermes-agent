"""Gateway degraded-mode gate, dropped-request ledger, and replay.

hermes-home #233 P0-1 / P3-11. The narrowest gateway seam is
``GatewayRunner._run_agent`` — the wrapper every inbound platform message goes
through before an ``AIAgent`` is constructed. With no capable runtime the
gate must: not construct/run the agent, reply with the notice once per chat
per hour (then a one-line "still queued"), and append to the ledger.
"""

import importlib
import sys
import types
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent import pending_intents as pi
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.SLACK):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.handled = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SendResult(success=True, message_id="m-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def handle_message(self, event: MessageEvent) -> None:
        self.handled.append(event)


class ExplodingAgent:
    """Constructing this means the gate failed — the agent must never be built."""

    def __init__(self, **kwargs):
        raise AssertionError("AIAgent must not be constructed while degraded")


class NormalAgent:
    def __init__(self, **kwargs):
        self.tools = []
        self._interrupt_requested = False

    @property
    def is_interrupted(self):
        return False

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        return {"final_response": "done", "messages": [], "api_calls": 1,
                "turn_exit_reason": "text_response(finish_reason=stop)"}


class SubFloorAgent(NormalAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        return {"final_response": "DEGRADED notice", "messages": [], "api_calls": 0,
                "completed": False, "turn_exit_reason": "degraded_sub_floor"}


class InterruptedAgent(NormalAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        return {"final_response": "", "messages": [], "api_calls": 1,
                "completed": False, "interrupted": True,
                "turn_exit_reason": "interrupted_during_api_call"}


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    return runner


def _config(tmp_path):
    return {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "fallback_providers": [{"provider": "local-ollama", "model": "llama3.2:3b"}],
        "provider_circuits": {"enabled": True, "state_path": str(tmp_path / "circuits.json")},
        "degraded_mode": {
            "enabled": True,
            "floor": {"providers": ["local-ollama", "ollama"], "models": []},
            "queue_path": str(tmp_path / "pending.jsonl"),
            "replay_max_age_hours": 24,
            "replay_interval_seconds": 300,
            "notify_template": "NOTICE {detail} eta {eta}",
        },
    }


@pytest.fixture
def harness(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    cfg = _config(tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    adapter = CaptureAdapter()
    runner = _make_runner(adapter)
    return SimpleNamespace(runner=runner, adapter=adapter, cfg=cfg, queue=tmp_path / "pending.jsonl")


def _install_agent(monkeypatch, agent_cls):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _source(chat_id="C123", thread_id=None):
    return SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="channel",
                         user_id="U1", user_name="gui", thread_id=thread_id)


async def _run(runner, text, source=None, session_id="sess-1"):
    return await runner._run_agent(
        message=text, context_prompt="", history=[], source=source or _source(),
        session_id=session_id, session_key="agent:main:slack:channel:C123",
        persist_user_message=text,
    )


@pytest.mark.asyncio
async def test_gate_queues_and_notifies_once_per_hour(monkeypatch, harness):
    _install_agent(monkeypatch, ExplodingAgent)
    monkeypatch.setattr(
        "agent.degraded_mode.capable_runtime_available",
        lambda cfg, now=None: (False, "open circuits: openai-codex/gpt-5.6-sol until 2026-09-06T00:00Z (rate_limit)"),
    )

    first = await _run(harness.runner, "ok go")
    assert first["degraded_mode"] is True
    assert first["final_response"].startswith("NOTICE open circuits: openai-codex/gpt-5.6-sol")
    assert first["api_calls"] == 0

    second = await _run(harness.runner, "yes")
    assert second["final_response"] == "Still queued (2 pending) — will replay when a capable model is back."

    pending = pi.list_pending(24, path=harness.queue)
    assert [r["text"] for r in pending] == ["ok go", "yes"]
    assert {r["reason"] for r in pending} == {"fallbacks_exhausted"}
    assert pending[0]["platform"] == "slack" and pending[0]["chat_id"] == "C123"
    assert pending[0]["metadata"]["notified"] is True
    assert pending[0]["metadata"]["last_notified_at"]
    assert "notified" not in pending[1]["metadata"]

    # A different chat gets its own full notice.
    other = await _run(harness.runner, "hi", source=_source("C999"))
    assert other["final_response"].startswith("NOTICE")

    # After the hour window the full notice is repeated.
    harness.runner._degraded_notified_at["slack:C123"] -= 3601
    again = await _run(harness.runner, "still there?")
    assert again["final_response"].startswith("NOTICE")


@pytest.mark.asyncio
async def test_notice_dedupe_survives_restart_via_ledger(monkeypatch, harness):
    _install_agent(monkeypatch, ExplodingAgent)
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (False, "x"))
    await _run(harness.runner, "first")
    # Simulate a fresh gateway process: no in-memory dedupe state.
    fresh = _make_runner(harness.adapter)
    result = await _run(fresh, "second")
    assert result["final_response"].startswith("Still queued (2 pending)")


@pytest.mark.asyncio
async def test_normal_path_when_capable(monkeypatch, harness):
    _install_agent(monkeypatch, NormalAgent)
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (True, "openai-codex/gpt-5.6-sol"))
    result = await _run(harness.runner, "hello")
    assert result["final_response"] == "done"
    assert result.get("turn_exit_reason") == "text_response(finish_reason=stop)"
    assert not harness.queue.exists() or pi.list_pending(24, path=harness.queue) == []


@pytest.mark.asyncio
async def test_gate_fails_open_on_error(monkeypatch, harness):
    _install_agent(monkeypatch, NormalAgent)

    def _boom(cfg, now=None):
        raise RuntimeError("circuit state unreadable")

    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", _boom)
    result = await _run(harness.runner, "hello")
    assert result["final_response"] == "done"


@pytest.mark.asyncio
async def test_gate_disabled_by_config(monkeypatch, harness):
    _install_agent(monkeypatch, NormalAgent)
    harness.cfg["degraded_mode"]["enabled"] = False
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (False, "x"))
    result = await _run(harness.runner, "hello")
    assert result["final_response"] == "done"


@pytest.mark.asyncio
async def test_sub_floor_turn_is_recorded(monkeypatch, harness):
    _install_agent(monkeypatch, SubFloorAgent)
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (True, "zai/glm-5.2"))
    result = await _run(harness.runner, "run the report")
    assert result["final_response"] == "DEGRADED notice"
    pending = pi.list_pending(24, path=harness.queue)
    assert len(pending) == 1
    assert pending[0]["reason"] == "sub_floor_model"
    assert pending[0]["text"] == "run the report"
    assert pending[0]["metadata"]["turn_exit_reason"] == "degraded_sub_floor"


@pytest.mark.asyncio
async def test_shutdown_interrupt_is_recorded_but_user_stop_is_not(monkeypatch, harness):
    _install_agent(monkeypatch, InterruptedAgent)
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (True, "zai/glm-5.2"))

    # User-initiated stop/steer (gateway not draining): nothing recorded.
    await _run(harness.runner, "user stopped this")
    assert not harness.queue.exists() or pi.list_pending(24, path=harness.queue) == []

    # Gateway shutdown/restart drain: recorded with user_initiated=False.
    harness.runner._draining = True
    harness.runner._restart_requested = True
    await _run(harness.runner, "survive the restart")
    pending = pi.list_pending(24, path=harness.queue)
    assert len(pending) == 1
    assert pending[0]["reason"] == "interrupted"
    assert pending[0]["text"] == "survive the restart"
    assert pending[0]["metadata"]["user_initiated"] is False
    assert pending[0]["metadata"]["restart_requested"] is True


@pytest.mark.asyncio
async def test_replay_reinjects_oldest_first_and_expires_stale(monkeypatch, harness):
    queue = harness.queue
    stale = pi.new_record(platform="slack", chat_id="C123", user_id="U1", thread_id=None,
                          text="too old", reason="fallbacks_exhausted")
    stale["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
    stale_id = pi.enqueue(stale, path=queue)
    a = pi.enqueue(pi.new_record(platform="slack", chat_id="C123", user_id="U1", thread_id="171.5",
                                 text="ok go", reason="fallbacks_exhausted",
                                 metadata={"source": {"chat_type": "channel", "chat_name": "rentals"}}), path=queue)
    b = pi.enqueue(pi.new_record(platform="slack", chat_id="C123", user_id="U1", thread_id="171.5",
                                 text="yes", reason="sub_floor_model"), path=queue)
    # A platform this gateway does not serve stays pending.
    c = pi.enqueue(pi.new_record(platform="telegram", chat_id="42", user_id="7", thread_id=None,
                                 text="later", reason="interrupted"), path=queue)

    # Still degraded: nothing happens.
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (False, "x"))
    assert await harness.runner._replay_pending_intents() == 0
    assert harness.adapter.handled == []

    # Capable again: replay oldest first, serial, one preface per chat/thread.
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (True, "zai/glm-5.2"))
    replayed = await harness.runner._replay_pending_intents()
    assert replayed == 2
    texts = [e.text for e in harness.adapter.handled]
    assert texts == ["ok go", "yes"]
    ev = harness.adapter.handled[0]
    assert ev._hermes_replayed_from == a
    assert ev.source.platform == Platform.SLACK
    assert ev.source.chat_id == "C123" and ev.source.thread_id == "171.5"
    assert ev.source.user_id == "U1" and ev.source.chat_type == "channel" and ev.source.chat_name == "rentals"
    prefaces = [s["content"] for s in harness.adapter.sent]
    assert len(prefaces) == 1
    assert prefaces[0].startswith("Replaying your message from ") and "zai/glm-5.2 is back" in prefaces[0]

    assert pi.get(a, path=queue)["status"] == "replayed"
    assert pi.get(a, path=queue)["metadata"]["replayed_by"] == "zai/glm-5.2"
    assert pi.get(b, path=queue)["status"] == "replayed"
    assert pi.get(stale_id, path=queue)["status"] == "expired"
    assert pi.get(c, path=queue)["status"] == "pending"
    # Nothing was posted for the expired record.
    assert all("too old" not in s["content"] for s in harness.adapter.sent)

    # Second pass is a no-op (only the foreign-platform record remains).
    assert await harness.runner._replay_pending_intents() == 0


@pytest.mark.asyncio
async def test_pending_command_lists_queue(monkeypatch, harness):
    queue = harness.queue
    pi.enqueue(pi.new_record(platform="slack", chat_id="C123", user_id="U1", thread_id=None,
                             text="ok go", reason="fallbacks_exhausted"), path=queue)
    pi.enqueue(pi.new_record(platform="slack", chat_id="C999", user_id="U1", thread_id=None,
                             text="elsewhere", reason="interrupted"), path=queue)
    monkeypatch.setattr("agent.degraded_mode.capable_runtime_available", lambda cfg, now=None: (True, "zai/glm-5.2"))

    event = MessageEvent(text="/pending", source=_source())
    out = await harness.runner._handle_pending_command(event)
    assert out.startswith("1 pending intent(s):")
    assert "ok go" in out and "elsewhere" not in out
    assert "Capable runtime: zai/glm-5.2" in out

    event = MessageEvent(text="/pending all", source=_source())
    out = await harness.runner._handle_pending_command(event)
    assert out.startswith("2 pending intent(s) (all chats):")
    assert "slack:C999" in out
