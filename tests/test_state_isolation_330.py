"""Regression tests for runtime state isolation (hermes-home#330).

1. An AIAgent built WITHOUT a session DB must never gain a writable handle as a
   side effect of ``session_search`` recall. Before the fix,
   ``_get_session_db_for_recall`` assigned ``SessionDB()`` (read-write, live
   ~/.hermes/state.db) to ``agent._session_db`` and the agent then persisted
   its whole run there (28 sessions / 780 messages leaked from one harness).

2. ``run_agent._hermes_home`` / ``gateway.run._hermes_home`` /
   ``hermes_state.DEFAULT_DB_PATH`` were resolved at IMPORT time, so a process
   that imported them before setting HERMES_HOME logged into (and opened the
   state DB of) the live home. They now resolve lazily.

3. A session whose first billed call ran on a fallback recorded the REQUESTED
   model next to the FALLBACK billing_provider. The first billed call now
   records its model too.
"""

import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gateway.run  # noqa: F401  (import BEFORE HERMES_HOME changes, on purpose)
import hermes_logging
import hermes_state
import run_agent
from hermes_constants import LazyHermesPath
from hermes_state import SessionDB


def _session_search_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "session_search",
                "description": "search past sessions",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]


def _counts(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return sessions, messages
    finally:
        conn.close()


def _seed_state_db(db_path: Path) -> None:
    db = SessionDB(db_path=db_path)
    db.create_session("seed-session", source="cli", model="seed/model")
    db.append_message("seed-session", "user", content="where is the zebracorn ledger?")
    db.append_message("seed-session", "assistant", content="The zebracorn ledger is in vault B.")
    db.close()


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(**kwargs):
    with (
        patch("run_agent.get_tool_definitions", return_value=_session_search_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


# ── Bug 1: recall must not make a session-less agent persistent ─────────────


def test_session_search_without_session_db_is_read_only(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = home / "state.db"
    _seed_state_db(db_path)
    before = _counts(db_path)
    assert before == (1, 2)

    agent = _make_agent(session_db=None)
    assert agent._session_db is None

    call = SimpleNamespace(
        id="call_recall",
        type="function",
        function=SimpleNamespace(
            name="session_search", arguments=json.dumps({"query": "zebracorn"})
        ),
    )
    agent.client.chat.completions.create.side_effect = [
        _mock_response(finish_reason="tool_calls", tool_calls=[call]),
        _mock_response(content="found it", finish_reason="stop"),
    ]
    with patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation("what did we say about the zebracorn ledger?")

    assert result["final_response"] == "found it"

    # Recall still worked: the tool result names the seeded session.
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "session_search never ran"
    assert "seed-session" in tool_msgs[0]["content"]

    # ...through a read-only handle that is NOT the agent's persistence DB.
    assert agent._session_db is None
    assert agent._recall_session_db is not None
    assert agent._recall_session_db.read_only is True

    # And nothing was written to the DB.
    assert _counts(db_path) == before

    agent.close()
    assert agent._recall_session_db is None


def test_recall_handle_rejects_writes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_state_db(home / "state.db")

    agent = _make_agent(session_db=None)
    db = agent._get_session_db_for_recall()
    assert db is not None and db.read_only
    assert db is agent._get_session_db_for_recall()  # cached
    try:
        db._conn.execute("DELETE FROM messages")
    except sqlite3.OperationalError as exc:
        assert "readonly" in str(exc).lower()
    else:  # pragma: no cover - the fix is broken if we get here
        raise AssertionError("recall connection accepted a write")
    agent.close()


def test_recall_without_state_db_creates_nothing(tmp_path, monkeypatch):
    home = tmp_path / "empty-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = _make_agent(session_db=None)
    assert agent._get_session_db_for_recall() is None
    assert not (home / "state.db").exists()
    assert agent._session_db is None


def test_recall_reuses_frontend_session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "frontend.db")
    agent = _make_agent(session_db=db)
    assert agent._get_session_db_for_recall() is db
    assert getattr(agent, "_recall_session_db", None) is None
    db.close()


# ── Bug 2: HERMES_HOME must be resolved lazily ───────────────────────────────


def test_module_home_shims_follow_hermes_home_set_after_import(tmp_path, monkeypatch):
    # The modules were imported at the top of this file, before this change.
    new_home = tmp_path / "late-home"
    new_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(new_home))

    assert isinstance(run_agent._hermes_home, LazyHermesPath)
    assert isinstance(gateway.run._hermes_home, LazyHermesPath)
    assert Path(run_agent._hermes_home) == new_home
    assert gateway.run._hermes_home / "config.yaml" == new_home / "config.yaml"
    assert Path(hermes_state.DEFAULT_DB_PATH) == new_home / "state.db"
    assert SessionDB(read_only=False).db_path == new_home / "state.db"


def test_agent_built_after_home_change_logs_under_new_home(tmp_path, monkeypatch):
    new_home = tmp_path / "late-home"
    new_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(new_home))

    root = logging.getLogger()
    original = list(root.handlers)
    monkeypatch.setattr(hermes_logging, "_logging_initialized", False)
    try:
        _make_agent(session_db=None)
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.FileHandler) and h not in original
        ]
        assert file_handlers, "AIAgent init attached no log file handlers"
        for h in file_handlers:
            assert Path(h.baseFilename).resolve().is_relative_to(new_home.resolve()), (
                h.baseFilename
            )
    finally:
        for h in list(root.handlers):
            if h not in original:
                root.removeHandler(h)
                h.close()


# ── Session row: model paired with the billed provider ──────────────────────


def test_first_billed_call_records_actual_model_with_provider(tmp_path):
    db = SessionDB(db_path=tmp_path / "s.db")
    db.create_session("s1", source="cli", model="gpt-5.6")
    # The first billed call ran on the fallback route.
    db.update_token_counts(
        "s1", input_tokens=10, output_tokens=5,
        model="glm-5.3", billing_provider="zai", api_call_count=1,
    )
    row = db.get_session("s1")
    assert (row["model"], row["billing_provider"]) == ("glm-5.3", "zai")

    # Later calls keep the first billed route (COALESCE semantics unchanged).
    db.update_token_counts(
        "s1", input_tokens=1, model="gpt-5.6", billing_provider="openai", api_call_count=1,
    )
    row = db.get_session("s1")
    assert (row["model"], row["billing_provider"]) == ("glm-5.3", "zai")
    db.close()


def test_token_update_without_provider_keeps_model(tmp_path):
    db = SessionDB(db_path=tmp_path / "s.db")
    db.create_session("s1", source="cli", model="requested")
    db.update_token_counts("s1", input_tokens=1, model="other", api_call_count=1)
    assert db.get_session("s1")["model"] == "requested"
    db.close()
