"""Tests for the daily message challenger export + image bulk-reply helper.

Covers:
  * The pure flag logic (stale / over_analyzed / duplicate / unread / image
    fetch failure) on plain unit dicts — no DB required.
  * The dependency-free near-duplicate clustering.
  * The image-fetch-failure detector (shared by both tools).
  * End-to-end ``collect_units`` / ``build_items`` against temp SQLite DBs
    populated with the real state.db + kanban.db table shapes.
  * Graceful degradation when the stores are missing/empty (fresh container /
    ledger-absent case).
  * The ``daily-message-challenger`` blueprint fills into a valid cron spec.
  * The image bulk-reply helper detects affected Slack threads, is idempotent,
    and never sends in dry-run.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from cron.scripts import daily_challenger_export as dce
from cron.scripts import image_bulk_reply as ibr


NOW = 1_700_000_000.0
HOUR = 3600.0


# ---------------------------------------------------------------------------
# Pure text / duplicate helpers
# ---------------------------------------------------------------------------

class TestTextHelpers:
    def test_normalize_strips_punctuation(self):
        assert dce.normalize_text("Hello, WORLD! 123") == ["hello", "world", "123"]

    def test_shingles_short_text_falls_back_to_tokens(self):
        assert dce.shingles("fix bug", k=3) == {"fix", "bug"}

    def test_jaccard_identical_is_one(self):
        s = dce.shingles("deploy the new billing service today")
        assert dce.jaccard(s, s) == 1.0

    def test_jaccard_disjoint_is_zero(self):
        a = dce.shingles("completely unrelated alpha topic here")
        b = dce.shingles("totally different beta subject entirely")
        assert dce.jaccard(a, b) == 0.0


class TestDuplicateClusters:
    def test_near_duplicates_cluster_together(self):
        units = [
            {"id": "a", "title": "Deploy billing service", "summary": "roll out the new billing service to prod"},
            {"id": "b", "title": "Deploy billing service", "summary": "roll out the new billing service to production"},
            {"id": "c", "title": "Unrelated onboarding docs", "summary": "write onboarding docs for interns"},
        ]
        clusters = dce.assign_duplicate_clusters(units, threshold=0.5)
        assert clusters["a"] is not None
        assert clusters["a"] == clusters["b"]
        assert clusters["c"] is None  # singleton


# ---------------------------------------------------------------------------
# Image-fetch-failure detector (shared)
# ---------------------------------------------------------------------------

class TestImageFetchFailureDetector:
    @pytest.mark.parametrize(
        "text",
        [
            "Slack attachment access failed for photo.png. Missing scope: files:read.",
            "Slack attachment photo.png returned HTTP 404 and is no longer reachable.",
            "download failed with HTTP 403",
            "error: missing_scope",
            "the files.slack.com link has expired",
            "Slack returned an HTML/login or non-media response",
            "ratelimited",
            "token_revoked",
        ],
    )
    def test_positive(self, text):
        assert dce.detect_image_fetch_failure(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "here is a normal message with no problems",
            "the image shows a cat sitting on a mat",
            "uploaded successfully",
        ],
    )
    def test_negative(self, text):
        assert dce.detect_image_fetch_failure(text) is False


# ---------------------------------------------------------------------------
# Pure flag logic
# ---------------------------------------------------------------------------

def _unit(**kw):
    base = {
        "id": "u1", "kind": "task", "title": "t", "summary": "",
        "status": "ready", "open": True, "last_activity_ts": NOW,
        "analysis_turns": 0, "has_terminal_disposition": False,
        "has_attachment": False, "attachment_processed": False, "texts": [],
    }
    base.update(kw)
    return base


class TestFlags:
    def test_stale_fires_for_idle_open_item(self):
        u = _unit(open=True, last_activity_ts=NOW - 20 * HOUR)
        assert dce.flag_stale(u, NOW, stale_hours=12) is True

    def test_stale_not_for_recent(self):
        u = _unit(open=True, last_activity_ts=NOW - 2 * HOUR)
        assert dce.flag_stale(u, NOW, stale_hours=12) is False

    def test_stale_not_for_closed(self):
        u = _unit(open=False, last_activity_ts=NOW - 100 * HOUR)
        assert dce.flag_stale(u, NOW, stale_hours=12) is False

    def test_over_analyzed_fires_past_bar(self):
        u = _unit(analysis_turns=5, has_terminal_disposition=False)
        assert dce.flag_over_analyzed(u, packet_bar=2) is True

    def test_over_analyzed_suppressed_by_disposition(self):
        u = _unit(analysis_turns=5, has_terminal_disposition=True)
        assert dce.flag_over_analyzed(u, packet_bar=2) is False

    def test_over_analyzed_below_bar(self):
        u = _unit(analysis_turns=2, has_terminal_disposition=False)
        assert dce.flag_over_analyzed(u, packet_bar=2) is False

    def test_unread_attachment(self):
        assert dce.flag_unread_attachment(_unit(has_attachment=True, attachment_processed=False)) is True
        assert dce.flag_unread_attachment(_unit(has_attachment=True, attachment_processed=True)) is False
        assert dce.flag_unread_attachment(_unit(has_attachment=False)) is False

    def test_image_fetch_failure_flag(self):
        u = _unit(texts=["all good", "download failed with HTTP 401"])
        assert dce.flag_image_fetch_failure(u) is True
        assert dce.flag_image_fetch_failure(_unit(texts=["fine", "great"])) is False


class TestBuildItems:
    def test_only_flagged_by_default(self):
        units = [
            _unit(id="clean", title="fresh onboarding docs", analysis_turns=1),  # no flags
            _unit(id="stale", title="deploy billing service",
                  open=True, last_activity_ts=NOW - 48 * HOUR),
        ]
        items = dce.build_items(units, NOW, stale_hours=12)
        ids = {i["id"] for i in items}
        assert ids == {"stale"}
        assert "stale" in items[0]["challenge_flags"]

    def test_all_flag_emits_everything(self):
        units = [_unit(id="clean", analysis_turns=0)]
        items = dce.build_items(units, NOW, only_flagged=False)
        assert len(items) == 1

    def test_items_have_classify_compatible_fields(self):
        units = [_unit(id="x", title="Title", summary="Sum", analysis_turns=9)]
        items = dce.build_items(units, NOW, packet_bar=2)
        it = items[0]
        assert it["id"] == "x" and it["title"] == "Title"
        assert "text" in it and it["summary"] == "Sum"
        assert it["flags"]["over_analyzed"] is True


# ---------------------------------------------------------------------------
# End-to-end against temp SQLite (real table shapes)
# ---------------------------------------------------------------------------

def _make_state_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, user_id TEXT, model TEXT,
            started_at REAL NOT NULL, ended_at REAL, title TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT, tool_calls TEXT, tool_name TEXT,
            timestamp REAL NOT NULL, platform_message_id TEXT
        );
        """
    )
    return conn


def _make_kanban_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
            status TEXT NOT NULL, priority INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            author TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            run_id INTEGER, kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
        CREATE TABLE task_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            filename TEXT NOT NULL, stored_path TEXT NOT NULL, content_type TEXT,
            size INTEGER NOT NULL DEFAULT 0, uploaded_by TEXT, created_at INTEGER NOT NULL
        );
        """
    )
    return conn


class TestStoreReaders:
    def test_missing_stores_degrade_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_STATE_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        # No DBs on disk -> no units, no crash.
        assert dce.collect_units(24, NOW) == []

    def test_state_db_over_analyzed_thread(self, tmp_path, monkeypatch):
        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, ended_at, title) VALUES (?,?,?,?,?)",
            ("s1", "slack", NOW - 5 * HOUR, None, "Billing thread"),
        )
        # 4 assistant/tool turns, still open -> over_analyzed at bar=2.
        for i in range(4):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
                ("s1", "assistant", f"analysis turn {i}", NOW - (4 - i) * HOUR),
            )
        conn.commit()
        conn.close()
        monkeypatch.setenv("HERMES_STATE_DB", str(db))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "nope.db"))

        items = dce.build_items(dce.collect_units(24, NOW), NOW, packet_bar=2)
        assert len(items) == 1
        assert items[0]["id"] == "thread:s1"
        assert items[0]["flags"]["over_analyzed"] is True

    def test_kanban_flags(self, tmp_path, monkeypatch):
        db = tmp_path / "kanban.db"
        conn = _make_kanban_db(db)
        # Stale open task, no recent change but an in-window comment keeps it in scope.
        conn.execute(
            "INSERT INTO tasks (id, title, body, status, created_at, started_at) VALUES (?,?,?,?,?,?)",
            ("t1", "Review screenshot", "see attached screenshot.png", "ready",
             int(NOW - 40 * HOUR), int(NOW - 40 * HOUR)),
        )
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
            ("t1", "agent", "could not open the file: HTTP 403 access_denied", int(NOW - 20 * HOUR)),
        )
        conn.execute(
            "INSERT INTO task_attachments (task_id, filename, stored_path, created_at) VALUES (?,?,?,?)",
            ("t1", "screenshot.png", "/x/screenshot.png", int(NOW - 40 * HOUR)),
        )
        conn.commit()
        conn.close()
        monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
        monkeypatch.setenv("HERMES_STATE_DB", str(tmp_path / "nope.db"))

        items = dce.build_items(dce.collect_units(24, NOW), NOW, stale_hours=12)
        assert len(items) == 1
        f = items[0]["flags"]
        assert items[0]["id"] == "task:t1"
        assert f["stale"] is True
        assert f["unread_attachment"] is True
        assert f["image_fetch_failure"] is True


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

class TestChallengerBlueprint:
    def test_blueprint_fills_to_valid_spec(self):
        from cron.blueprint_catalog import fill_blueprint, get_blueprint

        bp = get_blueprint("daily-message-challenger")
        assert bp is not None
        spec = fill_blueprint(bp, {"time": "06:00", "packet_bar": "2", "deliver": "local"})
        assert spec["schedule"] == "0 6 * * *"
        assert "EXTERNAL CHALLENGER" in spec["prompt"]
        assert "2 analysis packets" in spec["prompt"]
        assert "[SILENT]" in spec["prompt"]


# ---------------------------------------------------------------------------
# Image bulk-reply helper
# ---------------------------------------------------------------------------

class TestImageBulkReply:
    def test_build_reply_body_appends_marker_and_caps(self):
        body = ibr.build_reply_body("short template")
        assert ibr.RESOLUTION_MARKER in body
        assert len(body) <= ibr.MAX_BODY_CHARS

    def test_finds_affected_slack_thread(self, tmp_path):
        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, title) VALUES (?,?,?,?)",
            ("sx", "slack", NOW - 2 * HOUR, "Broken image thread"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, platform_message_id) "
            "VALUES (?,?,?,?,?)",
            ("sx", "assistant", "Slack attachment access failed: missing_scope",
             NOW - HOUR, "C123:1700000000.001"),
        )
        conn.commit()
        conn.close()
        rows = ibr.find_affected_threads(ibr._open_ro(db), NOW - 24 * HOUR)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == "C123"
        assert rows[0]["thread_ts"] == "1700000000.001"

    def test_idempotent_skips_handled_threads(self, tmp_path):
        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, title) VALUES (?,?,?,?)",
            ("sy", "slack", NOW - 2 * HOUR, "Already handled"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("sy", "assistant", "download failed HTTP 404", NOW - 2 * HOUR),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("sy", "assistant", f"Heads up... {ibr.RESOLUTION_MARKER}", NOW - HOUR),
        )
        conn.commit()
        conn.close()
        rows = ibr.find_affected_threads(ibr._open_ro(db), NOW - 24 * HOUR)
        assert rows == []

    def test_non_slack_source_ignored(self, tmp_path):
        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, title) VALUES (?,?,?,?)",
            ("sz", "telegram", NOW - 2 * HOUR, "TG thread"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("sz", "assistant", "download failed HTTP 404", NOW - HOUR),
        )
        conn.commit()
        conn.close()
        rows = ibr.find_affected_threads(ibr._open_ro(db), NOW - 24 * HOUR)
        assert rows == []

    def test_dry_run_never_sends(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "state.db"
        conn = _make_state_db(db)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, title) VALUES (?,?,?,?)",
            ("sd", "slack", NOW - 2 * HOUR, "Dry run thread"),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, platform_message_id) "
            "VALUES (?,?,?,?,?)",
            ("sd", "assistant", "missing_scope on files.slack.com", NOW - HOUR, "C9:1.2"),
        )
        conn.commit()
        conn.close()
        monkeypatch.setenv("HERMES_STATE_DB", str(db))
        # Freeze the helper's clock to the fixture epoch so inserts fall in-window.
        monkeypatch.setattr(ibr.time, "time", lambda: NOW)

        # Guard: if _send_reply is ever called in dry-run, fail loudly.
        def _boom(*a, **k):
            raise AssertionError("dry-run must not send")

        monkeypatch.setattr(ibr, "_send_reply", _boom)
        rc = ibr.main(["--hours", "24"])  # no --apply
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY-RUN" in out
        assert "C9" in out
