"""Tests for the pending-intents ledger (hermes-home #233 P3-11)."""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent import pending_intents as pi


def _rec(text="hello", chat_id="C1", reason="fallbacks_exhausted", **kw):
    return pi.new_record(
        platform="slack", chat_id=chat_id, user_id="U1", thread_id=None,
        text=text, reason=reason, **kw,
    )


def test_enqueue_list_mark_roundtrip(tmp_path):
    path = tmp_path / "pending.jsonl"
    rid = pi.enqueue(_rec("first"), path=path)
    rid2 = pi.enqueue(_rec("second"), path=path)
    assert pi.lock_path(path).exists()

    pending = pi.list_pending(24, path=path)
    assert [r["id"] for r in pending] == [rid, rid2]
    assert pending[0]["status"] == "pending"
    assert pending[0]["attempts"] == 0
    assert pending[0]["reason"] == "fallbacks_exhausted"

    assert pi.mark(rid, "replayed", path=path, metadata={"replayed_by": "zai/glm-5.2"}) is True
    rec = pi.get(rid, path=path)
    assert rec["status"] == "replayed"
    assert rec["attempts"] == 1
    assert rec["replayed_at"]
    assert rec["metadata"]["replayed_by"] == "zai/glm-5.2"

    assert [r["id"] for r in pi.list_pending(24, path=path)] == [rid2]
    assert pi.count_pending(path=path) == 1
    assert pi.mark("nope", "answered", path=path) is False

    # File stays valid JSONL after the rewrite.
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["id"] for line in lines)


def test_update_metadata_merges(tmp_path):
    path = tmp_path / "pending.jsonl"
    rid = pi.enqueue(_rec(metadata={"a": 1}), path=path)
    assert pi.update_metadata(rid, {"b": 2}, path=path)
    assert pi.get(rid, path=path)["metadata"] == {"a": 1, "b": 2}
    assert pi.get(rid, path=path)["status"] == "pending"


def test_expire_stale_and_age_filter(tmp_path):
    path = tmp_path / "pending.jsonl"
    old = _rec("old")
    old["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
    old_id = pi.enqueue(old, path=path)
    fresh_id = pi.enqueue(_rec("fresh"), path=path)

    # Age filter hides the old record but does not mutate it.
    assert [r["id"] for r in pi.list_pending(24, path=path)] == [fresh_id]
    assert [r["id"] for r in pi.list_pending(None, path=path)] == [old_id, fresh_id]

    expired = pi.expire_stale(24, path=path)
    assert [r["id"] for r in expired] == [old_id]
    assert pi.get(old_id, path=path)["status"] == "expired"
    assert pi.get(fresh_id, path=path)["status"] == "pending"
    # Idempotent.
    assert pi.expire_stale(24, path=path) == []


def test_invalid_reason_and_status_rejected(tmp_path):
    path = tmp_path / "pending.jsonl"
    with pytest.raises(ValueError):
        pi.new_record(platform="slack", chat_id="C", text="x", reason="bogus")
    with pytest.raises(ValueError):
        pi.enqueue({"platform": "slack", "chat_id": "C", "text": "x", "reason": "bogus"}, path=path)
    rid = pi.enqueue(_rec(), path=path)
    with pytest.raises(ValueError):
        pi.mark(rid, "bogus", path=path)


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "pending.jsonl"
    rid = pi.enqueue(_rec(), path=path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    assert [r["id"] for r in pi.list_pending(24, path=path)] == [rid]


def test_concurrent_appends_do_not_interleave(tmp_path):
    path = tmp_path / "pending.jsonl"
    per_thread = 200
    payload = "x" * 2000  # long lines make interleaving observable

    def worker(tag):
        for i in range(per_thread):
            pi.enqueue(_rec(f"{tag}-{i}-{payload}", chat_id=tag), path=path)

    threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 * per_thread
    parsed = [json.loads(line) for line in lines]  # every line must be valid JSON
    assert len({r["id"] for r in parsed}) == 2 * per_thread
    assert sorted(r["chat_id"] for r in parsed).count("t0") == per_thread
