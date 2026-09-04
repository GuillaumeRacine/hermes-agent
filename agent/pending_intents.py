"""Pending-intents ledger: user requests that could not be answered honestly.

Append-only JSONL at ``degraded_mode.queue_path`` (default
``HERMES_HOME/state/pending_intents.jsonl``). Every write holds an exclusive
``fcntl`` lock on a sibling ``.lock`` file so the gateway, cron, and CLI can
share the ledger without interleaving lines. ``mark`` and ``expire_stale``
perform a locked read-modify-write and rewrite the file atomically.

Record shape::

    {
      "id": str, "created_at": iso, "platform": str, "chat_id": str,
      "user_id": str|None, "thread_id": str|None, "text": str,
      "reason": "sub_floor_model" | "interrupted" | "fallbacks_exhausted",
      "status": "pending" | "replayed" | "expired" | "answered",
      "replayed_at": iso|None, "attempts": int, "session_id": str|None,
      "metadata": dict
    }

Dependency-free (stdlib only) so it is trivial to test and to import from
the CLI, the gateway, and the conversation loop alike.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:  # pragma: no cover - Windows has no fcntl; fall back to thread lock only.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

REASONS = ("sub_floor_model", "interrupted", "fallbacks_exhausted")
STATUSES = ("pending", "replayed", "expired", "answered")
DEFAULT_MAX_AGE_HOURS = 24

_THREAD_LOCK = threading.RLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def queue_path(config: Optional[Dict[str, Any]] = None) -> Path:
    block = (config or {}).get("degraded_mode") or {}
    configured = str(block.get("queue_path") or "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "pending_intents.jsonl"


def lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive cross-process + cross-thread lock around ledger mutations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        lock_file = lock_path(path)
        fh = open(lock_file, "a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _read_all(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("id"):
                records.append(rec)
    return records


def _write_all(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def new_record(
    *,
    platform: str,
    chat_id: str,
    text: str,
    reason: str,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if reason not in REASONS:
        raise ValueError(f"unknown pending-intent reason: {reason!r}")
    return {
        "id": uuid.uuid4().hex[:16],
        "created_at": _iso(_utc_now()),
        "platform": str(platform or ""),
        "chat_id": str(chat_id or ""),
        "user_id": None if user_id is None else str(user_id),
        "thread_id": None if thread_id is None else str(thread_id),
        "text": str(text or ""),
        "reason": reason,
        "status": "pending",
        "replayed_at": None,
        "attempts": 0,
        "session_id": None if session_id is None else str(session_id),
        "metadata": dict(metadata or {}),
    }


def enqueue(record: Dict[str, Any], *, path: Optional[Path] = None) -> str:
    """Append ``record`` (from :func:`new_record` or a compatible dict)."""
    target = Path(path) if path is not None else queue_path()
    rec = dict(record)
    rec.setdefault("id", uuid.uuid4().hex[:16])
    rec.setdefault("created_at", _iso(_utc_now()))
    rec.setdefault("status", "pending")
    rec.setdefault("attempts", 0)
    rec.setdefault("replayed_at", None)
    rec.setdefault("metadata", {})
    if rec.get("reason") not in REASONS:
        raise ValueError(f"unknown pending-intent reason: {rec.get('reason')!r}")
    line = json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
    with _locked(target):
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    return str(rec["id"])


def list_all(*, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = Path(path) if path is not None else queue_path()
    with _locked(target):
        return _read_all(target)


def list_pending(
    max_age_hours: Optional[float] = DEFAULT_MAX_AGE_HOURS,
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Pending records not older than ``max_age_hours`` (None = no age cap), oldest first."""
    current = now or _utc_now()
    cutoff = None
    if max_age_hours is not None and max_age_hours > 0:
        cutoff = current - timedelta(hours=float(max_age_hours))
    out: List[Dict[str, Any]] = []
    for rec in list_all(path=path):
        if rec.get("status") != "pending":
            continue
        created = _parse_iso(rec.get("created_at"))
        if cutoff is not None and created is not None and created < cutoff:
            continue
        out.append(rec)
    # Stable sort: records written in the same second keep their append order.
    out.sort(key=lambda r: r.get("created_at") or "")
    return out


def get(record_id: str, *, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    for rec in list_all(path=path):
        if rec.get("id") == record_id:
            return rec
    return None


def mark(
    record_id: str,
    status: str,
    *,
    path: Optional[Path] = None,
    **fields: Any,
) -> bool:
    """Set ``status`` (+ arbitrary ``fields``) on one record under the lock.

    ``status="replayed"`` stamps ``replayed_at`` and bumps ``attempts`` unless
    the caller supplies them. Keys inside ``metadata=`` are merged, not
    replaced. Returns False when the id is unknown.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown pending-intent status: {status!r}")
    target = Path(path) if path is not None else queue_path()
    with _locked(target):
        records = _read_all(target)
        hit = False
        for rec in records:
            if rec.get("id") != record_id:
                continue
            hit = True
            rec["status"] = status
            if status == "replayed":
                rec.setdefault("replayed_at", None)
                if "replayed_at" not in fields:
                    rec["replayed_at"] = _iso(_utc_now())
                if "attempts" not in fields:
                    rec["attempts"] = int(rec.get("attempts") or 0) + 1
            meta_update = fields.pop("metadata", None)
            if isinstance(meta_update, dict):
                merged = dict(rec.get("metadata") or {})
                merged.update(meta_update)
                rec["metadata"] = merged
            rec.update(fields)
            break
        if hit:
            _write_all(target, records)
        return hit


def update_metadata(record_id: str, updates: Dict[str, Any], *, path: Optional[Path] = None) -> bool:
    """Merge ``updates`` into one record's metadata without touching status."""
    target = Path(path) if path is not None else queue_path()
    with _locked(target):
        records = _read_all(target)
        for rec in records:
            if rec.get("id") == record_id:
                merged = dict(rec.get("metadata") or {})
                merged.update(updates)
                rec["metadata"] = merged
                _write_all(target, records)
                return True
    return False


def expire_stale(
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Flip pending records older than ``max_age_hours`` to ``expired``.

    Returns the records that were expired by this call.
    """
    target = Path(path) if path is not None else queue_path()
    current = now or _utc_now()
    cutoff = current - timedelta(hours=float(max_age_hours))
    expired: List[Dict[str, Any]] = []
    with _locked(target):
        records = _read_all(target)
        for rec in records:
            if rec.get("status") != "pending":
                continue
            created = _parse_iso(rec.get("created_at"))
            if created is not None and created < cutoff:
                rec["status"] = "expired"
                rec["expired_at"] = _iso(current)
                expired.append(dict(rec))
        if expired:
            _write_all(target, records)
    return expired


def count_pending(*, path: Optional[Path] = None, chat_id: Optional[str] = None) -> int:
    pending = list_pending(None, path=path)
    if chat_id is not None:
        pending = [r for r in pending if str(r.get("chat_id")) == str(chat_id)]
    return len(pending)


__all__ = [
    "DEFAULT_MAX_AGE_HOURS",
    "REASONS",
    "STATUSES",
    "count_pending",
    "enqueue",
    "expire_stale",
    "get",
    "list_all",
    "list_pending",
    "lock_path",
    "mark",
    "new_record",
    "queue_path",
    "update_metadata",
]
