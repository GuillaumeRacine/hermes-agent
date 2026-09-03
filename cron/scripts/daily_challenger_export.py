#!/usr/bin/env python3
"""Daily "all-messages challenger" export — a ``--script`` preprocessor.

The scheduled, ledger-wide analog of ``agent/background_review.py``. Where the
background reviewer forks the agent after every *turn* to ask "should any
skill/memory be saved?", this script runs once a day over ALL recent activity
and asks the harder question: "which open items are stuck, over-analyzed,
duplicated, or blocked on unread/failed attachments?" It emits a compact JSON
export that a cron job (the *challenger*, see the ``daily-message-challenger``
blueprint) feeds to the agent, which then posts ONE tight challenge digest.

It reads the framework's REAL stores — not the custom decision-packet/ledger
layer, which lives only at runtime in ``~/.hermes`` and is NOT in this repo:

  * ``state.db`` (hermes_state): ``sessions`` + ``messages``.
  * ``kanban.db`` (hermes_cli.kanban_db): ``tasks`` + ``task_comments`` +
    ``task_events`` + ``task_attachments``.

Both DBs are located via ``get_hermes_home()`` (never hardcoded) with the same
env overrides the codebase honours (``HERMES_STATE_DB`` / ``HERMES_KANBAN_DB``).
When a store is missing or empty — the common fresh-container / ledger-absent
case — the corresponding units are simply skipped; the script NEVER crashes and
emits nothing (empty stdout) so the wrapping cron job stays ``[SILENT]``.

Challenge flags (all computed by the pure functions below, so they are unit
testable without a database):

  * ``stale``            — an open/triage item with no state change in > N hours.
  * ``over_analyzed``    — ``packet_depth`` (agent/analysis turns on the item)
                           exceeds the WIP bar (default 2) with no terminal
                           disposition. This is the headline failure mode.
  * ``duplicate_cluster``— near-duplicate items, grouped by dependency-free
                           word-shingle Jaccard similarity.
  * ``unread_attachment``— an item with an image/file attachment but no sign the
                           content was ever read/extracted/acted on.
  * ``image_fetch_failure`` — an attachment fetch that errored (401/403/404,
                           expired ``files.slack.com`` URL, missing scope, rate
                           limit), detected from message/tool-error text using
                           the SAME phrases the Slack adapter emits.

Usage (standalone):
  python3 -m cron.scripts.daily_challenger_export --hours 24 --stale-hours 12

Usage (wired to the challenger cron job): see the ``hermes cron create`` command
in ``skills/productivity/daily-message-challenger/SKILL.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Statuses that count as "still open" (a challenge target). Terminal statuses
# (done/archived) are excluded — matches kanban_db.VALID_STATUSES semantics
# without importing it (keeps this script dependency-free / import-cheap).
OPEN_TASK_STATUSES = frozenset(
    {"triage", "todo", "scheduled", "ready", "running", "blocked", "review"}
)
TERMINAL_TASK_STATUSES = frozenset({"done", "archived"})

# Defaults (all overridable via CLI).
DEFAULT_HOURS = 24
DEFAULT_STALE_HOURS = 12
DEFAULT_PACKET_BAR = 2          # > this many analysis turns w/o disposition = over-analyzed
DEFAULT_DUP_THRESHOLD = 0.6     # Jaccard >= this => same duplicate cluster
DEFAULT_SHINGLE_K = 3


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------
# Store location (never hardcoded — mirrors hermes_state / kanban_db resolution)
# ---------------------------------------------------------------------------

def _hermes_home() -> Path:
    """Return the Hermes home dir, importing the single source of truth if present."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        val = os.environ.get("HERMES_HOME", "").strip()
        if val:
            return Path(val)
        return Path.home() / ".hermes"


def state_db_path() -> Path:
    """Path to ``state.db`` (``HERMES_STATE_DB`` overrides; else ``<home>/state.db``)."""
    override = os.environ.get("HERMES_STATE_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return _hermes_home() / "state.db"


def kanban_db_path() -> Path:
    """Path to ``kanban.db`` (``HERMES_KANBAN_DB`` overrides; else ``<home>/kanban.db``).

    Matches ``kanban_db.kanban_db_path`` for the default board's back-compat
    location without importing the heavy module.
    """
    override = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return _hermes_home() / "kanban.db"


def _open_ro(path: Path) -> Optional[sqlite3.Connection]:
    """Open a SQLite DB read-only. Returns None if missing/unopenable — never raises."""
    try:
        if not path.exists():
            return None
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:  # pragma: no cover - defensive
        _eprint(f"daily_challenger_export: cannot open {path}: {exc}")
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Pure text helpers (dependency-free near-duplicate detection)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> List[str]:
    """Lowercase, strip punctuation, return a token list. Pure."""
    return _WORD_RE.findall((text or "").lower())


def shingles(text: str, k: int = DEFAULT_SHINGLE_K) -> Set[str]:
    """Return the set of k-word shingles for ``text``. Pure.

    Falls back to the individual tokens when the text is shorter than ``k``
    words, so short titles still cluster on exact/near matches.
    """
    tokens = normalize_text(text)
    if len(tokens) < k:
        return set(tokens)
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity of two sets. Pure. Empty/empty => 0.0."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Image-fetch-failure detection — reusable, mirrors the Slack adapter phrasing
# ---------------------------------------------------------------------------
# These patterns intentionally track plugins/platforms/slack/adapter.py's
# _describe_slack_api_error / _describe_slack_download_failure so a failure the
# adapter surfaced to the user is also detected here. Also matches expired
# files.slack.com URLs and rate-limit signals.

_IMAGE_FAILURE_PATTERNS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"slack attachment access failed",
        r"attachment .* (?:is no longer|returned http 404|no longer reachable)",
        r"http\s*40[134]\b",
        r"\b40[134]\s+(?:error|forbidden|unauthorized|not found)",
        r"missing[_ ]scope",
        r"\bnot[_ ]authed\b",
        r"\binvalid[_ ]auth\b",
        r"\btoken[_ ]revoked\b",
        r"\baccount[_ ]inactive\b",
        r"file[_ ](?:not[_ ]found|deleted|access[_ ]denied)",
        r"access[_ ]denied",
        r"no[_ ]permission",
        r"restricted[_ ]action",
        r"not[_ ]allowed[_ ]token[_ ]type",
        r"expired.{0,40}files\.slack\.com",
        r"files\.slack\.com.{0,40}expired",
        r"\bratelimited\b",
        r"rate[ _]?limit",
        r"http\s*429\b",
        r"returned (?:an )?html(?:/login)? (?:or non-media )?response",
        r"slack returned html instead of media",
    )
)


def detect_image_fetch_failure(text: str) -> bool:
    """Return True if ``text`` looks like a failed image/attachment fetch. Pure.

    Reused by both the export and the image bulk-reply helper so the two agree
    on exactly which threads are affected.
    """
    if not text:
        return False
    return any(p.search(text) for p in _IMAGE_FAILURE_PATTERNS)


# Markers suggesting an attachment's content WAS read/extracted/acted on.
_PROCESSED_MARKERS: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bextracted\b",
        r"\btranscrib",
        r"\bocr\b",
        r"the (?:image|screenshot|photo|attachment|file) shows",
        r"i (?:can see|see) (?:in |that )",
        r"caption",
        r"described the (?:image|attachment)",
        r"read the (?:attached|attachment|file|document|pdf)",
        r"contents? of the (?:attachment|file|image)",
    )
)


def _mentions_attachment(text: str) -> bool:
    return bool(
        re.search(
            r"attachment|attached|files\.slack\.com|image_url|\.(?:png|jpe?g|gif|webp|pdf|heic)\b",
            text or "",
            re.IGNORECASE,
        )
    )


def _looks_processed(text: str) -> bool:
    return any(p.search(text or "") for p in _PROCESSED_MARKERS)


# ---------------------------------------------------------------------------
# Pure flag logic — operates on plain "unit" dicts, no DB required
# ---------------------------------------------------------------------------
# A "unit" is one thread (state.db session) or one task (kanban) reduced to:
#   {
#     "id": str, "kind": "thread"|"task", "title": str, "summary": str,
#     "status": str,              # open/triage/running/... or "" for threads
#     "open": bool,               # still open (not terminal / not ended)
#     "last_activity_ts": float,  # epoch seconds of most recent change
#     "analysis_turns": int,      # agent/analysis turns (packet_depth proxy)
#     "has_terminal_disposition": bool,
#     "has_attachment": bool,
#     "attachment_processed": bool,
#     "texts": [str, ...],        # message/comment bodies (for dup + failure scan)
#   }


def flag_stale(unit: Dict[str, Any], now: float, stale_hours: float) -> bool:
    """True if an open item has had no state change in > ``stale_hours`` hours."""
    if not unit.get("open"):
        return False
    last = unit.get("last_activity_ts")
    if not isinstance(last, (int, float)):
        return False
    return (now - float(last)) > stale_hours * 3600.0


def packet_depth(unit: Dict[str, Any]) -> int:
    """The analysis-turn count for an item (the 'packet depth')."""
    n = unit.get("analysis_turns", 0)
    return int(n) if isinstance(n, (int, float)) else 0


def flag_over_analyzed(unit: Dict[str, Any], packet_bar: int = DEFAULT_PACKET_BAR) -> bool:
    """True if the item is past the WIP bar of analysis turns without a disposition.

    The headline failure mode the challenger enforces: any item past
    ``packet_bar`` analysis packets with no terminal disposition is over-analyzed.
    """
    if unit.get("has_terminal_disposition"):
        return False
    return packet_depth(unit) > packet_bar


def flag_unread_attachment(unit: Dict[str, Any]) -> bool:
    """True if the item has an attachment with no sign its content was processed."""
    return bool(unit.get("has_attachment")) and not bool(unit.get("attachment_processed"))


def flag_image_fetch_failure(unit: Dict[str, Any]) -> bool:
    """True if any of the item's texts look like a failed attachment fetch."""
    return any(detect_image_fetch_failure(t) for t in unit.get("texts", []) or [])


def assign_duplicate_clusters(
    units: List[Dict[str, Any]],
    threshold: float = DEFAULT_DUP_THRESHOLD,
    k: int = DEFAULT_SHINGLE_K,
) -> Dict[str, Optional[int]]:
    """Group units into near-duplicate clusters by shingle Jaccard similarity.

    Pure (operates on the unit dicts only). Returns ``{unit_id: cluster_id}``
    where a cluster_id is shared by >=2 similar units, else ``None`` (singleton).
    Single-linkage: a unit joins the first existing cluster it is similar to.
    """
    sigs: List[Tuple[str, Set[str]]] = []
    for u in units:
        text = f"{u.get('title', '')} {u.get('summary', '')}".strip()
        sigs.append((str(u.get("id")), shingles(text, k)))

    cluster_of: Dict[str, int] = {}
    clusters: List[List[int]] = []  # cluster_id -> list of indices
    for i, (uid_i, sig_i) in enumerate(sigs):
        placed = False
        for cid, members in enumerate(clusters):
            if any(jaccard(sig_i, sigs[j][1]) >= threshold for j in members):
                clusters[cid].append(i)
                cluster_of[uid_i] = cid
                placed = True
                break
        if not placed:
            cluster_of[uid_i] = len(clusters)
            clusters.append([i])

    # Only clusters with >=2 members are "duplicate" clusters; singletons -> None.
    result: Dict[str, Optional[int]] = {}
    sizes = {cid: len(members) for cid, members in enumerate(clusters)}
    for uid, cid in cluster_of.items():
        result[uid] = cid if sizes[cid] >= 2 else None
    return result


def compute_flags(
    unit: Dict[str, Any],
    now: float,
    *,
    stale_hours: float = DEFAULT_STALE_HOURS,
    packet_bar: int = DEFAULT_PACKET_BAR,
    duplicate_cluster: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the full challenge-flag set for one unit. Pure."""
    return {
        "stale": flag_stale(unit, now, stale_hours),
        "over_analyzed": flag_over_analyzed(unit, packet_bar),
        "packet_depth": packet_depth(unit),
        "duplicate_cluster": duplicate_cluster,
        "unread_attachment": flag_unread_attachment(unit),
        "image_fetch_failure": flag_image_fetch_failure(unit),
    }


def _has_any_challenge(flags: Dict[str, Any]) -> bool:
    return bool(
        flags.get("stale")
        or flags.get("over_analyzed")
        or flags.get("duplicate_cluster") is not None
        or flags.get("unread_attachment")
        or flags.get("image_fetch_failure")
    )


def build_items(
    units: List[Dict[str, Any]],
    now: float,
    *,
    stale_hours: float = DEFAULT_STALE_HOURS,
    packet_bar: int = DEFAULT_PACKET_BAR,
    dup_threshold: float = DEFAULT_DUP_THRESHOLD,
    shingle_k: int = DEFAULT_SHINGLE_K,
    only_flagged: bool = True,
) -> List[Dict[str, Any]]:
    """Turn raw units into the emitted ``items`` list. Pure.

    Output items are compatible with the ``classify_items.py`` contract:
    each has ``id`` / ``title`` / ``summary`` / ``text``, plus the flags and a
    top-level ``challenge_flags`` list naming which fired (for the challenger).
    """
    clusters = assign_duplicate_clusters(units, dup_threshold, shingle_k)
    items: List[Dict[str, Any]] = []
    for u in units:
        uid = str(u.get("id"))
        flags = compute_flags(
            u,
            now,
            stale_hours=stale_hours,
            packet_bar=packet_bar,
            duplicate_cluster=clusters.get(uid),
        )
        if only_flagged and not _has_any_challenge(flags):
            continue
        challenge_flags = [
            name
            for name in ("stale", "over_analyzed", "unread_attachment", "image_fetch_failure")
            if flags.get(name)
        ]
        if flags.get("duplicate_cluster") is not None:
            challenge_flags.append("duplicate_cluster")
        items.append(
            {
                "id": uid,
                "kind": u.get("kind"),
                "title": u.get("title") or uid,
                "summary": u.get("summary") or "",
                "text": _item_text(u, flags, challenge_flags),
                "status": u.get("status") or "",
                "open": bool(u.get("open")),
                "flags": flags,
                "challenge_flags": challenge_flags,
            }
        )
    return items


def _item_text(unit: Dict[str, Any], flags: Dict[str, Any], challenge_flags: List[str]) -> str:
    """A compact human/LLM-readable one-liner summarizing the unit's challenges."""
    parts = [
        f"{unit.get('kind', 'item')} '{unit.get('title') or unit.get('id')}'",
        f"status={unit.get('status') or ('open' if unit.get('open') else 'closed')}",
        f"packet_depth={flags.get('packet_depth')}",
    ]
    if challenge_flags:
        parts.append("flags=" + ",".join(challenge_flags))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Store readers — best-effort; return [] when the store is missing/empty
# ---------------------------------------------------------------------------

def load_thread_units(conn: Optional[sqlite3.Connection], since_ts: float, now: float) -> List[Dict[str, Any]]:
    """Reduce recent state.db sessions/messages into units. Never raises."""
    if conn is None or not _table_exists(conn, "sessions") or not _table_exists(conn, "messages"):
        return []
    units: List[Dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT s.id AS id, s.title AS title, s.started_at AS started_at,
                   s.ended_at AS ended_at, s.source AS source,
                   MAX(m.timestamp) AS last_ts,
                   COUNT(m.id) AS msg_count,
                   SUM(CASE WHEN m.role IN ('assistant','tool') THEN 1 ELSE 0 END) AS analysis_turns
            FROM sessions s
            JOIN messages m ON m.session_id = s.id
            WHERE m.timestamp >= ?
            GROUP BY s.id
            """,
            (since_ts,),
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        _eprint(f"daily_challenger_export: state.db read failed: {exc}")
        return []

    for r in rows:
        sid = r["id"]
        texts: List[str] = []
        has_attachment = False
        try:
            mrows = conn.execute(
                "SELECT role, content, tool_name, tool_calls FROM messages "
                "WHERE session_id = ? AND timestamp >= ? ORDER BY timestamp",
                (sid, since_ts),
            ).fetchall()
        except Exception:
            mrows = []
        for m in mrows:
            for field in ("content", "tool_calls"):
                val = m[field]
                if val:
                    texts.append(str(val))
                    if _mentions_attachment(str(val)):
                        has_attachment = True
        blob = "\n".join(texts)
        attachment_processed = _looks_processed(blob) if has_attachment else False
        ended = r["ended_at"]
        units.append(
            {
                "id": f"thread:{sid}",
                "kind": "thread",
                "title": r["title"] or f"session {str(sid)[:8]}",
                "summary": (texts[0][:280] if texts else ""),
                "status": "closed" if ended else "open",
                "open": ended is None,
                "last_activity_ts": float(r["last_ts"]) if r["last_ts"] is not None else now,
                "analysis_turns": int(r["analysis_turns"] or 0),
                "has_terminal_disposition": ended is not None,
                "has_attachment": has_attachment,
                "attachment_processed": attachment_processed,
                "texts": texts,
            }
        )
    return units


def load_task_units(conn: Optional[sqlite3.Connection], since_ts: float, now: float) -> List[Dict[str, Any]]:
    """Reduce recent kanban tasks/comments/events/attachments into units. Never raises."""
    if conn is None or not _table_exists(conn, "tasks"):
        return []
    has_comments = _table_exists(conn, "task_comments")
    has_events = _table_exists(conn, "task_events")
    has_attach = _table_exists(conn, "task_attachments")
    units: List[Dict[str, Any]] = []
    try:
        trows = conn.execute("SELECT * FROM tasks").fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        _eprint(f"daily_challenger_export: kanban.db read failed: {exc}")
        return []

    for t in trows:
        tid = t["id"]
        status = (t["status"] or "").strip().lower()
        # created_at/started_at/completed_at are epoch seconds (INTEGER).
        stamps = [t["created_at"], t["started_at"], t["completed_at"]]
        comment_bodies: List[str] = []
        comment_count = 0
        event_count = 0
        if has_comments:
            try:
                crows = conn.execute(
                    "SELECT body, created_at FROM task_comments WHERE task_id = ?", (tid,)
                ).fetchall()
                comment_count = len(crows)
                for c in crows:
                    comment_bodies.append(str(c["body"] or ""))
                    stamps.append(c["created_at"])
            except Exception:
                pass
        if has_events:
            try:
                erows = conn.execute(
                    "SELECT kind, payload, created_at FROM task_events WHERE task_id = ?", (tid,)
                ).fetchall()
                event_count = len(erows)
                for e in erows:
                    if e["payload"]:
                        comment_bodies.append(str(e["payload"]))
                    stamps.append(e["created_at"])
            except Exception:
                pass
        has_attachment = False
        if has_attach:
            try:
                arow = conn.execute(
                    "SELECT COUNT(*) AS n FROM task_attachments WHERE task_id = ?", (tid,)
                ).fetchone()
                has_attachment = bool(arow and arow["n"])
            except Exception:
                has_attachment = False

        last_activity = max((float(s) for s in stamps if isinstance(s, (int, float))), default=now)
        # Skip tasks with no activity in the window (keeps the export focused).
        if last_activity < since_ts:
            continue

        body = str(t["body"] or "")
        texts = [body] + comment_bodies
        blob = "\n".join(texts)
        if body and _mentions_attachment(body):
            has_attachment = True
        attachment_processed = _looks_processed(blob) if has_attachment else False
        is_open = status in OPEN_TASK_STATUSES
        # packet_depth heuristic for tasks: comments + agent events approximate
        # the number of analysis turns spent on the item.
        analysis_turns = comment_count + event_count
        units.append(
            {
                "id": f"task:{tid}",
                "kind": "task",
                "title": t["title"] or f"task {tid}",
                "summary": body[:280],
                "status": status,
                "open": is_open,
                "last_activity_ts": last_activity,
                "analysis_turns": analysis_turns,
                "has_terminal_disposition": status in TERMINAL_TASK_STATUSES,
                "has_attachment": has_attachment,
                "attachment_processed": attachment_processed,
                "texts": texts,
            }
        )
    return units


def collect_units(hours: float, now: float) -> List[Dict[str, Any]]:
    """Load units from both stores. Missing stores contribute nothing."""
    since_ts = now - hours * 3600.0
    units: List[Dict[str, Any]] = []
    sconn = _open_ro(state_db_path())
    try:
        units.extend(load_thread_units(sconn, since_ts, now))
    finally:
        if sconn is not None:
            sconn.close()
    kconn = _open_ro(kanban_db_path())
    try:
        units.extend(load_task_units(kconn, since_ts, now))
    finally:
        if kconn is not None:
            kconn.close()
    return units


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    keys = ("stale", "over_analyzed", "unread_attachment", "image_fetch_failure")
    out = {k: 0 for k in keys}
    out["duplicate_cluster"] = 0
    for it in items:
        f = it.get("flags", {})
        for k in keys:
            if f.get(k):
                out[k] += 1
        if f.get("duplicate_cluster") is not None:
            out["duplicate_cluster"] += 1
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export recent activity with challenge flags for the daily challenger."
    )
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                        help=f"Look-back window in hours (default {DEFAULT_HOURS}).")
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_HOURS,
                        help=f"Open item is 'stale' after this many hours idle (default {DEFAULT_STALE_HOURS}).")
    parser.add_argument("--packet-bar", type=int, default=DEFAULT_PACKET_BAR,
                        help=f"Analysis turns past which an item is 'over_analyzed' (default {DEFAULT_PACKET_BAR}).")
    parser.add_argument("--dup-threshold", type=float, default=DEFAULT_DUP_THRESHOLD,
                        help=f"Jaccard similarity for a duplicate cluster (default {DEFAULT_DUP_THRESHOLD}).")
    parser.add_argument("--shingle-k", type=int, default=DEFAULT_SHINGLE_K,
                        help=f"Word-shingle size for dup detection (default {DEFAULT_SHINGLE_K}).")
    parser.add_argument("--all", action="store_true",
                        help="Emit every unit, not just flagged ones.")
    parser.add_argument("--emit-empty", action="store_true",
                        help="Emit the JSON envelope even when there are no items "
                             "(default: print nothing so cron stays [SILENT]).")
    parser.add_argument("--out-dir", default=os.environ.get("HERMES_CRON_OUTPUT_DIR") or None,
                        help="Also write the full export JSON to this directory.")
    args = parser.parse_args(argv)

    now = time.time()
    try:
        units = collect_units(args.hours, now)
    except Exception as exc:  # pragma: no cover - top-level safety net
        _eprint(f"daily_challenger_export: unexpected error, emitting empty: {exc}")
        units = []

    items = build_items(
        units,
        now,
        stale_hours=args.stale_hours,
        packet_bar=args.packet_bar,
        dup_threshold=args.dup_threshold,
        shingle_k=args.shingle_k,
        only_flagged=not args.all,
    )

    envelope = {
        "generated_at": now,
        "window_hours": args.hours,
        "packet_bar": args.packet_bar,
        "units_scanned": len(units),
        "counts": _counts(items),
        "items": items,
    }

    # Optionally persist the full export to the job output dir (audit trail).
    if args.out_dir:
        try:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "daily_challenger_export.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - non-fatal
            _eprint(f"daily_challenger_export: could not write out-dir: {exc}")

    # Empty + not forced -> print nothing so the wrapping cron job stays [SILENT].
    if not items and not args.emit_empty:
        return 0

    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
