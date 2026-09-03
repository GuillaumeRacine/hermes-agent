#!/usr/bin/env python3
"""Image bulk-reply helper — one templated reply per image-fetch-failure thread.

The recurring "image issue" surface (see plugins/platforms/slack/adapter.py's
attachment-fetch diagnostics) leaves individual Slack threads stuck on an
attachment the bot could not read: expired ``files.slack.com`` URLs, missing
scopes, 401/403/404, or rate limits. This helper scans recent activity, reuses
``daily_challenger_export.detect_image_fetch_failure`` to find affected threads,
and posts ONE templated reply to each so the user can clear them in bulk.

Safety posture (deliberate):
  * DRY-RUN BY DEFAULT. It prints exactly which threads it would reply to and
    the exact body. Nothing is sent unless ``--apply`` is passed.
  * Refuses to send in a test/CI context (``CI`` / ``PYTEST_CURRENT_TEST`` /
    ``HERMES_DISABLE_SENDS`` set) even with ``--apply``, unless
    ``--force-ci`` is also given.
  * Sends via the framework's ``tools.send_message_tool._send_to_platform``
    (the same path cron delivery uses), which honours the Slack adapter's
    SSRF guard and ``MAX_MESSAGE_LENGTH`` chunking. The template body is
    truncated to a conservative cap before send.

Only Slack threads are targeted (this is a Slack file-URL problem); state.db
sessions whose ``source`` is a Slack platform and that carry a detected image
fetch failure are the candidates. Threads already replied to in the window (a
later bot message mentioning the resolution marker) are skipped so re-runs are
idempotent.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the export's store-location + failure-detection so the two agree.
try:  # normal package import
    from cron.scripts.daily_challenger_export import (
        detect_image_fetch_failure,
        state_db_path,
        _open_ro,
        _table_exists,
    )
except Exception:  # pragma: no cover - direct-run fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from cron.scripts.daily_challenger_export import (  # type: ignore
        detect_image_fetch_failure,
        state_db_path,
        _open_ro,
        _table_exists,
    )

# Conservative body cap; the Slack adapter enforces its own 39k limit and
# chunks, but the template is short so this is just belt-and-suspenders.
MAX_BODY_CHARS = 3000

# A stable marker embedded in the reply so re-runs skip already-handled threads.
RESOLUTION_MARKER = "[image-attachment-notice]"

DEFAULT_TEMPLATE = (
    "Heads up — I couldn't open the image/file attached in this thread. "
    "Slack's private ``files.slack.com`` links expire and require the bot's "
    "file scopes, so the fetch failed (expired link / missing scope / access "
    "denied). To resolve: re-share the file here, paste the content inline, or "
    "ask a workspace admin to grant the bot ``files:read`` and reinstall the "
    "app. Once it's reachable again I'll pick it up automatically. "
    + RESOLUTION_MARKER
)

# Slack platform source names as they appear in state.db ``sessions.source``.
_SLACK_SOURCES = frozenset({"slack", "gateway:slack", "slack_gateway"})


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _is_slack_source(source: str) -> bool:
    s = (source or "").strip().lower()
    return s in _SLACK_SOURCES or s.startswith("slack")


def find_affected_threads(
    conn: Optional[sqlite3.Connection], since_ts: float
) -> List[Dict[str, Any]]:
    """Return Slack threads in the window with a detected image-fetch failure.

    Each entry: ``{session_id, chat_id, thread_ts, title, evidence}``. A thread
    is skipped if a later message already contains ``RESOLUTION_MARKER`` (so the
    helper is idempotent across re-runs). Never raises.
    """
    if conn is None or not _table_exists(conn, "sessions") or not _table_exists(conn, "messages"):
        return []
    # Be robust if handed a raw connection without a row factory.
    conn.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    try:
        srows = conn.execute(
            "SELECT id, title, source FROM sessions WHERE started_at IS NOT NULL"
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        _eprint(f"image_bulk_reply: state.db read failed: {exc}")
        return []

    for s in srows:
        if not _is_slack_source(s["source"]):
            continue
        sid = s["id"]
        try:
            mrows = conn.execute(
                "SELECT role, content, tool_calls, platform_message_id, timestamp "
                "FROM messages WHERE session_id = ? AND timestamp >= ? ORDER BY timestamp",
                (sid, since_ts),
            ).fetchall()
        except Exception:
            continue

        evidence: Optional[str] = None
        chat_id: Optional[str] = None
        thread_ts: Optional[str] = None
        already_handled = False
        for m in mrows:
            blob = " ".join(str(m[f] or "") for f in ("content", "tool_calls"))
            if RESOLUTION_MARKER in blob:
                already_handled = True
                break
            if evidence is None and detect_image_fetch_failure(blob):
                evidence = blob[:280]
            pmid = m["platform_message_id"]
            if pmid and chat_id is None:
                chat_id, thread_ts = _parse_platform_message_id(pmid)

        if already_handled or evidence is None:
            continue
        # Fall back to the session id if we couldn't parse a chat/thread id;
        # the caller (dry-run) still surfaces it for a human to route.
        out.append(
            {
                "session_id": sid,
                "chat_id": chat_id,
                "thread_ts": thread_ts,
                "title": s["title"] or f"session {str(sid)[:8]}",
                "evidence": evidence,
            }
        )
    return out


def _parse_platform_message_id(pmid: str) -> tuple[Optional[str], Optional[str]]:
    """Best-effort parse of a Slack ``platform_message_id`` into (chat_id, thread_ts).

    Slack platform ids are commonly stored as ``<channel>:<ts>`` (and sometimes
    ``<team>:<channel>:<ts>``). We take the last two colon-separated fields as
    channel + ts. Returns (None, None) if it doesn't look parseable.
    """
    text = str(pmid or "").strip()
    if not text:
        return None, None
    parts = text.split(":")
    if len(parts) >= 2 and parts[-2] and parts[-1]:
        return parts[-2], parts[-1]
    return None, None


def build_reply_body(template: str) -> str:
    """Return the reply body, ensuring the resolution marker is present + capped."""
    body = template or DEFAULT_TEMPLATE
    if RESOLUTION_MARKER not in body:
        body = f"{body} {RESOLUTION_MARKER}"
    return body[:MAX_BODY_CHARS]


def _in_ci() -> bool:
    return any(
        os.environ.get(v, "").strip()
        for v in ("CI", "PYTEST_CURRENT_TEST", "HERMES_DISABLE_SENDS")
    )


def _send_reply(chat_id: str, thread_ts: Optional[str], body: str) -> Optional[str]:
    """Send one reply via the shared platform sender. Returns None or an error str."""
    try:
        import asyncio

        from gateway.config import Platform, load_gateway_config
        from tools.send_message_tool import _send_to_platform
    except Exception as exc:
        return f"send path unavailable: {exc}"

    try:
        config = load_gateway_config()
        platform = Platform("slack")
        pconfig = config.platforms.get(platform)
    except Exception as exc:
        return f"could not load Slack gateway config: {exc}"
    if not pconfig or not getattr(pconfig, "enabled", False):
        return "Slack platform not configured/enabled"

    coro = _send_to_platform(platform, pconfig, chat_id, body, thread_id=thread_ts)
    try:
        asyncio.run(coro)
        return None
    except RuntimeError:
        # A running loop was detected; retry in a fresh thread (mirrors scheduler).
        coro.close()
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    asyncio.run,
                    _send_to_platform(platform, pconfig, chat_id, body, thread_id=thread_ts),
                )
                fut.result(timeout=30)
            return None
        except Exception as exc:
            return f"send failed: {exc}"
    except Exception as exc:
        return f"send failed: {exc}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post one templated reply to each thread stuck on an image-fetch failure."
    )
    parser.add_argument("--hours", type=float, default=72.0,
                        help="Look-back window in hours (default 72).")
    parser.add_argument("--template", default=None,
                        help="Override the reply body (a resolution marker is appended if absent).")
    parser.add_argument("--apply", action="store_true",
                        help="Actually send. Without this, dry-run only (default).")
    parser.add_argument("--force-ci", action="store_true",
                        help="Permit sending even in a CI/test context (dangerous).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of threads acted on (0 = no cap).")
    args = parser.parse_args(argv)

    now = time.time()
    since_ts = now - args.hours * 3600.0
    body = build_reply_body(args.template)

    conn = _open_ro(state_db_path())
    try:
        threads = find_affected_threads(conn, since_ts)
    finally:
        if conn is not None:
            conn.close()

    if args.limit and args.limit > 0:
        threads = threads[: args.limit]

    if not threads:
        print("No image-fetch-failure threads found in the last "
              f"{args.hours:g}h. Nothing to do.")
        return 0

    dry_run = not args.apply
    ci_blocked = _in_ci() and not args.force_ci
    print(f"Found {len(threads)} affected thread(s). "
          f"Mode: {'DRY-RUN' if dry_run else 'APPLY'}"
          f"{' (blocked: CI/test context)' if (not dry_run and ci_blocked) else ''}")
    print(f"\nReply body:\n{body}\n")

    errors = 0
    for i, t in enumerate(threads, 1):
        target = t.get("chat_id") or "<unresolved-channel>"
        thr = t.get("thread_ts") or "<no-thread-ts>"
        print(f"[{i}] {t['title']}  channel={target} thread_ts={thr}")
        print(f"     evidence: {t['evidence']}")
        if dry_run:
            continue
        if ci_blocked:
            print("     SKIPPED (CI/test context; pass --force-ci to override).")
            continue
        if not t.get("chat_id"):
            print("     SKIPPED (could not resolve channel id from platform_message_id).")
            errors += 1
            continue
        err = _send_reply(t["chat_id"], t.get("thread_ts"), body)
        if err:
            print(f"     ERROR: {err}")
            errors += 1
        else:
            print("     SENT.")

    if dry_run:
        print(f"\nDry-run complete. Re-run with --apply to send to {len(threads)} thread(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
