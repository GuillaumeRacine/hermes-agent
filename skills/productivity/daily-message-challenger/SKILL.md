---
name: daily-message-challenger
description: A scheduled external critic that reviews all recent threads and tasks once a day and posts one tight challenge digest, plus a bulk-reply helper for image-fetch-failure threads.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [cron, challenger, critic, review, slack, attachments, wip]
    category: productivity
---

# Daily Message Challenger Skill

A scheduled, ledger-wide critic — the analog of the per-turn
`agent/background_review.py` reviewer, but running once a day over ALL recent
activity. It reads the framework's real stores, flags the items that are stuck
or sloppy, and hands the agent a compact export so it can post **one** tight
challenge digest. A companion helper bulk-replies to Slack threads that got
stuck on an unfetchable image.

It builds only on **framework stores** — `state.db` (`sessions` + `messages`)
and `kanban.db` (`tasks` + `task_comments` + `task_events` +
`task_attachments`). The custom decision-packet / ledger layer
(`item.json` / `ledger.json` / dispositions) is a runtime overlay that lives in
`~/.hermes` and is **not** required. When a store is missing or empty (fresh
container, ledger absent), the export degrades gracefully: it emits nothing and
the cron job stays `[SILENT]`.

## The rubric (what the challenger scores)

The challenger is an EXTERNAL CHALLENGER, not a cheerleader. Each open item is
scored 0-5 on:

| Dimension | Question |
| --- | --- |
| Decisiveness | Is there a ruling, or just more analysis? |
| Content-extraction | Was attached/linked content actually read and acted on? |
| Ledger integrity | Do claimed states match the live status? |
| Signal-to-noise | Is there fluff or repetition to cut? |
| Image reliability | Any attachment the bot never fetched? |
| Proactivity | Is anything just sitting, waiting to be pushed? |

**WIP posture (the headline failure mode):** any item past **N analysis
packets** (default 2) with no terminal disposition is flagged `over_analyzed`
and called out explicitly.

## Challenge flags (computed by the export)

`cron/scripts/daily_challenger_export.py` groups recent activity into per-thread
/ per-task units and computes:

- **stale** — an open/triage item with no state change in > `--stale-hours` (12).
- **over_analyzed** / **packet_depth** — analysis turns on the item exceed the
  WIP bar with no terminal disposition. Heuristic: for threads, `packet_depth` =
  count of `assistant`/`tool` messages in the window; for tasks, comments +
  events. Terminal disposition = session ended, or task status `done`/`archived`.
- **duplicate_cluster** — near-duplicate items grouped by dependency-free word
  -shingle Jaccard similarity (`--dup-threshold`, default 0.6).
- **unread_attachment** — an item with an image/file attachment and no sign the
  content was read (no "extracted / transcribed / the image shows / …" marker).
- **image_fetch_failure** — an attachment fetch that errored (401/403/404,
  expired `files.slack.com` URL, `missing_scope`, rate limit). Detected from
  message/tool-error text using the **same phrases the Slack adapter emits**
  (`plugins/platforms/slack/adapter.py`), so a failure surfaced to the user is
  also detected here.

The pure flag functions (`flag_stale`, `flag_over_analyzed`,
`flag_unread_attachment`, `flag_image_fetch_failure`, `assign_duplicate_clusters`,
`build_items`) operate on plain dicts and are unit-tested in isolation
(`tests/cron/test_daily_challenger_export.py`).

The export emits a JSON envelope (`{generated_at, window_hours, counts, items}`)
compatible with the `classify_items.py` contract — each item has
`id` / `title` / `summary` / `text` plus its `flags` and a `challenge_flags`
list. Empty result → empty stdout → `[SILENT]`.

## Enable the daily challenger (cron)

The challenger is the `daily-message-challenger` blueprint. Wire the export as
the job's `--script` preprocessor so its JSON lands in the prompt under
`## Script Output`:

```bash
hermes cron create \
  --name "Daily message challenger" \
  --schedule "0 6 * * *" \
  --script "python3 -m cron.scripts.daily_challenger_export --hours 24 --stale-hours 12 --packet-bar 2" \
  --deliver slack \
  --prompt "You are an EXTERNAL CHALLENGER reviewing today's activity export (the ## Script Output above: a JSON envelope with an items list, each carrying challenge flags). You are a critic, not a cheerleader. Score each open item on the rubric (Decisiveness, Content-extraction, Ledger integrity, Signal-to-noise, Image reliability, Proactivity). Enforce the WIP posture: ANY item past 2 analysis packets with no terminal disposition (flag over_analyzed) is the failure mode — call it out. Also surface stale, duplicate_cluster, unread_attachment, and image_fetch_failure items. Post ONE tight digest to the channel: prefer a compact table (item | worst flag | one-line challenge), no preamble, no fluff. If the export is empty or nothing warrants a challenge, respond with exactly [SILENT]."
```

Adjust `--schedule` (cron expr or a human interval like `every 24h`),
`--deliver` (any connected platform, or `local` to save without messaging),
`--hours`, `--stale-hours`, and `--packet-bar` to taste. Do not run this against
production without reviewing the target channel first.

## Image bulk-reply helper

`cron/scripts/image_bulk_reply.py` scans recent Slack threads for the recurring
image-fetch failure (reusing the export's `detect_image_fetch_failure`) and
posts **one** templated reply per affected thread so you can clear them in bulk.

- **Dry-run by default.** It prints exactly which threads it would reply to and
  the body. Nothing sends without `--apply`.
- Refuses to send in a CI/test context (`CI` / `PYTEST_CURRENT_TEST` /
  `HERMES_DISABLE_SENDS`) even with `--apply`, unless `--force-ci` is also given.
- Sends through `tools.send_message_tool._send_to_platform` (the same path cron
  delivery uses), which honours the Slack adapter's SSRF guard and message
  -length cap. The reply carries a stable marker so re-runs are idempotent
  (already-replied threads are skipped).

```bash
# Preview (safe, default):
python3 -m cron.scripts.image_bulk_reply --hours 72

# Actually send after reviewing the preview:
python3 -m cron.scripts.image_bulk_reply --hours 72 --apply

# Custom template + cap the number of threads:
python3 -m cron.scripts.image_bulk_reply --template "Custom note about the file issue" --limit 5 --apply
```

## Notes

- Store paths are resolved via `get_hermes_home()` (never hardcoded), honouring
  `HERMES_HOME`, `HERMES_STATE_DB`, and `HERMES_KANBAN_DB`.
- Everything is read-only against the stores except the bulk-reply `--apply`
  send. No live cron is registered by this skill — you run the
  `hermes cron create` command above when you want it.
