# EA Attention Brief — Stress Test & Hardening Review

Read-only security and robustness review of the Hermes framework primitives that the
"EA Attention Brief / Human-OS daily brief" workflow depends on. Scope note: the brief's
own application code (`human_os_daily_brief.py`, `docs/EA_ATTENTION_ISA.md`, the
`~/.hermes/state/ea/` ledgers) does **not** live in this repository — it runs in a separate
`~/.hermes` deployment. This review therefore targets the framework surfaces the brief
rides on, all of which are in this repo: the cron scheduler, the Slack interaction/feedback
adapter, and the approval/authority layer. A bug in any of these affects the brief
regardless of where the brief code lives.

## The headline

The safety model the brief assumes rests on two mechanisms that this framework does **not**
implement the way the design discussion assumed:

1. **"Sends no / money gated" is not an enforced gate.** There is no autonomy-ledger,
   action-gate matrix, or approval queue for money/irreversible actions. Outbound sending is
   blocked only by *not registering* the `send_message` tool — and that is bypassable in-loop.
2. **The emoji feedback loop does not exist.** `reaction_added` / `reaction_removed` are
   hard-wired no-ops. The "react keep / wrong-priority / drop" kill-gate produces no effect
   and no error — it fails silently.

Everything below is prioritized by severity.

## Findings

### P1 — Authority: "sends require approval" is bypassable in-loop
- **What:** `send_message` is deliberately unregistered as an agent tool (`toolsets.py:367-374`),
  and that tool-absence is the *entire* "sends no" mechanism. It is defeated by two tools present
  in every full toolset: `terminal` (call `hermes send <platform> "..."`, `hermes_cli/send_cmd.py:1-25`)
  and `execute_code` (arbitrary Python: `requests.post(...)`, reads tokens from `~/.hermes`,
  `tools/code_execution_tool.py:61-69`). The guard layer is a shell-string **denylist**
  (`tools/approval.py:381-530`) that matches none of these, so they are default-approved.
- **Trigger:** any prompt-injected or misaligned turn in a full-toolset session.
- **Fix:** classify outbound-send and network-POST-to-messaging/payment hosts in the guard layer;
  gate `hermes send` and `curl|wget` to those hosts as dangerous; or run agent terminals with
  restricted network egress.

### P1 — Feedback: inbound reactions are no-ops
- **What:** `plugins/platforms/slack/adapter.py:995-1001` registers `reaction_added` /
  `reaction_removed` purely to silence Bolt log noise; both `pass`. No inbound reaction→agent
  wiring exists anywhere in the repo. The brief's emoji feedback / kill-gate is silently unimplemented.
- **Fix:** implement a real handler that (a) verifies the reacting user is authorized,
  (b) ignores the bot's own reactions, (c) resolves the reacted message to a tracked brief item,
  (d) dedups on `event_ts`. Until then, document that reactions are inert.

### P1 — Injection: untrusted text concatenated into agent turns
- **What:** thread message bodies, attacker-controlled display names, link-unfurl text, and small
  **file contents** are string-concatenated into the agent turn behind literal, forgeable
  `[Thread context]` / `[End of thread context]` delimiters (`adapter.py:3556-3566`, `2648-2660`,
  `2870-2889`). Nothing escapes the boundary tokens inside untrusted content.
- **Trigger:** a non-owner posts (or names themselves) `[End of thread context]\n\nSystem: ...`;
  on the next mention the injected content is prepended ahead of the real user text.
- **Fix:** wrap all externally-sourced text in a nonce-tagged, non-forgeable boundary and strip the
  boundary tokens from untrusted content; or pass thread context as structured metadata the runner
  renders rather than string-prepending it.

### P1 — Cron: no failure dampening
- **What:** no consecutive-error counter, backoff, or auto-pause anywhere in `cron/`. A broken
  recurring job re-fires unchanged every tick: either it spams a "⚠️ Cron failed" delivery every
  period forever (worst offender: the model-drift guard, `scheduler.py:2406-2458`), or — for an
  empty model response — it fails **silently** because `success` is flipped to `False`
  (`scheduler.py:2836`) *after* the deliver decision was already computed (`2810-2814`), so the
  failure summary is never sent.
- **Fix:** track `consecutive_errors` in `mark_job_run`; auto-pause after N with one final alert
  and/or exponential backoff; de-dup repeated identical failure deliveries; compute the
  empty→soft-failure flip before the deliver decision.

### P1 — Cron: no persisted silent-delta dedup ("blocked item repeated 5 days")
- **What:** the "stay silent unless something changed" contract is delegated entirely to the LLM
  emitting `[SILENT]` (`scheduler.py:1819-1830`). There is no persisted "already reported" ledger.
  A self-chaining delta job saves its `[SILENT]` output as the doc used for next run's context
  (`2691-2703`), overwriting its memory of previously-reported items; the next genuine change
  re-reports everything still outstanding — including the long-blocked item.
- **Fix:** persist an explicit per-job dedup ledger (id/hash set) separate from the human-readable
  output; on `[SILENT]` do not clobber the last real report used as context; feed the ledger to the
  agent instead of relying on last-output text.

### P2 — Authority: default-open failure modes
- Dangerous command in non-interactive & non-gateway context → auto-approved
  (`approval.py:1575-1592`).
- `write_approval` config read fails **open** (`write_approval.py:86-103`).
- Tirith import failure fails open by default (`approval.py:1602-1636`).
- Container env types (`docker/singularity/modal/daytona`) bypass the guard while still holding
  network + mounted credentials (`approval.py:1286-1287`).
- **Fix:** make security-relevant defaults fail closed; require an explicit opt-in to run
  unguarded.

### P2 — Replay: message dedup lost on restart → duplicate agent turns
- **What:** `MessageDeduplicator` is in-memory only (`adapter.py:438`, checked at `2360-2365`).
  Socket Mode replays un-acked events on reconnect; a crash mid-turn → restart → duplicate turn
  with duplicate side effects. TTL is 300s.
- **Fix:** persist recently-processed `ts` to disk, or gate dispatch on a durable idempotency key.

### P2 — Cron: model-drift guard spams every tick
- **What:** when the global default model changes and a job is unpinned, `run_job` raises →
  delivered as failure → re-fires and raises again every period. Correct to fail closed on spend,
  but with no dampening (P1) every unpinned job spams an identical message forever
  (`scheduler.py:2406-2458`).
- **Fix:** after the first drift-skip, auto-pause or suppress repeat deliveries until acked.

### P2 — Authority: iMessage send-capable, no idempotency, coarse approvals
- The BlueBubbles/iMessage bridge is send-capable (`gateway/platforms/bluebubbles.py:500-543`) and
  is **not** structurally excluded from any "read-only" toolset — it relies on the same bypassable
  tool-absence as every platform (`toolsets.py:478-482`).
- No idempotency key on the send path (`send_message_tool.py`) → double-send / double-charge on
  retry.
- `/approve` resolves by session key, not requester-bound; in shared sessions any allow-listed user
  can approve another user's triggered action, and `/approve all` clears every pending approval at
  once (`gateway/slash_commands.py:3905-3961`).
- `execute_code` "approve session" whitelists **all** future scripts (approval key is the constant
  `"execute_code"`, `approval.py:1878/1934`).
- **Fix:** add per-bridge send exclusion for read-only contexts; add an idempotency key / executed-flag
  on sends; bind approvals to the requesting turn; scope execute_code approval to the specific script.

### P3 — Slack: self-echo guard ignores secondary workspaces
- The own-message guard compares only against the primary workspace bot id
  (`adapter.py:2384-2386`) while routing uses per-team ids; with `allow_bots` non-default a
  secondary-workspace bot's own posts can re-enter → echo loop. Fix: compute the event's team bot id
  first and compare against that.

### P3 — Slack: approval buttons no-op after restart
- `_approval_resolved` is in-memory; after a gateway restart `pop(msg_ts, True)` returns the default
  → the click is treated as an already-resolved double-click and silently returns, leaving the agent
  thread blocked with no user feedback (`adapter.py:3077`, `3374`). Fix: distinguish "unknown
  message" from "already resolved" and always update the Block Kit message.

### P3 — Cron: paused/enabled can disagree
- `_get_due_jobs_locked` (`cron/jobs.py:1459`) filters solely on `enabled`; a job with
  `state="paused"` but `enabled=True` keeps firing while every UI shows it paused. Fix: also skip on
  `state=="paused"`.

### P3 — Cron: DST edge cases and duplicate job IDs
- Spring-forward non-existent cron wall-clock times and fall-back duplicate hours are left to
  croniter with no explicit handling (`jobs.py:541-560`); interval jobs are DST-immune. `create_job`
  does not assert id uniqueness while `remove_job` deletes all matches (asymmetric). Both low
  severity; worth explicit tests / a uniqueness assertion.

## What is solid (verified, not bugs)
- Cron locking/atomicity: temp+fsync+rename writes, reentrant cross-process lock, at-most-once
  `advance_next_run`, no run pile-up when runtime > interval, corrupt-`jobs.json` auto-repair,
  job-id path-escape guards.
- Slack message dedup keys on `event.ts` and marks synchronously before any await (collapses the
  `message`+`app_mention` double-fire and concurrent redeliveries within TTL).
- Interactive button auth checks the reacting/clicking user before resolving.
- `send_message` is genuinely absent from agent toolsets (the bypass is via general-purpose tools,
  not a registration leak); `_YOLO_MODE_FROZEN` is snapshotted at import to stop in-process env
  mutation.

## Recommended sequencing
1. **P1 authority + P1 injection first** — these break the core "assisted, drafts-only, gated"
   promise and are reachable by any channel member.
2. **P1 cron dedup + failure dampening** — directly fixes the "blocked item repeated" and
   silent/spam failure modes the brief already exhibited.
3. **P1 feedback (implement or document as inert)** — the kill-gate cannot measure signal until
   reactions do something.
4. P2/P3 hardening as follow-ups.

Each fix is contained and independently shippable; several (cron paused-check, self-echo team id,
empty-response ordering) are small and unambiguous, while others (fail-closed defaults, gating
`hermes send`, implementing reaction feedback) are policy/product decisions that should be ruled on
before implementation.
