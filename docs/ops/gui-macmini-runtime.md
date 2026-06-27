# Gui Mac Mini Hermes Runtime

Status date: 2026-06-27

This is the non-secret source of truth for Gui's personal Hermes runtime. It is
intended to survive a local machine loss. Do not add API keys, tokens, phone
numbers, raw message content, or private contact data to this file.

## Tracking

GitHub fork:

- `GuillaumeRacine/hermes-agent`

Reliability review issues:

- Parent: `GuillaumeRacine/hermes-agent#1`
- P0 SOUL context loading: `GuillaumeRacine/hermes-agent#2`
- P1 session-skill-review launchd: `GuillaumeRacine/hermes-agent#3`
- P1 auxiliary provider pinning: `GuillaumeRacine/hermes-agent#4`
- P2 latency and long tool turns: `GuillaumeRacine/hermes-agent#5`
- P2 abandoned iMessage path and invisible Unicode sweep: `GuillaumeRacine/hermes-agent#6`
- P3 local backup hygiene and model drift: `GuillaumeRacine/hermes-agent#7`
- Runtime source-of-truth documentation: `GuillaumeRacine/hermes-agent#8`

## Local Layout

Runtime home:

- Compatibility path: `/Users/gui/.hermes`
- Actual path: `/Volumes/SSD/runtime/hermes/home`
- Repo checkout: `/Volumes/SSD/runtime/hermes/home/hermes-agent`
- Runtime backup root: `/Volumes/SSD/runtime/hermes/backups`

Important runtime files:

- Default config: `/Users/gui/.hermes/config.yaml`
- Default loaded soul: `/Users/gui/.hermes/SOUL.md`
- Canonical soul source: `/Users/gui/.hermes/MASTER_SOUL.md`
- Profile roots: `/Users/gui/.hermes/profiles/{music,ops,personal,research,tao}`
- Soul sync script: `/Users/gui/.hermes/scripts/sync_souls.sh`
- Gateway watchdog: `/Users/gui/.hermes/scripts/gateway_watchdog.py`

Secrets stay only in local runtime files such as `.env`, profile `.env` files,
auth files, and platform token files. Keep backup copies under
`/Volumes/SSD/runtime/hermes/backups/config-secrets/<date>/` with mode `700` on
directories and `600` on files.

## Services

Launch agents expected on Gui's Mac mini:

| Label | Purpose | Healthy shape |
|---|---|---|
| `ai.hermes.gateway` | Main multi-channel gateway | Running PID, may show non-zero previous status |
| `ai.hermes.tunnel` | Public tunnel for inbound platform traffic | Running PID, status `0` |
| `ai.hermes.watchdog` | Five-minute gateway health checks | Loaded with status `0` |
| `com.gui.hermes.session-skill-review` | Daily session skill review | Loaded with status `0` |

Check:

```bash
launchctl list | rg 'hermes|session-skill-review'
```

The `session-skill-review` launch agent should call a boot-drive wrapper:

```text
/Users/gui/.local/bin/hermes-session-skill-review
```

That wrapper executes the SSD script through the stable compatibility path:

```bash
exec /bin/bash /Users/gui/.hermes/scripts/run_daily_session_skill_review.sh
```

The launch agent writes logs on the boot drive:

- `/Users/gui/Library/Logs/hermes-session-skill-review.out.log`
- `/Users/gui/Library/Logs/hermes-session-skill-review.err.log`

## Current Reliability Baseline

As of 2026-06-27:

- Gateway channels: Slack, SMS, iMessage via BlueBubbles.
- `SOUL.md` is loaded again after removing a hidden U+200D character from the
  calendar table.
- `hermes doctor` reports no SOUL prompt-safety warning.
- `session-skill-review` is loaded with launchd status `0`.
- Default model is `gpt-5.5`; named profiles currently remain on `gpt-5.4`.
- Auxiliary tasks are pinned to `openai-codex` with `gpt-5.4-mini` models.
- Default `command_allowlist` excludes the abandoned `imsg` commands and keeps
  `/Users/gui/.hermes/scripts/bb_imessage.py`.
- `code_execution.timeout` is capped at `180` seconds across default and named
  profiles.
- Secret-bearing backup files were moved from the active Hermes home to
  `/Volumes/SSD/runtime/hermes/backups/config-secrets/2026-06-27/`.

Known remaining warnings:

- `hermes doctor` still reports optional missing API-key coverage.
- Anthropic and Gemini connectivity warnings may appear until those credentials
  are refreshed or removed.
- Slack Socket Mode disconnect/backfill and long-turn latency need a separate
  evidence pass before code changes.

## Tao Profile

Local profile:

- Name: `tao`
- Alias: `/Users/gui/.local/bin/tao`
- Root: `/Users/gui/.hermes/profiles/tao`
- Model at creation: `gpt-5.4` via `openai-codex`
- Gateway status at creation: stopped

Purpose:

- Outcome owner for Tao of Founders editorial quality, audience resonance,
  virality, reach, engagement, and the weekly growth learning loop.
- Sits above the Tao repo agents. It does not replace repo ownership in
  `tao-content`, `tao-growth`, `tao-insights`, or `tao-commerce`.
- Approval-gated for all outward actions: publishing, scheduling, posting,
  DMs, comments, email, lead contact, DNS, Ghost/Substack/Shopify settings, and
  mailing-list changes.

Context sources covered in the profile:

| Source | Role |
|---|---|
| `/Volumes/SSD/1_Projects/Apps/tao-of-founders/README.md` | Current Tao umbrella and repo split |
| `/Volumes/SSD/1_Projects/Apps/tao-of-founders/*/AGENTS.md` | Repo-specific boundaries and sensitive-file rules |
| `/Users/gui/.claude/projects/-Users-gui/memory/tao_growth_system.md` | Claude-built growth system memory |
| `/Users/gui/.claude/projects/-Volumes-SSD-1-Projects-Apps-tao-founders/memory/project_domain_setup.md` | Current domain architecture |
| `/Users/gui/.claude/projects/-Volumes-SSD-1-Projects-Apps-tao-founders/memory/reference_posting_pipeline.md` | Active Substack Notes pipeline reference |
| `/Users/gui/Context/scripts/substack_notes/RUNBOOK_NOTES.md` | Substack Notes operating runbook |
| `/Users/gui/Context/scripts/substack_notes/NOTES_STRATEGY.md` | Notes voice, format, cadence, and engagement strategy |
| `/Users/gui/Context/scripts/tao_analytics.py` | Legacy/partial analytics script; verify against newer PostHog notes before relying on it |
| `/Users/gui/Context/data/tao_analytics/analytics_2026-04-02.md` | Historical analytics snapshot |
| `/Users/gui/.gemini/tmp/gui/AGENT_HANDOFF_SUBSTACK.md` | Historical Substack extraction handoff |
| `/Volumes/SSD/1_Projects/Apps/tao-founders/archive/tao-apps/growth_system.md` | Legacy acquisition and metrics model |

Important reconciliation notes:

- Prefer the current `tao-of-founders` four-repo split over older monorepo
  paths when docs conflict.
- Later Claude memory says PostHog replaced Plausible on 2026-04-13; treat the
  Plausible-based analytics script as legacy until refreshed.
- `post_note_v2.py`, `schedule.json`, `pool.json`, launchd controls, and any
  Substack/Ghost/Shopify/lead-contact operation are outward-action surfaces.
  Use status/dry-run first and require Gui approval before writes.

Validation performed on 2026-06-27:

```bash
/Users/gui/.local/bin/hermes profile show tao
/Users/gui/.local/bin/hermes profile list
perl -ne 'while(/([\x{200B}-\x{200D}\x{FEFF}\x{2060}])/g){ printf "%s:%d invisible U+%04X\n", $ARGV, $., ord($1) }' \
  /Users/gui/.hermes/profiles/tao/SOUL.md \
  /Users/gui/.hermes/profiles/tao/memories/MEMORY.md \
  /Users/gui/.hermes/profiles/tao/memories/USER.md
```

Expected result:

- `tao` appears in `hermes profile list`.
- `hermes profile show tao` reports existing `.env`, `SOUL.md`, alias, and
  profile root.
- The invisible-Unicode scan prints no matches.

## Validation Commands

Run these after runtime config or source changes:

```bash
/Users/gui/.local/bin/hermes config check
/Users/gui/.local/bin/hermes doctor
/Users/gui/.hermes/scripts/sync_souls.sh
launchctl list | rg 'hermes|session-skill-review'
```

Summarize recent slow gateway turns:

```bash
cd /Users/gui/.hermes/hermes-agent
venv/bin/python -m hermes_cli.main logs slow-turns --threshold 300 --since 7d
```

Scan loaded souls for hidden prompt-safety blockers:

```bash
python3 - <<'PY'
from pathlib import Path
paths = [Path('/Users/gui/.hermes/SOUL.md'),
         Path('/Users/gui/.hermes/MASTER_SOUL.md'),
         *Path('/Users/gui/.hermes/profiles').glob('*/SOUL.md')]
chars = {'\u200b':'U+200B','\u200c':'U+200C','\u200d':'U+200D','\u2060':'U+2060','\ufeff':'U+FEFF'}
for p in paths:
    text = p.read_text(encoding='utf-8', errors='replace')
    hits = [(label, text.count(ch)) for ch, label in chars.items() if text.count(ch)]
    print(f'{p}: {hits or "clean"}')
PY
```

Run focused repo tests for the SOUL prompt-safety changes:

```bash
cd /Users/gui/.hermes/hermes-agent
scripts/run_tests.sh tests/tools/test_threat_patterns.py tests/hermes_cli/test_doctor.py -- -q
```

## Recovery Outline

1. Restore or clone `GuillaumeRacine/hermes-agent` to
   `/Volumes/SSD/runtime/hermes/home/hermes-agent`.
2. Restore private runtime files from encrypted backup material, not from this
   repo: `.env`, profile `.env` files, auth files, platform tokens, and local
   launch agents.
3. Recreate `/Users/gui/.hermes` as a symlink to
   `/Volumes/SSD/runtime/hermes/home`.
4. Install or verify the launch agents listed in this document.
5. Run the validation commands above.
6. Update the tracking issues with exact checks, pass/fail result, and residual
   risks.
