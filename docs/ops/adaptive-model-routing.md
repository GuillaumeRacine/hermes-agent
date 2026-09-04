# Adaptive Model Routing and Authentication Recovery

Hermes can observe a deterministic per-turn routing policy without changing the
live provider. This is intentionally a shadow-only integration: the configured
primary and fallback chain remain authoritative until an evaluated candidate
clears every promotion gate.

## Shadow Routing

Enable observation in `~/.hermes/config.yaml`:

```yaml
adaptive_routing:
  enabled: true
  mode: shadow
  policy_path: /absolute/path/to/routing-policy-shadow.json
  telemetry_path: ~/.hermes/logs/adaptive-routing.jsonl
  private_provider_allowlist:
    - local-ollama
    - openai-codex
    - xai-oauth
    - claude-code
```

The classifier assigns each message to one class:

- `C1`: routine, bounded, reversible transformation or drafting.
- `C2`: multi-step planning, implementation, review, or synthesis.
- `C3`: high-risk security, credential, legal, financial, migration, or
  infrastructure work.
- `C4`: production, deployment, billing, payment, destructive, or incident
  work that also needs a different-provider critic.

Rules are explicit and do not call a model. Each decision receives a random
opaque `decision_id`. Telemetry contains the class, feature counts, current
route, shadow recommendation, and a process-keyed fingerprint. A correlated
`route_outcome` adds only the actual route, completion/failure class, latency,
API/fallback counts, context size, token/cache/reasoning count deltas, and
estimated cost delta. `startup_ready` records classic-CLI input-readiness
latency. These events never contain prompt or response content, raw provider
errors, session/account identifiers, access tokens, or credentials. A private
task suppresses a recommendation whose runtime provider is not allowlisted.

The policy generator must treat output quality as a hard gate before optimizing
latency and cost. A route is not promotion-eligible until it also has the
required sample count, observation span, error rate, privacy compatibility, and
independent critic where required. Shadow mode never applies a selected route.
Live outcomes can quarantine a route or trigger controlled evaluation, but
cannot promote one; signed eval/canary evidence and the configured independent
Claude/Codex review remain mandatory.

## Authentication Recovery

In Slack, type `!auth` because thread replies reserve native slash commands.
The native manifest remains within Slack's command limit, so `/hermes auth ...`
is the equivalent native command.

| Provider type | Status/recovery | Browser behavior |
|---|---|---|
| Hermes OAuth (`xai-oauth`, `openai-codex`, `minimax-oauth`, `nous`) | `!auth status <provider>`, then `!auth open <provider>` | Opens only after the explicit `open` action |
| Claude Code subscription | Run `claude auth status`; recover with `claude auth login` | Claude CLI initiates its own login |
| Codex subscription | Run `codex login status`; recover with `codex login --device-auth` | Codex CLI initiates device/browser login |
| API key (`zai`, OpenRouter, direct MiniMax) | Unlock 1Password, verify/sync, then `!auth refresh` | Browser login does not apply |
| Rate limit / quota exhaustion | Hermes uses the configured fallback and retries later | Never launches login |

An ordinary 401, expired token, missing key, or rate limit never opens a browser
and never invokes 1Password. The user must explicitly request the recovery
action. Work continues through configured fallbacks where possible.

## 1Password

Hermes stores only 1Password secret references:

```sh
hermes secrets onepassword setup \
  --reference 'ZAI_API_KEY=op://vault/item/field' \
  --reference 'OPENROUTER_API_KEY=op://vault/item/field'
hermes secrets onepassword status
hermes secrets onepassword sync
```

`sync` verifies references without retaining values in that CLI process. On
gateway startup the configured references are resolved into process memory.
While the gateway is running, `!auth refresh` explicitly reloads approved secret
sources and evicts the cached agent so the next call uses the refreshed
credential. Secret values are never written to config, logs, Slack, or routing
telemetry.

After login or refresh, use `!auth status <provider>` and retry a small,
read-only request. Authentication failure and rate limiting are distinct:
re-authenticate only for missing/invalid/expired credentials; let fallback
handle 429/quota states.
