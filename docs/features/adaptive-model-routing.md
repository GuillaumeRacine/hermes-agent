# Adaptive Model Routing

Hermes supports two policy modes:

- `shadow`: classify and record a content-free recommendation without changing
  the configured runtime.
- `guarded`: for an isolated background task epoch, consume only the eval
  policy's quality-gated `selected` native Hermes route.

Conversational sessions stay on their configured provider/model. A short
follow-up is not a new routing decision, and Hermes does not downgrade or
switch a long-context conversation merely because the latest message is easy.

```yaml
adaptive_routing:
  enabled: true
  mode: shadow
  policy_path: /absolute/path/to/routing-policy-shadow.json
  telemetry_path: ~/.hermes/logs/adaptive-routing.jsonl
  pin_path: ~/.hermes/state/adaptive-route-pins.json
  maximum_session_pins: 1024
  maximum_policy_age_seconds: 172800
  private_provider_allowlist:
    - local-ollama
    - openai-codex
    - xai-oauth
    - claude-code
```

Guarded selection fails closed to the configured model when the policy is
missing/malformed, `selected` is null, the runtime is an external worker, the
privacy boundary is violated, credentials cannot resolve, or the selected
provider/model has an open circuit. A private task is blocked entirely if
neither the selected route nor configured primary is allowlisted, and its
fallback chain is filtered to the same boundary. Guarded C4 activation is
intentionally blocked in the current runtime path even when a
different-provider critic is named, because this resolver cannot yet execute
and verify the second pass.
Release QA and allocation changes must still pass an independent Claude or
Codex review.

Uninspected image, audio, or file attachments are conservatively private.
Deterministic privacy detection also covers common credentials, email
addresses, phone numbers, employee/payroll records, and account identifiers.
Unreadable or unwritable task-pin state blocks guarded execution rather than
allowing a task epoch to change provider.

## Persistent provider circuits

Provider/model failures are remembered across CLI, gateway, cron, and
background sessions:

```yaml
provider_circuits:
  enabled: true
  state_path: ~/.hermes/state/provider-circuits.json
  transient_failure_threshold: 3
  maximum_cooldown_seconds: 2592000
  probe_lease_seconds: 120
  max_entries: 256
  stale_record_days: 90
  cooldown_seconds:
    rate_limit: 3600
    billing: 86400
    auth: 86400
    auth_permanent: 86400
    overloaded: 300
    server_error: 180
    timeout: 180
    model_not_found: 86400
    unknown: 180
```

Rate-limit, billing, authentication, and model-not-found failures open
immediately. Transient failures open at the configured threshold. When
rate-limit response headers expose a longer reset, Hermes honors that reset up
to the maximum cooldown. Open fallback entries are skipped before client
construction; an open primary starts the turn on the first healthy fallback.
Authentication failures also open a provider-wide circuit and remain sticky
until a successful provider response or explicit `hermes auth add <provider>`
credential refresh re-admits the provider.

The state file stores only provider/model identifiers, normalized reasons,
counters, and timestamps. It never stores prompts, provider error bodies,
tokens, credentials, or account identifiers.

Quota exhaustion is not an authentication failure: Hermes does not
automatically log in, replace secrets, purchase credits, or increase limits.
After cooldown, the circuit is probe-eligible and normal health/evaluation
automation decides whether it can re-enter service. Re-entry is single-flight:
one caller receives a short half-open probe lease while other sessions continue
to skip the provider. Named custom endpoints retain their own logical circuit
identity instead of sharing a global `custom/*` circuit.
