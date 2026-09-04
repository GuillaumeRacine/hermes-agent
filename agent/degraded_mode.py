"""Degraded-mode policy: is a capable (non-sub-floor) runtime available?

Pure functions, no I/O beyond reading the provider-circuit state file.
Background: hermes-home #233 RC1 — on 2026-09-01 every capable provider was
exhausted, the fallback chain bottomed out on ``local-ollama/llama3.2:3b``,
and the 3B model impersonated the agent in Slack while silently dropping
four user requests. The policy here decides when the gateway must refuse to
answer as the agent and queue the message instead.

Terminology:

* **floor** — the configured set of providers/model-globs that are too weak
  to answer as the agent (``degraded_mode.floor``).
* **sub-floor runtime** — a ``provider/model`` pair that matches the floor.
* **capable runtime** — an entry of the primary + ``fallback_providers`` chain
  that is not sub-floor AND whose provider circuit is not open.
"""

from __future__ import annotations

import fnmatch
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_NOTIFY_TEMPLATE = (
    "Running on an emergency local model — primary providers are exhausted "
    "({detail}). Your message is queued and will be replayed automatically "
    "when a capable model is back (expected {eta})."
)
SUB_FLOOR_EXIT_REASON = "degraded_sub_floor"


def degraded_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    block = (config or {}).get("degraded_mode")
    return block if isinstance(block, dict) else {}


def is_enabled(config: Optional[Dict[str, Any]]) -> bool:
    return bool(degraded_config(config).get("enabled", True))


def _floor(config: Optional[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    floor = degraded_config(config).get("floor")
    if not isinstance(floor, dict):
        floor = {}
    providers = [
        str(p).strip().lower()
        for p in (floor.get("providers") or [])
        if str(p or "").strip()
    ]
    models = [str(m).strip() for m in (floor.get("models") or []) if str(m or "").strip()]
    return providers, models


def is_sub_floor(provider: str, model: str, config: Optional[Dict[str, Any]]) -> bool:
    """True when ``provider/model`` sits at or below the configured floor.

    Provider match is case-insensitive; model globs use :mod:`fnmatch` and are
    matched case-insensitively against the bare model slug.
    """
    providers, models = _floor(config)
    prov = str(provider or "").strip().lower()
    slug = str(model or "").strip()
    if prov and prov in providers:
        return True
    if slug:
        low = slug.lower()
        for pattern in models:
            if fnmatch.fnmatchcase(low, pattern.lower()):
                return True
    return False


def primary_runtime(config: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """Return ``(provider, model)`` for the configured primary runtime."""
    cfg = config or {}
    model_block = cfg.get("model")
    if isinstance(model_block, dict):
        provider = str(model_block.get("provider") or "").strip().lower()
        model = str(model_block.get("default") or "").strip()
    else:
        provider = str(cfg.get("provider") or "").strip().lower()
        model = str(model_block or "").strip()
    return provider, model


def runtime_chain(config: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Primary followed by every valid ``fallback_providers`` entry."""
    chain: List[Tuple[str, str]] = []
    provider, model = primary_runtime(config)
    if provider or model:
        chain.append((provider, model))
    raw = (config or {}).get("fallback_providers") or []
    if isinstance(raw, dict):
        raw = [raw]
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        p = str(entry.get("provider") or "").strip().lower()
        m = str(entry.get("model") or "").strip()
        if p and m:
            chain.append((p, m))
    return chain


def _circuits_enabled(config: Optional[Dict[str, Any]]) -> bool:
    return bool(((config or {}).get("provider_circuits") or {}).get("enabled", True))


def _status_for(provider: str, model: str, config: Dict[str, Any], now) -> Dict[str, Any]:
    from hermes_cli.provider_circuits import circuit_status, state_path

    path = state_path(config)
    # ``circuit_status`` already consults the provider-wide ``provider/*`` key
    # alongside the exact model key; keep an explicit check as well so a
    # provider-wide open circuit is reported by its own key in ``detail``.
    exact = circuit_status(provider, model, path=path, now=now)
    if exact.get("status") == "open":
        return exact
    wide = circuit_status(provider, "*", path=path, now=now)
    if wide.get("status") == "open":
        return wide
    return exact


def _parse_open_until(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def capable_runtime_available(
    config: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """Walk primary + fallback chain for a runtime that may answer as the agent.

    Returns ``(True, "provider/model")`` for the first chain entry that is not
    sub-floor and whose circuit is not open. Otherwise ``(False, detail)``
    where ``detail`` lists every open circuit as ``provider/model until
    <open_until>`` (plus sub-floor entries that were skipped). The earliest
    ``open_until`` is exposed through :func:`earliest_recovery_eta`.
    """
    cfg = config or {}
    chain = runtime_chain(cfg)
    if not chain:
        # No model configured at all (e.g. a bare test config): there is no
        # evidence of exhaustion, so fail open — the gate only trips on open
        # circuits, never on absent configuration.
        return True, "no primary or fallback runtime configured"

    open_parts: List[str] = []
    skipped_sub_floor: List[str] = []
    circuits_on = _circuits_enabled(cfg)
    for provider, model in chain:
        label = f"{provider or '?'}/{model or '?'}"
        if is_sub_floor(provider, model, cfg):
            skipped_sub_floor.append(label)
            continue
        if not circuits_on:
            return True, label
        try:
            status = _status_for(provider, model, cfg, now)
        except Exception as exc:  # pragma: no cover - defensive, fail-open
            logger.warning("degraded_mode: circuit lookup failed for %s: %s", label, exc)
            return True, label
        if status.get("status") != "open":
            return True, label
        until = status.get("open_until") or "unknown"
        reason = status.get("reason") or "unknown"
        open_parts.append(f"{label} until {until} ({reason})")

    detail_bits: List[str] = []
    if open_parts:
        detail_bits.append("open circuits: " + "; ".join(open_parts))
    if skipped_sub_floor:
        detail_bits.append("sub-floor only: " + ", ".join(skipped_sub_floor))
    return False, "; ".join(detail_bits) or "no capable runtime in chain"


def earliest_recovery_eta(
    config: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> Optional[str]:
    """ISO timestamp of the earliest ``open_until`` across the chain, or None."""
    cfg = config or {}
    if not _circuits_enabled(cfg):
        return None
    best: Optional[datetime] = None
    for provider, model in runtime_chain(cfg):
        if is_sub_floor(provider, model, cfg):
            continue
        try:
            status = _status_for(provider, model, cfg, now)
        except Exception:
            continue
        if status.get("status") != "open":
            continue
        dt = _parse_open_until(status.get("open_until"))
        if dt is not None and (best is None or dt < best):
            best = dt
    if best is None:
        return None
    if best.tzinfo is None:
        best = best.replace(tzinfo=timezone.utc)
    return best.astimezone(timezone.utc).isoformat(timespec="minutes").replace("+00:00", "Z")


def format_notice(
    config: Optional[Dict[str, Any]],
    detail: str,
    eta: Optional[str],
) -> str:
    template = degraded_config(config).get("notify_template") or DEFAULT_NOTIFY_TEMPLATE
    values = {"detail": detail or "unknown", "eta": eta or "unknown"}
    try:
        return str(template).format(**values)
    except (KeyError, IndexError, ValueError):
        return DEFAULT_NOTIFY_TEMPLATE.format(**values)


def assess(
    config: Optional[Dict[str, Any]],
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """One-shot bundle: availability, detail, eta and the rendered notice."""
    ok, detail = capable_runtime_available(config, now=now)
    eta = None if ok else earliest_recovery_eta(config, now=now)
    return {
        "available": ok,
        "detail": detail,
        "eta": eta,
        "notice": None if ok else format_notice(config, detail, eta),
    }


def _load_runtime_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config

    try:
        return load_config() or {}
    except Exception:  # pragma: no cover - config unreadable → fail-open
        return {}


def mark_agent_sub_floor_if_needed(agent: Any, provider: str, model: str) -> bool:
    """Flag ``agent`` when the fallback it just activated is sub-floor.

    Called by ``_try_activate_fallback`` right after the runtime switch. Sets
    ``agent._degraded_sub_floor`` and ``agent._degraded_notice`` so the
    conversation loop can end the turn with the fixed notice. Returns True
    when the flag was set.
    """
    cfg = _load_runtime_config()
    if not is_enabled(cfg) or not is_sub_floor(provider, model, cfg):
        return False
    verdict = assess(cfg)
    if verdict["available"]:
        # Circuits say a capable runtime exists, yet the in-process chain
        # still reached the floor (e.g. circuits disabled, or the capable
        # entry sits earlier in the chain and just failed). Be explicit.
        detail = f"fallback chain reached {provider}/{model}"
        notice = format_notice(cfg, detail, verdict.get("eta"))
    else:
        notice = verdict["notice"]
    agent._degraded_sub_floor = True
    agent._degraded_notice = notice
    logger.warning(
        "degraded_mode: fallback landed on sub-floor runtime %s/%s — the turn "
        "will end with the degraded notice instead of an answer",
        provider, model,
    )
    return True


def refresh_agent_sub_floor_flag(agent: Any) -> bool:
    """Re-evaluate a previously set sub-floor flag against the agent's runtime.

    A cached gateway agent may have recovered its primary provider between
    turns; clear the flag then. Keep it while the agent is still parked on a
    sub-floor fallback. Returns the new flag value.
    """
    cfg = _load_runtime_config()
    still = bool(
        is_enabled(cfg)
        and getattr(agent, "_fallback_activated", False)
        and is_sub_floor(getattr(agent, "provider", ""), getattr(agent, "model", ""), cfg)
    )
    agent._degraded_sub_floor = still
    if not still:
        agent._degraded_notice = None
    return still


__all__ = [
    "SUB_FLOOR_EXIT_REASON",
    "mark_agent_sub_floor_if_needed",
    "refresh_agent_sub_floor_flag",
    "assess",
    "capable_runtime_available",
    "earliest_recovery_eta",
    "format_notice",
    "is_enabled",
    "is_sub_floor",
    "primary_runtime",
    "runtime_chain",
]
