"""Persistent, secret-safe provider/model circuit state.

The fallback chain historically remembered a rate limit only on one
``AIAgent`` instance and only for sixty seconds. A new session therefore
retried the same provider even when its quota reset was hours or days away.

This module keeps a small atomic state file shared by CLI, gateway, cron, and
background agents. It stores only provider/model identifiers, a normalized
failure reason, counters, and timestamps. Raw exceptions, response bodies,
prompts, credentials, and account identifiers are never persisted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home
from utils import atomic_json_write

try:  # pragma: no cover - Windows does not provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
try:  # pragma: no cover - non-Windows platforms do not provide msvcrt.
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


SCHEMA_VERSION = 1
_LOCK = threading.RLock()
_DEFAULT_COOLDOWNS = {
    "rate_limit": 3600,
    "billing": 86400,
    "auth": 86400,
    "auth_permanent": 86400,
    "overloaded": 300,
    "server_error": 180,
    "timeout": 180,
    "model_not_found": 86400,
    "unknown": 180,
}
_IMMEDIATE_REASONS = {
    "rate_limit",
    "billing",
    "auth",
    "auth_permanent",
    "model_not_found",
}
_TRANSIENT_REASONS = {"overloaded", "server_error", "timeout", "unknown"}
_PROVIDER_WIDE_REASONS = {"auth", "auth_permanent", "billing"}


def _utc_now(now: float | None = None) -> str:
    return datetime.fromtimestamp(
        time.time() if now is None else now,
        tz=timezone.utc,
    ).isoformat()


def _reason_value(reason: Any) -> str | None:
    value = getattr(reason, "value", reason)
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _DEFAULT_COOLDOWNS else None


def circuit_key(provider: str, model: str = "*") -> str:
    return f"{str(provider or '').strip().lower()}/{str(model or '*').strip() or '*'}"


def state_path(config: dict[str, Any] | None = None) -> Path:
    block = (config or {}).get("provider_circuits") or {}
    configured = str(block.get("state_path") or "").strip()
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return get_hermes_home() / "state" / "provider-circuits.json"


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "circuits": {},
    }


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else state_path()
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return _empty_state()
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("circuits"), dict)
    ):
        return _empty_state()
    return payload


def _load_state_for_write(path: Path) -> dict[str, Any]:
    """Return state without ever replacing a corrupt existing snapshot."""
    if not path.exists():
        return _empty_state()
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"provider circuit state is unreadable: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("circuits"), dict)
    ):
        raise ValueError(f"provider circuit state has an unsupported schema: {path}")
    return payload


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(lock_path, "a+b") as handle:
            if msvcrt is not None:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                if msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, payload)


def _retry_hint_seconds(agent: Any) -> float | None:
    state = getattr(agent, "_rate_limit_state", None)
    if state is None:
        return None
    buckets = [
        getattr(state, "requests_min", None),
        getattr(state, "requests_hour", None),
        getattr(state, "tokens_min", None),
        getattr(state, "tokens_hour", None),
    ]
    depleted = []
    for bucket in buckets:
        if bucket is None:
            continue
        if getattr(bucket, "limit", 0) > 0 and getattr(bucket, "remaining", 0) <= 0:
            try:
                depleted.append(float(bucket.remaining_seconds_now))
            except (TypeError, ValueError):
                continue
    return max(depleted) if depleted else None


def _entry_is_open(entry: dict[str, Any], now: float) -> bool:
    try:
        return float(entry.get("open_until_epoch") or 0) > now
    except (TypeError, ValueError):
        return False


def _entry_updated_epoch(entry: dict[str, Any]) -> float:
    for field in ("last_success_at", "last_failure_at", "opened_at"):
        value = entry.get(field)
        if not value:
            continue
        try:
            return datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            ).timestamp()
        except (TypeError, ValueError):
            continue
    return 0


def circuit_status(
    provider: str,
    model: str,
    *,
    path: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    current = time.time() if now is None else float(now)
    target = Path(path) if path is not None else state_path()
    try:
        payload = _load_state_for_write(target)
    except ValueError:
        return {
            "status": "unavailable",
            "key": circuit_key(provider, model),
            "reason": "state_unreadable",
            "open_until": None,
        }
    entries = [
        entry
        for entry in (
            payload["circuits"].get(circuit_key(provider, model)),
            payload["circuits"].get(circuit_key(provider, "*")),
        )
        if isinstance(entry, dict)
    ]
    if not entries:
        return {"status": "closed", "key": circuit_key(provider, model)}
    active = [entry for entry in entries if _entry_is_open(entry, current)]
    if active:
        entry = active[0]
        return {
            "status": "open",
            "key": str(entry.get("key") or circuit_key(provider, model)),
            "reason": str(entry.get("reason") or "unknown"),
            "open_until": entry.get("open_until"),
        }
    probe = [entry for entry in entries if entry.get("status") == "open"]
    if probe:
        entry = probe[0]
        return {
            "status": "probe_eligible",
            "key": str(entry.get("key") or circuit_key(provider, model)),
            "reason": str(entry.get("reason") or "unknown"),
            "open_until": entry.get("open_until"),
        }
    entry = entries[0]
    if entry.get("status") == "closed":
        return {
            "status": "closed",
            "key": str(entry.get("key") or circuit_key(provider, model)),
            "reason": str(entry.get("reason") or "unknown"),
            "open_until": None,
        }
    return {"status": "closed", "key": circuit_key(provider, model)}


def claim_probe(
    provider: str,
    model: str,
    *,
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    now: float | None = None,
) -> bool:
    """Atomically lease one half-open probe while keeping other callers out."""
    current = time.time() if now is None else float(now)
    target = Path(path) if path is not None else state_path(config)
    block = (config or {}).get("provider_circuits") or {}
    lease_seconds = max(5, int(block.get("probe_lease_seconds") or 120))
    keys = {
        circuit_key(provider, model),
        circuit_key(provider, "*"),
    }
    with _state_lock(target):
        payload = _load_state_for_write(target)
        entries = [
            entry
            for key in keys
            for entry in [payload["circuits"].get(key)]
            if isinstance(entry, dict)
        ]
        if not entries or any(_entry_is_open(entry, current) for entry in entries):
            return False
        expired = [entry for entry in entries if entry.get("status") == "open"]
        if not expired:
            return False
        lease_until = current + lease_seconds
        for entry in expired:
            entry.update(
                half_open=True,
                probe_started_at=_utc_now(current),
                open_until=_utc_now(lease_until),
                open_until_epoch=lease_until,
            )
        payload["updated_at"] = _utc_now(current)
        _write_state(target, payload)
        return True


def is_circuit_open(
    provider: str,
    model: str,
    *,
    path: str | Path | None = None,
    now: float | None = None,
) -> bool:
    return circuit_status(provider, model, path=path, now=now)["status"] == "open"


def record_failure(
    provider: str,
    model: str,
    reason: Any,
    *,
    agent: Any = None,
    retry_after_seconds: float | None = None,
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Record a normalized failure and open the circuit when policy requires."""
    current = time.time() if now is None else float(now)
    reason_name = _reason_value(reason)
    if reason_name is None:
        return {
            "status": "ignored",
            "provider": str(provider or "").strip().lower(),
            "model": str(model or "").strip(),
            "reason": str(getattr(reason, "value", reason) or "unsupported"),
        }
    block = (config or {}).get("provider_circuits") or {}
    threshold = max(1, int(block.get("transient_failure_threshold") or 3))
    cooldowns = dict(_DEFAULT_COOLDOWNS)
    for key, value in (block.get("cooldown_seconds") or {}).items():
        if key in cooldowns:
            try:
                cooldowns[key] = max(1, int(value))
            except (TypeError, ValueError):
                pass
    maximum = max(60, int(block.get("maximum_cooldown_seconds") or 2_592_000))
    hinted = retry_after_seconds
    if hinted is None and agent is not None:
        hinted = _retry_hint_seconds(agent)
    cooldown = float(hinted if hinted is not None and hinted > 0 else cooldowns[reason_name])
    cooldown = min(maximum, max(1.0, cooldown))
    target = Path(path) if path is not None else state_path(config)
    keys = [circuit_key(provider, model)]
    if reason_name in _PROVIDER_WIDE_REASONS:
        keys.append(circuit_key(provider, "*"))

    with _state_lock(target):
        payload = _load_state_for_write(target)
        circuits = payload["circuits"]
        entries = []
        for key in keys:
            previous = (
                circuits.get(key) if isinstance(circuits.get(key), dict) else {}
            )
            prior_failures = int(previous.get("consecutive_failures") or 0)
            consecutive = prior_failures + 1
            should_open = reason_name in _IMMEDIATE_REASONS or (
                reason_name in _TRANSIENT_REASONS and consecutive >= threshold
            )
            previous_until = float(previous.get("open_until_epoch") or 0)
            open_until_epoch = (
                max(previous_until, current + cooldown) if should_open else 0
            )
            key_model = "*" if key.endswith("/*") else str(model or "").strip()
            entry = {
                "key": key,
                "provider": str(provider or "").strip().lower(),
                "model": key_model,
                "reason": reason_name,
                "status": "open" if should_open else "closed",
                "sticky": reason_name in {"auth", "auth_permanent"} and should_open,
                "consecutive_failures": consecutive,
                "opened_at": (
                    previous.get("opened_at")
                    if previous_until > current or previous.get("sticky")
                    else (_utc_now(current) if should_open else None)
                ),
                "last_failure_at": _utc_now(current),
                "open_until": _utc_now(open_until_epoch) if open_until_epoch else None,
                "open_until_epoch": open_until_epoch,
            }
            circuits[key] = entry
            entries.append(entry)

        stale_days = max(1, int(block.get("stale_record_days") or 90))
        stale_before = current - stale_days * 86400
        for stale_key, stale_entry in list(circuits.items()):
            if (
                isinstance(stale_entry, dict)
                and not _entry_is_open(stale_entry, current)
                and not stale_entry.get("sticky")
                and _entry_updated_epoch(stale_entry) < stale_before
            ):
                del circuits[stale_key]

        max_entries = max(16, int(block.get("max_entries") or 256))
        if len(circuits) > max_entries:
            protected = {
                item_key: item
                for item_key, item in circuits.items()
                if isinstance(item, dict)
                and (_entry_is_open(item, current) or item.get("sticky"))
            }
            candidates = sorted(
                (
                    (item_key, item)
                    for item_key, item in circuits.items()
                    if item_key not in protected
                ),
                key=lambda item: str((item[1] or {}).get("last_failure_at") or ""),
                reverse=True,
            )
            remaining = max(0, max_entries - len(protected))
            payload["circuits"] = {
                **protected,
                **dict(candidates[:remaining]),
            }
        payload["updated_at"] = _utc_now(current)
        _write_state(target, payload)
    return entries[0]


def record_success(
    provider: str,
    model: str,
    *,
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    now: float | None = None,
    force: bool = False,
    reasons: set[str] | None = None,
) -> bool:
    """Close an exact or provider-wide circuit after a successful probe."""
    current = time.time() if now is None else float(now)
    target = Path(path) if path is not None else state_path(config)
    changed = False
    with _state_lock(target):
        payload = _load_state_for_write(target)
        if str(model or "").strip() == "*":
            prefix = f"{str(provider or '').strip().lower()}/"
            keys = {
                key
                for key in payload["circuits"]
                if str(key).startswith(prefix)
            }
        else:
            keys = {
                circuit_key(provider, model),
                circuit_key(provider, "*"),
            }
            provider_wide = payload["circuits"].get(circuit_key(provider, "*"))
            if (
                isinstance(provider_wide, dict)
                and str(provider_wide.get("reason") or "")
                in {"auth", "auth_permanent"}
                and not _entry_is_open(provider_wide, current)
            ):
                prefix = f"{str(provider or '').strip().lower()}/"
                keys.update(
                    key
                    for key, value in payload["circuits"].items()
                    if str(key).startswith(prefix)
                    and isinstance(value, dict)
                    and str(value.get("reason") or "")
                    in {"auth", "auth_permanent"}
                )
        for key in keys:
            entry = payload["circuits"].get(key)
            if not isinstance(entry, dict):
                continue
            if reasons is not None and str(entry.get("reason") or "") not in reasons:
                continue
            if (
                not force
                and _entry_is_open(entry, current)
                and not entry.get("half_open")
            ):
                continue
            entry.update(
                status="closed",
                sticky=False,
                half_open=False,
                consecutive_failures=0,
                open_until=None,
                open_until_epoch=0,
                last_success_at=_utc_now(current),
            )
            changed = True
        if changed:
            payload["updated_at"] = _utc_now(current)
            _write_state(target, payload)
    return changed
