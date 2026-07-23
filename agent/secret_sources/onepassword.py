"""1Password CLI-backed secret source.

Hermes stores only ``op://`` references in ``config.yaml``. Secret values are
resolved with the local 1Password CLI and injected into the current process
environment; they are never written to Hermes files or printed.

Background services deliberately fail open when 1Password is locked or the CLI
is unavailable. An operator can unlock the desktop app and explicitly run
``hermes secrets onepassword sync --apply`` (or the gateway ``!auth refresh``)
to retry.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OP_REFERENCE_RE = re.compile(r"^op://[^/\s]+/[^/\s]+/.+$")
_OP_TIMEOUT_SECONDS = 20


@dataclass
class OnePasswordApplyResult:
    """Outcome metadata that never contains secret values."""

    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def find_op(binary: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the configured/PATH ``op`` executable, if available."""
    if binary:
        candidate = Path(binary).expanduser()
        return candidate if candidate.is_file() else None
    resolved = shutil.which("op")
    return Path(resolved) if resolved else None


def validate_references(references: Mapping[str, object]) -> list[str]:
    """Return non-secret validation warnings for configured references."""
    warnings: list[str] = []
    for env_name, reference in references.items():
        if not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name):
            warnings.append(f"ignored invalid environment variable name: {env_name!r}")
            continue
        if not isinstance(reference, str) or not _OP_REFERENCE_RE.fullmatch(reference):
            warnings.append(
                f"{env_name}: reference must use op://vault/item/field syntax"
            )
    return warnings


def _safe_cli_error(stderr: str) -> str:
    """Classify ``op`` failures without echoing provider-controlled details."""
    lowered = (stderr or "").lower()
    if any(token in lowered for token in ("not signed in", "sign in", "signin")):
        return "1Password CLI is not signed in"
    if any(token in lowered for token in ("locked", "authorization prompt")):
        return "1Password is locked; unlock the desktop app and retry"
    if any(token in lowered for token in ("isn't installed", "not found")):
        return "1Password CLI is unavailable"
    if any(token in lowered for token in ("could not find", "doesn't exist", "not exist")):
        return "a configured 1Password reference was not found"
    return "1Password could not resolve a configured reference"


def read_reference(
    reference: str,
    *,
    account: str = "",
    binary: str | os.PathLike[str] | None = None,
    timeout_seconds: float = _OP_TIMEOUT_SECONDS,
) -> str:
    """Resolve one ``op://`` reference without persisting or printing it."""
    op_binary = find_op(binary)
    if op_binary is None:
        raise RuntimeError("1Password CLI (`op`) is not installed or not on PATH")
    if not _OP_REFERENCE_RE.fullmatch(reference):
        raise ValueError("reference must use op://vault/item/field syntax")

    command = [str(op_binary), "read", reference, "--no-newline"]
    if account:
        command.extend(["--account", account])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("1Password CLI timed out") from exc
    except OSError as exc:
        raise RuntimeError("1Password CLI could not be started") from exc

    if completed.returncode != 0:
        raise RuntimeError(_safe_cli_error(completed.stderr or completed.stdout))
    return completed.stdout


def apply_onepassword_secrets(
    *,
    enabled: bool,
    references: Mapping[str, object] | None,
    override_existing: bool = True,
    account: str = "",
    binary: str | os.PathLike[str] | None = None,
) -> OnePasswordApplyResult:
    """Resolve configured references and inject values into ``os.environ``.

    The operation is intentionally all-or-partial: an unavailable reference
    does not remove a working environment value and does not prevent other
    references from loading.
    """
    result = OnePasswordApplyResult()
    if not enabled:
        return result
    if not isinstance(references, Mapping) or not references:
        result.error = "enabled but no references are configured"
        return result
    if find_op(binary) is None:
        result.error = "1Password CLI (`op`) is not installed or not on PATH"
        return result

    for env_name, raw_reference in references.items():
        if not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name):
            result.warnings.append(
                f"ignored invalid environment variable name: {env_name!r}"
            )
            continue
        if not isinstance(raw_reference, str) or not _OP_REFERENCE_RE.fullmatch(
            raw_reference
        ):
            result.warnings.append(
                f"{env_name}: reference must use op://vault/item/field syntax"
            )
            continue
        if os.environ.get(env_name) and not override_existing:
            result.skipped.append(env_name)
            continue
        try:
            value = read_reference(
                raw_reference,
                account=account,
                binary=binary,
            )
        except (RuntimeError, ValueError) as exc:
            result.warnings.append(f"{env_name}: {exc}")
            continue
        if not value:
            result.warnings.append(f"{env_name}: resolved value was empty")
            continue
        os.environ[env_name] = value
        result.applied.append(env_name)

    if not result.applied and result.warnings:
        result.error = "no configured 1Password references could be applied"
    return result
