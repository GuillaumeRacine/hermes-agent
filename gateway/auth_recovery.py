"""Operator-facing authentication recovery helpers for gateway surfaces."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path


_BROWSER_OAUTH_PROVIDERS = frozenset(
    {"nous", "openai-codex", "minimax-oauth", "xai-oauth", "anthropic"}
)
_EXTERNAL_LOGIN_COMMANDS = {
    "qwen-oauth": "hermes auth add qwen-oauth",
    "claude-code": "claude /login",
    "copilot-acp": "copilot /login",
}


@dataclass(frozen=True)
class DashboardOpenResult:
    url: str
    dashboard_started: bool
    browser_opened: bool
    error: str = ""


def dashboard_auth_url(provider: str, *, port: int = 9119) -> str:
    provider_id = str(provider or "").strip()
    query = urllib.parse.urlencode({"oauth": provider_id})
    return f"http://127.0.0.1:{int(port)}/env?{query}"


def dashboard_is_listening(*, port: int = 9119, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def open_auth_dashboard(
    provider: str,
    *,
    port: int = 9119,
    startup_timeout: float = 15.0,
) -> DashboardOpenResult:
    """Start the loopback dashboard if needed, then open the login deep link.

    This helper must only be called from an explicit operator action such as
    ``!auth open xai-oauth``. Runtime auth failures should return instructions,
    never invoke it automatically.
    """
    provider_id = str(provider or "").strip()
    url = dashboard_auth_url(provider_id, port=port)
    if provider_id not in _BROWSER_OAUTH_PROVIDERS:
        return DashboardOpenResult(
            url=url,
            dashboard_started=False,
            browser_opened=False,
            error="this provider does not support a Hermes browser login",
        )

    started = False
    if not dashboard_is_listening(port=port):
        command = [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port)),
            "--no-open",
            "--skip-build",
        ]
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            started = True
        except OSError:
            return DashboardOpenResult(
                url=url,
                dashboard_started=False,
                browser_opened=False,
                error="the local Hermes dashboard could not be started",
            )

        deadline = time.monotonic() + max(1.0, float(startup_timeout))
        while time.monotonic() < deadline:
            if dashboard_is_listening(port=port):
                break
            time.sleep(0.2)
        else:
            return DashboardOpenResult(
                url=url,
                dashboard_started=True,
                browser_opened=False,
                error=(
                    "the dashboard did not become ready; run "
                    "`hermes dashboard` locally and retry"
                ),
            )

    try:
        opened = bool(webbrowser.open(url))
    except Exception:
        opened = False
    return DashboardOpenResult(
        url=url,
        dashboard_started=started,
        browser_opened=opened,
        error="" if opened else "the browser could not be opened automatically",
    )


def provider_descriptor(provider: str):
    from hermes_cli.provider_catalog import provider_catalog_by_slug

    return provider_catalog_by_slug().get(str(provider or "").strip())


def recovery_instructions(provider: str, *, typed_prefix: str = "/") -> str:
    """Return secret-safe recovery guidance for one provider."""
    provider_id = str(provider or "").strip()
    descriptor = provider_descriptor(provider_id)
    label = descriptor.label if descriptor else provider_id or "current provider"

    if provider_id in _BROWSER_OAUTH_PROVIDERS:
        return (
            f"{label} uses an account login.\n"
            f"• Open the login on this Mac: `{typed_prefix}auth open {provider_id}`\n"
            f"• CLI fallback: `hermes auth add {provider_id}`\n"
            "Hermes will continue through configured fallbacks until login succeeds."
        )
    if provider_id in _EXTERNAL_LOGIN_COMMANDS:
        return (
            f"{label} is authenticated by an external CLI.\n"
            f"Run locally: `{_EXTERNAL_LOGIN_COMMANDS[provider_id]}`\n"
            "Then retry the request; Hermes will use fallbacks in the meantime."
        )

    env_names = tuple(getattr(descriptor, "api_key_env_vars", ()) or ())
    env_label = ", ".join(env_names) if env_names else "the provider API key"
    return (
        f"{label} uses an API key ({env_label}); browser login does not apply.\n"
        f"• If mapped in 1Password: unlock 1Password, then run "
        f"`{typed_prefix}auth refresh`\n"
        "• Otherwise rotate the key in the provider dashboard and update its "
        "op:// reference or Hermes key configuration.\n"
        "Never paste the key into Slack."
    )


def auth_status_summary(provider: str, *, typed_prefix: str = "/") -> str:
    """Return a best-effort current status without exposing credential values."""
    provider_id = str(provider or "").strip()
    descriptor = provider_descriptor(provider_id)
    if descriptor is None:
        return f"Unknown provider `{provider_id}`."

    if descriptor.tab == "accounts":
        try:
            from hermes_cli.auth import get_auth_status

            status = get_auth_status(provider_id)
        except Exception:
            status = {}
        if bool((status or {}).get("logged_in")):
            return (
                f"✅ {descriptor.label}: account login is available.\n"
                f"If the last request still returned 401, use "
                f"`{typed_prefix}auth open {provider_id}` to replace the login."
            )
        return (
            f"⚠️ {descriptor.label}: login is missing or unusable.\n"
            + recovery_instructions(provider_id, typed_prefix=typed_prefix)
        )

    env_names = tuple(descriptor.api_key_env_vars or ())
    try:
        import os
        from hermes_cli.env_loader import get_secret_source

        active = next((name for name in env_names if os.environ.get(name)), "")
        source = get_secret_source(active) if active else None
    except Exception:
        active = ""
        source = None
    if active:
        suffix = f" via {source}" if source else ""
        return (
            f"✅ {descriptor.label}: {active} is loaded{suffix}.\n"
            "If the provider rejected it, browser login does not apply: unlock "
            f"1Password and run `{typed_prefix}auth refresh`, or rotate the key."
        )
    return (
        f"⚠️ {descriptor.label}: no configured API key is loaded.\n"
        + recovery_instructions(provider_id, typed_prefix=typed_prefix)
    )
