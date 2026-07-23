from __future__ import annotations

import asyncio
from unittest import mock

from gateway import auth_recovery
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def test_dashboard_auth_url_encodes_provider():
    assert (
        auth_recovery.dashboard_auth_url("xai-oauth")
        == "http://127.0.0.1:9119/env?oauth=xai-oauth"
    )


def test_open_rejects_non_browser_provider_without_starting_process(monkeypatch):
    popen = mock.Mock()
    monkeypatch.setattr(auth_recovery.subprocess, "Popen", popen)
    result = auth_recovery.open_auth_dashboard("zai")
    assert not result.browser_opened
    assert "does not support" in result.error
    popen.assert_not_called()


def test_open_starts_dashboard_only_after_explicit_call(monkeypatch):
    listening = iter([False, True])
    monkeypatch.setattr(
        auth_recovery,
        "dashboard_is_listening",
        lambda **_kwargs: next(listening),
    )
    popen = mock.Mock()
    monkeypatch.setattr(auth_recovery.subprocess, "Popen", popen)
    browser = mock.Mock(return_value=True)
    monkeypatch.setattr(auth_recovery.webbrowser, "open", browser)

    result = auth_recovery.open_auth_dashboard(
        "xai-oauth",
        startup_timeout=1,
    )

    assert result.dashboard_started
    assert result.browser_opened
    popen.assert_called_once()
    command = popen.call_args.args[0]
    assert command[-2:] == ["--no-open", "--skip-build"]
    browser.assert_called_once_with(
        "http://127.0.0.1:9119/env?oauth=xai-oauth"
    )


def test_api_key_recovery_never_suggests_browser_login(monkeypatch):
    descriptor = mock.Mock(
        label="Z.AI",
        api_key_env_vars=("ZAI_API_KEY",),
    )
    monkeypatch.setattr(
        auth_recovery,
        "provider_descriptor",
        mock.Mock(return_value=descriptor),
    )
    message = auth_recovery.recovery_instructions("zai", typed_prefix="!")
    assert "!auth refresh" in message
    assert "browser login does not apply" in message
    assert "Never paste the key into Slack" in message


def test_oauth_recovery_uses_slack_typed_prefix(monkeypatch):
    descriptor = mock.Mock(label="xAI")
    monkeypatch.setattr(
        auth_recovery,
        "provider_descriptor",
        mock.Mock(return_value=descriptor),
    )
    message = auth_recovery.recovery_instructions(
        "xai-oauth",
        typed_prefix="!",
    )
    assert "!auth open xai-oauth" in message
    assert "hermes auth add xai-oauth" in message


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id="operator",
            chat_id="rentals",
        ),
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        Platform.SLACK: mock.Mock(typed_command_prefix="!"),
    }
    return runner


def test_gateway_auth_status_uses_current_provider_and_slack_prefix():
    runner = _runner()
    with mock.patch(
        "gateway.run._load_gateway_config",
        return_value={"model": {"provider": "xai-oauth"}},
    ), mock.patch(
        "gateway.auth_recovery.auth_status_summary",
        return_value="status-result",
    ) as summary:
        result = asyncio.run(runner._handle_auth_command(_event("/auth")))

    assert result == "status-result"
    summary.assert_called_once_with("xai-oauth", typed_prefix="!")


def test_gateway_auth_open_is_explicit_and_returns_deep_link():
    runner = _runner()
    opened = auth_recovery.DashboardOpenResult(
        url="http://127.0.0.1:9119/env?oauth=xai-oauth",
        dashboard_started=True,
        browser_opened=True,
    )
    with mock.patch(
        "gateway.run._load_gateway_config",
        return_value={"model": {"provider": "xai-oauth"}},
    ), mock.patch(
        "gateway.auth_recovery.open_auth_dashboard",
        return_value=opened,
    ) as open_dashboard:
        result = asyncio.run(
            runner._handle_auth_command(_event("/auth open xai-oauth"))
        )

    open_dashboard.assert_called_once_with("xai-oauth")
    assert "opened the xai-oauth login" in result
    assert opened.url in result
