"""Per-channel bot identity routing (hermes-home #97).

One Socket Mode app receives events, but a channel can be owned by another bot
identity in the same workspace. ``SLACK_CHANNEL_BOT_TOKENS`` maps channel ids to
the token used for outbound posts to that channel.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch



def _ensure_slack_mock():
    """Install mock slack modules so SlackAdapter can be imported (mirrors test_slack.py)."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as slack_mod
from plugins.platforms.slack.adapter import (
    CHANNEL_BOT_TOKENS_ENV,
    SlackAdapter,
    parse_channel_bot_tokens,
    resolve_channel_bot_token,
)


class ParseChannelBotTokensTest(unittest.TestCase):
    def test_env_name_indirection(self):
        env = {"PRESENT_SLACK_BOT_TOKEN": "xoxb-present"}
        self.assertEqual(
            parse_channel_bot_tokens("C0B3CTXLCE8=PRESENT_SLACK_BOT_TOKEN", env),
            {"C0B3CTXLCE8": "xoxb-present"},
        )

    def test_literal_token_and_multiple_entries(self):
        env = {"OTHER": "xoxb-other"}
        parsed = parse_channel_bot_tokens(" C1=xoxb-literal , C2=OTHER ,, bad", env)
        self.assertEqual(parsed, {"C1": "xoxb-literal", "C2": "xoxb-other"})

    def test_unresolvable_env_name_is_dropped_not_substituted(self):
        # A missing env var must never silently resolve to another identity.
        self.assertEqual(parse_channel_bot_tokens("C1=MISSING_VAR", {}), {})

    def test_empty(self):
        self.assertEqual(parse_channel_bot_tokens("", {}), {})
        self.assertEqual(parse_channel_bot_tokens(None, {}), {})

    def test_resolve_channel_bot_token(self):
        env = {CHANNEL_BOT_TOKENS_ENV: "C0B3CTXLCE8=PT", "PT": "xoxb-present"}
        self.assertEqual(resolve_channel_bot_token("C0B3CTXLCE8", env), "xoxb-present")
        self.assertIsNone(resolve_channel_bot_token("C0BBXL6DW2V", env))


class GetClientRoutingTest(unittest.TestCase):
    def _adapter(self):
        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._channel_clients = {}
        adapter._channel_team = {}
        adapter._team_clients = {}
        adapter._app = MagicMock()
        adapter._app.client = "primary"
        return adapter

    def test_channel_identity_beats_workspace_and_primary(self):
        adapter = self._adapter()
        adapter._channel_team["C0B3CTXLCE8"] = "T1"
        adapter._team_clients["T1"] = "team-client"
        adapter._channel_clients["C0B3CTXLCE8"] = "present-client"
        self.assertEqual(adapter._get_client("C0B3CTXLCE8"), "present-client")
        self.assertEqual(adapter._get_client("C0BBXL6DW2V"), "primary")

    def test_unmapped_channel_uses_workspace_client(self):
        adapter = self._adapter()
        adapter._channel_team["C1"] = "T1"
        adapter._team_clients["T1"] = "team-client"
        self.assertEqual(adapter._get_client("C1"), "team-client")


class StandaloneSendTokenTest(unittest.TestCase):
    def test_standalone_send_uses_channel_token(self):
        import asyncio

        captured = {}

        class _Resp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def json(self):
                return {"ok": True, "ts": "1.0"}

        class _Session:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            def post(self, url, headers=None, json=None, **k):
                captured["auth"] = headers["Authorization"]
                captured["channel"] = json["channel"]
                return _Resp()

        aiohttp = MagicMock()
        aiohttp.ClientSession = _Session
        aiohttp.ClientTimeout = lambda **k: None
        env = {
            "SLACK_BOT_TOKEN": "xoxb-wink",
            CHANNEL_BOT_TOKENS_ENV: "C0B3CTXLCE8=PT",
            "PT": "xoxb-present",
        }
        pconfig = MagicMock(token="xoxb-wink")
        with patch.dict(os.environ, env, clear=False), patch.dict(
            sys.modules, {"aiohttp": aiohttp}
        ), patch.object(SlackAdapter, "format_message", lambda self, m: m):
            with patch(
                "gateway.platforms.base.resolve_proxy_url", return_value=None
            ), patch(
                "gateway.platforms.base.proxy_kwargs_for_aiohttp",
                return_value=({}, {}),
            ):
                result = asyncio.run(
                    slack_mod._standalone_send(pconfig, "C0B3CTXLCE8", "hi")
                )
                self.assertTrue(result.get("success"), result)
                self.assertEqual(captured["auth"], "Bearer xoxb-present")
                result = asyncio.run(
                    slack_mod._standalone_send(pconfig, "C0BBXL6DW2V", "hi")
                )
                self.assertEqual(captured["auth"], "Bearer xoxb-wink")


if __name__ == "__main__":
    unittest.main()
