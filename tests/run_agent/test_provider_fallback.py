"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent, _pool_may_recover_from_rate_limit
from hermes_cli.provider_circuits import record_failure


@pytest.fixture(autouse=True)
def _disable_provider_circuits_by_default(monkeypatch):
    """Keep legacy fallback tests independent of workstation circuit state."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"provider_circuits": {"enabled": False}},
    )


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"

    def test_skips_open_circuit_and_uses_next_fallback(self, tmp_path):
        path = tmp_path / "provider-circuits.json"
        record_failure(
            "zai",
            "glm-5.2",
            "rate_limit",
            retry_after_seconds=3600,
            path=path,
            now=1000,
        )
        fbs = [
            {"provider": "zai", "model": "glm-5.2"},
            {"provider": "openai-codex", "model": "gpt-5.5"},
        ]
        agent = _make_agent(fallback_model=fbs)
        called = []

        def _resolve(provider, model=None, **kwargs):
            called.append((provider, model))
            return _mock_client(), model

        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={
                    "provider_circuits": {
                        "enabled": True,
                        "state_path": str(path),
                    }
                },
            ),
            patch("time.time", return_value=1200),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=_resolve,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert called == [("openai-codex", "gpt-5.5")]
        assert agent._fallback_index == 2

    def test_normalizes_fallback_model_before_circuit_lookup(self):
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "anthropic",
                    "model": "anthropic/claude-sonnet-4.6",
                }
            ]
        )
        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={"provider_circuits": {"enabled": True}},
            ),
            patch(
                "hermes_cli.provider_circuits.circuit_status",
                return_value={"status": "open"},
            ) as status,
            patch(
                "agent.auxiliary_client.resolve_provider_client",
            ) as resolve,
        ):
            assert agent._try_activate_fallback() is False

        assert status.call_args.args == ("anthropic", "claude-sonnet-4-6")
        resolve.assert_not_called()

    def test_fallback_circuit_check_exception_fails_closed(self):
        agent = _make_agent(
            fallback_model=[
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "zai", "model": "glm-5.2"},
            ]
        )
        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={"provider_circuits": {"enabled": True}},
            ),
            patch(
                "hermes_cli.provider_circuits.circuit_status",
                side_effect=OSError("state unavailable"),
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
            ) as resolve,
        ):
            assert agent._try_activate_fallback() is False

        resolve.assert_not_called()
        assert agent._fallback_index == 2

    def test_open_primary_circuit_activates_fallback_before_request(self, tmp_path):
        path = tmp_path / "provider-circuits.json"
        agent = _make_agent(
            fallback_model=[{"provider": "openai-codex", "model": "gpt-5.5"}]
        )
        record_failure(
            agent.provider,
            agent.model,
            "rate_limit",
            retry_after_seconds=3600,
            path=path,
            now=1000,
        )

        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={
                    "provider_circuits": {
                        "enabled": True,
                        "state_path": str(path),
                    }
                },
            ),
            patch("time.time", return_value=1200),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "gpt-5.5"),
            ),
        ):
            assert agent._restore_primary_runtime() is True

        assert agent.provider == "openai-codex"
        assert agent.model == "gpt-5.5"

    def test_open_primary_without_fallback_sets_blocked_preflight(self, tmp_path):
        path = tmp_path / "provider-circuits.json"
        agent = _make_agent(fallback_model=None)
        record_failure(
            agent.provider,
            agent.model,
            "rate_limit",
            retry_after_seconds=3600,
            path=path,
            now=1000,
        )

        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={
                    "provider_circuits": {
                        "enabled": True,
                        "state_path": str(path),
                    }
                },
            ),
            patch("time.time", return_value=1200),
        ):
            assert agent._restore_primary_runtime() is False

        assert "no fallback" in agent._provider_circuit_blocked_reason.lower()

    def test_unavailable_circuit_state_blocks_primary_request(self):
        agent = _make_agent(
            fallback_model=[{"provider": "openai-codex", "model": "gpt-5.5"}]
        )
        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={"provider_circuits": {"enabled": True}},
            ),
            patch(
                "hermes_cli.provider_circuits.circuit_status",
                return_value={"status": "unavailable"},
            ),
            patch.object(agent, "_try_activate_fallback") as activate,
        ):
            assert agent._restore_primary_runtime() is False

        activate.assert_not_called()
        assert "circuit state is unavailable" in (
            agent._provider_circuit_blocked_reason.lower()
        )

    def test_open_active_fallback_advances_to_next_healthy_fallback(self):
        agent = _make_agent(
            fallback_model=[
                {"provider": "first", "model": "first-model"},
                {"provider": "second", "model": "second-model"},
            ]
        )
        agent._fallback_activated = True
        agent._fallback_index = 1
        agent.provider = "first"
        agent.model = "first-model"

        def _status(provider, model, **kwargs):
            if provider in {agent._primary_runtime["provider"], "first"}:
                return {"status": "open", "open_until": "later"}
            return {"status": "closed"}

        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={"provider_circuits": {"enabled": True}},
            ),
            patch(
                "hermes_cli.provider_circuits.circuit_status",
                side_effect=_status,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), "second-model"),
            ),
        ):
            assert agent._restore_primary_runtime() is True

        assert agent.provider == "second"
        assert agent.model == "second-model"


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False

    def test_single_credential_returns_false(self):
        """With one credential that just 429'd, rotation has nowhere to go.

        The pool may still report has_available() True once cooldown expires,
        but retrying against the same entry will hit the same daily-quota
        429 and burn the retry budget.  Must fall back.
        """
        assert _pool_may_recover_from_rate_limit(_pool(1)) is False

    def test_single_credential_in_cooldown_returns_false(self):
        assert _pool_may_recover_from_rate_limit(_pool(1, has_available=False)) is False

    def test_two_credentials_available_returns_true(self):
        """With >1 credentials and at least one available, rotate instead of fallback."""
        assert _pool_may_recover_from_rate_limit(_pool(2)) is True

    def test_multiple_credentials_all_in_cooldown_returns_false(self):
        """All credentials cooling down — fall back rather than wait."""
        assert _pool_may_recover_from_rate_limit(_pool(3, has_available=False)) is False

    def test_many_credentials_available_returns_true(self):
        assert _pool_may_recover_from_rate_limit(_pool(10)) is True


# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )

    def test_skips_entry_matching_current_base_url_and_model(self):
        """Two custom_providers entries pointing at the same shim URL
        with the same model should dedup even if their provider names differ."""
        fbs = [
            # Different provider name but same shim URL + model — same backend.
            {"provider": "claude-cli-alt", "model": "claude-opus-4.7",
             "base_url": "http://127.0.0.1:7891/v1"},
            # Real different fallback.
            {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "claude-cli"
        agent.model = "claude-opus-4.7"
        agent.base_url = "http://127.0.0.1:7891/v1"

        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # Same shim/base_url+model entry skipped, second one used.
        assert called == [("openrouter", "anthropic/claude-opus-4.7")], (
            f"expected base_url-aware dedup, got call order: {called}"
        )

    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()
