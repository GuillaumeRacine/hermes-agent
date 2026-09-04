"""Tests for the degraded-mode policy (hermes-home #233 P0-1)."""

from pathlib import Path

from agent import degraded_mode as dm
from hermes_cli.provider_circuits import record_failure


def _config(tmp_path: Path, *, circuits_enabled=True, floor_models=None, fallbacks=None):
    return {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "fallback_providers": fallbacks
        if fallbacks is not None
        else [
            {"provider": "zai", "model": "glm-5.2"},
            {"provider": "local-ollama", "model": "llama3.2:3b"},
        ],
        "provider_circuits": {
            "enabled": circuits_enabled,
            "state_path": str(tmp_path / "circuits.json"),
        },
        "degraded_mode": {
            "enabled": True,
            "floor": {"providers": ["local-ollama", "ollama"], "models": floor_models or []},
            "queue_path": str(tmp_path / "pending.jsonl"),
            "replay_max_age_hours": 24,
        },
    }


class TestFloorMatching:
    def test_provider_match_is_case_insensitive(self, tmp_path):
        cfg = _config(tmp_path)
        assert dm.is_sub_floor("Local-Ollama", "anything", cfg)
        assert dm.is_sub_floor("ollama", "llama3.2:3b", cfg)
        assert not dm.is_sub_floor("openai-codex", "gpt-5.5", cfg)

    def test_model_glob_match(self, tmp_path):
        cfg = _config(tmp_path, floor_models=["*:3b", "*-mini"])
        assert dm.is_sub_floor("openrouter", "llama3.2:3b", cfg)
        assert dm.is_sub_floor("openai", "GPT-4o-Mini", cfg)
        assert not dm.is_sub_floor("openai", "gpt-4o", cfg)

    def test_empty_floor_matches_nothing(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["degraded_mode"]["floor"] = {}
        assert not dm.is_sub_floor("local-ollama", "llama3.2:3b", cfg)


class TestCapableRuntimeAvailable:
    def test_no_circuits_open_returns_primary(self, tmp_path):
        ok, detail = dm.capable_runtime_available(_config(tmp_path), now=1000)
        assert ok is True
        assert detail == "openai-codex/gpt-5.6-sol"

    def test_primary_open_fallback_closed_returns_fallback(self, tmp_path):
        cfg = _config(tmp_path)
        path = tmp_path / "circuits.json"
        record_failure("openai-codex", "gpt-5.6-sol", "rate_limit", retry_after_seconds=3600, path=path, now=1000)
        ok, detail = dm.capable_runtime_available(cfg, now=1200)
        assert ok is True
        assert detail == "zai/glm-5.2"

    def test_everything_open_or_sub_floor_returns_false_with_detail_and_eta(self, tmp_path):
        cfg = _config(tmp_path)
        path = tmp_path / "circuits.json"
        record_failure("openai-codex", "gpt-5.6-sol", "rate_limit", retry_after_seconds=7200, path=path, now=1000)
        record_failure("zai", "glm-5.2", "rate_limit", retry_after_seconds=600, path=path, now=1000)
        ok, detail = dm.capable_runtime_available(cfg, now=1200)
        assert ok is False
        assert "openai-codex/gpt-5.6-sol until" in detail
        assert "zai/glm-5.2 until" in detail
        assert "local-ollama/llama3.2:3b" in detail  # reported as sub-floor, never selected
        eta = dm.earliest_recovery_eta(cfg, now=1200)
        assert eta is not None
        # zai reopens at t=1600 (epoch) which is the earliest of the two.
        assert eta.startswith("1970-01-01T00:26")
        notice = dm.format_notice(cfg, detail, eta)
        assert "queued" in notice and eta in notice

    def test_provider_wide_circuit_blocks_every_model_of_that_provider(self, tmp_path):
        cfg = _config(
            tmp_path,
            fallbacks=[
                {"provider": "openai-codex", "model": "gpt-5.5"},
                {"provider": "local-ollama", "model": "llama3.2:3b"},
            ],
        )
        path = tmp_path / "circuits.json"
        # ``billing`` is provider-wide → opens openai-codex/* as well.
        record_failure("openai-codex", "gpt-5.6-sol", "billing", retry_after_seconds=86400, path=path, now=1000)
        ok, detail = dm.capable_runtime_available(cfg, now=1200)
        assert ok is False
        assert "openai-codex/gpt-5.5" in detail

    def test_circuits_disabled_returns_primary(self, tmp_path):
        cfg = _config(tmp_path, circuits_enabled=False)
        path = tmp_path / "circuits.json"
        record_failure("openai-codex", "gpt-5.6-sol", "rate_limit", retry_after_seconds=7200, path=path, now=1000)
        ok, detail = dm.capable_runtime_available(cfg, now=1200)
        assert ok is True
        assert detail == "openai-codex/gpt-5.6-sol"
        assert dm.earliest_recovery_eta(cfg, now=1200) is None

    def test_empty_chain_fails_open(self):
        ok, detail = dm.capable_runtime_available({}, now=1000)
        assert ok is True
        assert "no primary or fallback runtime" in detail
        assert dm.assess({})["notice"] is None

    def test_assess_bundles_notice(self, tmp_path):
        cfg = _config(tmp_path)
        verdict = dm.assess(cfg, now=1000)
        assert verdict["available"] is True
        assert verdict["notice"] is None
        record_failure("openai-codex", "gpt-5.6-sol", "rate_limit", retry_after_seconds=7200, path=tmp_path / "circuits.json", now=1000)
        record_failure("zai", "glm-5.2", "rate_limit", retry_after_seconds=7200, path=tmp_path / "circuits.json", now=1000)
        verdict = dm.assess(cfg, now=1200)
        assert verdict["available"] is False
        assert "emergency local model" in verdict["notice"]


class TestFormatNotice:
    def test_custom_template(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["degraded_mode"]["notify_template"] = "down: {detail} / back {eta}"
        assert dm.format_notice(cfg, "x", "soon") == "down: x / back soon"

    def test_broken_template_falls_back(self, tmp_path):
        cfg = _config(tmp_path)
        cfg["degraded_mode"]["notify_template"] = "oops {missing}"
        out = dm.format_notice(cfg, "x", None)
        assert "emergency local model" in out and "unknown" in out


class TestAgentFlagHelpers:
    def test_mark_and_refresh_flag(self, tmp_path, monkeypatch):
        cfg = _config(tmp_path)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

        class _Agent:
            provider = "local-ollama"
            model = "llama3.2:3b"
            _fallback_activated = True

        agent = _Agent()
        assert dm.mark_agent_sub_floor_if_needed(agent, "local-ollama", "llama3.2:3b") is True
        assert agent._degraded_sub_floor is True
        assert "queued" in agent._degraded_notice
        # Still parked on the floor → flag stays.
        assert dm.refresh_agent_sub_floor_flag(agent) is True
        # Recovered primary → flag clears.
        agent.provider, agent.model = "openai-codex", "gpt-5.6-sol"
        assert dm.refresh_agent_sub_floor_flag(agent) is False
        assert agent._degraded_sub_floor is False

    def test_capable_provider_not_marked(self, tmp_path, monkeypatch):
        cfg = _config(tmp_path)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

        class _Agent:
            pass

        agent = _Agent()
        assert dm.mark_agent_sub_floor_if_needed(agent, "zai", "glm-5.2") is False
        assert not getattr(agent, "_degraded_sub_floor", False)
