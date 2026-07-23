from unittest import mock

from gateway import run as gateway_run


def _runtime():
    return {
        "api_key": "***",
        "base_url": "https://example.test/v1",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "max_tokens": None,
    }


def test_gateway_observes_shadow_route_without_mutating_runtime():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    decision = {
        "mode": "shadow",
        "class": "C2",
        "recommended": {
            "provider": "claude-code",
            "model": "sonnet",
            "worker_mode": "external_worker",
        },
    }
    with mock.patch(
        "hermes_cli.config.load_config",
        return_value={"adaptive_routing": {"enabled": True}},
    ), mock.patch(
        "hermes_cli.adaptive_routing.observe_shadow_route",
        return_value=decision,
    ) as observe:
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "review this implementation",
            "gpt-5.5",
            _runtime(),
        )

    assert route["model"] == "gpt-5.5"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["adaptive_routing"] == decision
    observe.assert_called_once()


def test_gateway_shadow_observer_failure_is_non_blocking():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    with mock.patch(
        "hermes_cli.adaptive_routing.observe_shadow_route",
        side_effect=RuntimeError("telemetry unavailable"),
    ):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "hello",
            "gpt-5.5",
            _runtime(),
        )

    assert route["model"] == "gpt-5.5"
    assert route["adaptive_routing"] is None
