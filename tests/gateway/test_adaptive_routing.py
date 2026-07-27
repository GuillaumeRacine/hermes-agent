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


def test_gateway_guarded_activation_requires_isolated_task_flag():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    decision = {"mode": "guarded", "class": "C1", "active": {"provider": "local"}}
    guarded = {
        "model": "llama3.2:3b",
        "runtime": {"provider": "local-ollama"},
        "signature": ("llama3.2:3b", "local-ollama"),
        "adaptive_routing": decision,
    }
    with (
        mock.patch(
            "hermes_cli.config.load_config",
            return_value={"adaptive_routing": {"enabled": True}},
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.observe_shadow_route",
            return_value=decision,
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.get_route_pin",
            return_value=None,
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.apply_guarded_route",
            return_value=guarded,
        ) as apply_guarded,
    ):
        foreground = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "summarize",
            "gpt-5.5",
            _runtime(),
        )
        background = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "summarize",
            "gpt-5.5",
            _runtime(),
            allow_guarded_activation=True,
            pin_key="task-1",
        )

    assert foreground["model"] == "gpt-5.5"
    assert background["model"] == "llama3.2:3b"
    apply_guarded.assert_called_once()
    assert apply_guarded.call_args.kwargs["pin_key"] == "task-1"


def test_gateway_guarded_preflight_exception_blocks_execution():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    with mock.patch(
        "hermes_cli.adaptive_routing.observe_shadow_route",
        side_effect=RuntimeError("policy state unavailable"),
    ):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "private background task",
            "gpt-5.5",
            _runtime(),
            allow_guarded_activation=True,
            pin_key="task-1",
        )

    assert route["adaptive_routing"]["block_execution"] is True
    assert route["adaptive_routing"]["active"] is None


def test_gateway_marks_uninspected_attachments_private():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    decision = {"mode": "shadow", "class": "C1", "private": True}
    with (
        mock.patch(
            "hermes_cli.config.load_config",
            return_value={"adaptive_routing": {"enabled": True}},
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.observe_shadow_route",
            return_value=decision,
        ) as observe,
    ):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "summarize the attachment",
            "gpt-5.5",
            _runtime(),
            uninspected_attachments=True,
        )
    assert route["adaptive_routing"]["private"] is True
    assert observe.call_args.kwargs["assume_private"] is True


def test_gateway_corrupt_pin_state_blocks_guarded_task():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._service_tier = None
    decision = {
        "mode": "guarded",
        "class": "C1",
        "active": {"provider": "local-ollama", "model": "local-model"},
    }
    with (
        mock.patch(
            "hermes_cli.config.load_config",
            return_value={"adaptive_routing": {"enabled": True}},
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.observe_shadow_route",
            return_value=decision,
        ),
        mock.patch(
            "hermes_cli.adaptive_routing.get_route_pin",
            side_effect=RuntimeError("pin state unavailable"),
        ),
    ):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "background task",
            "gpt-5.5",
            _runtime(),
            allow_guarded_activation=True,
            pin_key="task-1",
        )
    assert route["adaptive_routing"]["block_execution"] is True
