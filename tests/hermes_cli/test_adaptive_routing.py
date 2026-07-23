import json

from hermes_cli.adaptive_routing import (
    classify_task,
    decide_shadow_route,
    observe_shadow_route,
)


def _policy(path, provider="local-ollama", model="llama3.2:3b"):
    path.write_text(
        json.dumps(
            {
                "mode": "shadow",
                "classes": {
                    "C1": {
                        "status": "hold",
                        "shadow_recommendation": {
                            "provider": "candidate",
                            "runtime": {
                                "mode": "hermes",
                                "provider": provider,
                                "model": model,
                            },
                        },
                    },
                    "C4": {
                        "status": "hold",
                        "shadow_recommendation": {
                            "provider": "candidate",
                            "runtime": {
                                "mode": "external_worker",
                                "provider": "claude-code",
                                "model": "sonnet",
                            },
                        },
                        "critic": {
                            "runtime": {
                                "provider": "openai-codex",
                                "model": "gpt-5.5",
                            }
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_classifier_escalates_production_and_private_work():
    result = classify_task("Deploy this customer data migration to production")
    assert result["class"] == "C4"
    assert result["private"] is True


def test_disabled_shadow_mode_returns_none(tmp_path):
    result = decide_shadow_route(
        "summarize this",
        "gpt",
        "openai-codex",
        {"adaptive_routing": {"enabled": False}},
    )
    assert result is None


def test_shadow_recommends_worker_without_changing_current_route(tmp_path):
    policy = tmp_path / "policy.json"
    _policy(policy)
    decision = decide_shadow_route(
        "Deploy this to production with a canary",
        "gpt-5.5",
        "openai-codex",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "shadow",
                "policy_path": str(policy),
                "private_provider_allowlist": ["local-ollama", "openai-codex"],
            }
        },
    )
    assert decision["current"] == {
        "provider": "openai-codex",
        "model": "gpt-5.5",
    }
    assert decision["recommended"]["worker_mode"] == "external_worker"
    assert decision["recommended"]["critic_provider"] == "openai-codex"


def test_private_boundary_suppresses_unapproved_provider(tmp_path):
    policy = tmp_path / "policy.json"
    _policy(policy, provider="zai", model="glm-5.2")
    decision = decide_shadow_route(
        "Summarize this private family record",
        "gpt-5.5",
        "openai-codex",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "shadow",
                "policy_path": str(policy),
                "private_provider_allowlist": ["local-ollama", "openai-codex"],
            }
        },
    )
    assert decision["recommended"] is None
    assert decision["blocked_for_privacy"] is True


def test_telemetry_contains_no_prompt_content(tmp_path):
    policy = tmp_path / "policy.json"
    telemetry = tmp_path / "events.jsonl"
    _policy(policy)
    secret_phrase = "summarize private family codename ORCHID"
    observe_shadow_route(
        secret_phrase,
        "gpt-5.5",
        "openai-codex",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "shadow",
                "policy_path": str(policy),
                "telemetry_path": str(telemetry),
                "private_provider_allowlist": ["local-ollama", "openai-codex"],
            }
        },
    )
    content = telemetry.read_text(encoding="utf-8")
    assert "ORCHID" not in content
    event = json.loads(content)
    assert event["event"] == "route_decision"
    assert len(event["prompt_fingerprint"]) == 16
