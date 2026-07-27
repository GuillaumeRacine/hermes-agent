import hashlib
import hmac
import json
import os
import time
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.adaptive_routing import (
    apply_guarded_route,
    classify_task,
    decide_route,
    decide_shadow_route,
    filter_fallbacks_for_decision,
    load_policy,
    observe_shadow_route,
    capture_route_usage,
    record_route_outcome,
    record_startup_ready,
)
from hermes_cli.provider_circuits import record_failure

_POLICY_PRODUCER = "local-ai-router-evals/weekly_router_cycle.py:runtime-policy"


@pytest.fixture(autouse=True)
def _router_policy_key(tmp_path, monkeypatch):
    key_path = tmp_path / "router-policy.key"
    key_path.write_bytes(b"test-router-policy-key-material!!")
    key_path.chmod(0o600)
    monkeypatch.setenv("HERMES_ROUTER_EVIDENCE_KEY_PATH", str(key_path))


def _write_signed_policy(path, policy):
    unsigned = dict(policy)
    unsigned.pop("evidence_attestation", None)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    domain = (
        "hermes-router-evidence-v1\0hmac-sha256\0"
        f"{_POLICY_PRODUCER}\0"
    ).encode("utf-8")
    key = Path(os.environ["HERMES_ROUTER_EVIDENCE_KEY_PATH"]).read_bytes()
    signed = dict(unsigned)
    signed["evidence_attestation"] = {
        "schema_version": 1,
        "algorithm": "hmac-sha256",
        "producer": _POLICY_PRODUCER,
        "signature": hmac.new(
            key,
            domain + canonical,
            hashlib.sha256,
        ).hexdigest(),
    }
    path.write_text(json.dumps(signed), encoding="utf-8")


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


def test_classifier_marks_common_sensitive_values_private():
    samples = [
        "Use this secret API key sk-live-example for the task",
        "Review these employee payroll records",
        "Email alice@example.com and include her phone number 416-555-0199",
    ]
    assert all(classify_task(sample)["private"] for sample in samples)


def test_classifier_marks_raw_credential_formats_private_and_high_risk():
    samples = [
        "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhYmNkZWYifQ.signature",
        "Ab9_Zy8-Xw7_Vu6-Ts5_Rq4-Po3_Nm2-Lk1_Ji0-Hg9_Fd8",
    ]
    results = [classify_task(sample) for sample in samples]
    assert all(item["private"] for item in results)
    assert all(item["class"] in {"C3", "C4"} for item in results)


def test_uninspected_attachment_is_private_by_default():
    result = classify_task("summarize the attachment", assume_private=True)
    assert result["private"] is True
    assert result["features"]["private_matches"] >= 1


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
    assert len(event["decision_id"]) == 24
    assert len(event["prompt_fingerprint"]) == 16


def test_outcome_uses_turn_deltas_and_contains_no_content(tmp_path):
    policy = tmp_path / "policy.json"
    telemetry = tmp_path / "events.jsonl"
    _policy(policy)
    config = {
        "adaptive_routing": {
            "enabled": True,
            "mode": "shadow",
            "policy_path": str(policy),
            "telemetry_path": str(telemetry),
            "private_provider_allowlist": ["openai-codex"],
        }
    }
    decision = observe_shadow_route(
        "private prompt marker ORCHID",
        "gpt",
        "openai-codex",
        config,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt",
        session_input_tokens=100,
        session_output_tokens=25,
        session_cache_read_tokens=70,
        session_cache_write_tokens=5,
        session_reasoning_tokens=10,
        session_total_tokens=200,
        session_api_calls=2,
        session_estimated_cost_usd=0.05,
        session_cost_status="estimated",
        _fallback_index=0,
    )
    usage_before = capture_route_usage(agent)
    agent.session_input_tokens += 11
    agent.session_output_tokens += 7
    agent.session_cache_read_tokens += 9
    agent.session_reasoning_tokens += 3
    agent.session_total_tokens += 30
    agent.session_api_calls += 1
    agent.session_estimated_cost_usd += 0.004
    record_route_outcome(
        decision,
        {
            "completed": True,
            "api_calls": 1,
            "last_prompt_tokens": 90,
            "final_response": "response marker TULIP",
        },
        agent,
        config,
        usage_before=usage_before,
        started_monotonic=time.monotonic() - 1,
        surface="cli",
    )

    content = telemetry.read_text(encoding="utf-8")
    assert "ORCHID" not in content
    assert "TULIP" not in content
    decision_event, outcome = [
        json.loads(line) for line in content.splitlines()
    ]
    assert outcome["event"] == "route_outcome"
    assert outcome["decision_id"] == decision_event["decision_id"]
    assert outcome["tokens"] == {
        "cache_read_tokens": 9,
        "cache_write_tokens": 0,
        "input_tokens": 11,
        "output_tokens": 7,
        "reasoning_tokens": 3,
        "total_tokens": 30,
    }
    assert outcome["status"] == "completed"
    assert outcome["failure_kind"] is None
    assert outcome["api_calls"] == 1
    assert outcome["context_tokens"] == 90
    assert outcome["cost_status"] == "estimated"
    assert 900 <= outcome["latency_ms"] <= 2000


def test_outcome_redacts_secret_tainted_route_identifiers(tmp_path):
    telemetry = tmp_path / "events.jsonl"
    config = {
        "adaptive_routing": {
            "enabled": True,
            "telemetry_path": str(telemetry),
        }
    }
    secret_route = "token value with spaces"
    decision = {
        "decision_id": "a" * 24,
        "class": "C2",
    }
    agent = SimpleNamespace(provider=secret_route, model=secret_route)
    record_route_outcome(
        decision,
        {"failed": True, "error": "raw secret error"},
        agent,
        config,
        usage_before={},
        started_monotonic=time.monotonic(),
        surface="cli",
    )
    content = telemetry.read_text(encoding="utf-8")
    assert secret_route not in content
    assert "raw secret error" not in content
    event = json.loads(content)
    assert event["actual"] == {
        "provider": "redacted",
        "model": "redacted",
    }
    assert event["failure_kind"] == "unknown"


def test_startup_ready_is_content_free(tmp_path):
    telemetry = tmp_path / "events.jsonl"
    config = {
        "adaptive_routing": {
            "enabled": True,
            "telemetry_path": str(telemetry),
        }
    }
    record_startup_ready(
        config,
        started_monotonic=time.monotonic() - 2,
        surface="classic_cli",
        deferred=True,
        safe_mode=False,
    )
    event = json.loads(telemetry.read_text(encoding="utf-8"))
    assert event["event"] == "startup_ready"
    assert event["surface"] == "classic_cli"
    assert event["deferred"] is True
    assert 1900 <= event["latency_ms"] <= 3000


def test_guarded_mode_uses_only_selected_native_route(tmp_path):
    policy = {
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": {
            "C1": {
                "status": "candidate",
                "shadow_recommendation": None,
                "selected": {
                    "provider": "local",
                    "runtime": {
                        "mode": "hermes",
                        "provider": "local-ollama",
                        "model": "llama3.2:3b",
                    },
                },
                "critic": None,
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_signed_policy(policy_path, policy)
    config = {
        "adaptive_routing": {
            "enabled": True,
            "mode": "guarded",
            "policy_path": str(policy_path),
            "private_provider_allowlist": ["local-ollama"],
        },
        "provider_circuits": {"enabled": False},
    }
    decision = decide_route("summarize this", "grok", "xai-oauth", config)
    route = {
        "model": "grok",
        "runtime": {
            "provider": "xai-oauth",
            "base_url": "https://example.test",
            "api_mode": "codex_responses",
            "api_key": "secret",
            "command": None,
            "args": [],
            "credential_pool": None,
            "max_tokens": None,
        },
        "signature": ("grok", "xai-oauth"),
    }

    guarded = apply_guarded_route(
        route,
        decision,
        config,
        resolve_runtime=lambda **kwargs: {
            "provider": "local-ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_mode": "chat_completions",
            "api_key": "local",
        },
    )

    assert guarded["model"] == "llama3.2:3b"
    assert guarded["runtime"]["provider"] == "local-ollama"


def test_guarded_policy_requires_valid_attestation(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy = {
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": {},
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert load_policy(policy_path, maximum_age_seconds=3600) is None

    _write_signed_policy(policy_path, policy)
    assert load_policy(policy_path, maximum_age_seconds=3600) is not None

    tampered = json.loads(policy_path.read_text())
    tampered["classes"]["C1"] = {"status": "candidate"}
    policy_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert load_policy(policy_path, maximum_age_seconds=3600) is None


def test_policy_with_non_object_root_fails_closed(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("[]", encoding="utf-8")
    assert load_policy(policy_path, maximum_age_seconds=3600) is None


def test_guarded_mode_holds_external_worker_and_c4_without_critic(tmp_path):
    policy = {
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": {
            "C4": {
                "status": "candidate",
                "shadow_recommendation": None,
                "selected": {
                    "provider": "codex",
                    "runtime": {
                        "mode": "external_worker",
                        "provider": "openai-codex",
                        "model": "gpt-5.5",
                    },
                },
                "critic": None,
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_signed_policy(policy_path, policy)
    decision = decide_route(
        "deploy this to production",
        "grok",
        "xai-oauth",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "guarded",
                "policy_path": str(policy_path),
            }
        },
    )

    assert decision["active"] is None
    assert (
        "selected route is not a native Hermes runtime"
        in decision["activation_blockers"]
    )
    assert (
        "C4 requires a different-provider critic"
        in decision["activation_blockers"]
    )


def test_guarded_mode_rejects_stale_policy(tmp_path):
    policy = {
        "mode": "shadow",
        "generated_at": (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat(),
        "classes": {
            "C1": {
                "status": "candidate",
                "selected": {
                    "runtime": {
                        "mode": "hermes",
                        "provider": "local-ollama",
                        "model": "llama3.2:3b",
                    }
                },
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_signed_policy(policy_path, policy)
    decision = decide_route(
        "summarize this",
        "grok",
        "xai-oauth",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "guarded",
                "policy_path": str(policy_path),
                "maximum_policy_age_seconds": 3600,
            }
        },
    )
    assert decision["active"] is None
    assert "no promotion-eligible selected route" in decision["activation_blockers"]


def test_selected_current_route_is_still_pinned(tmp_path):
    policy = {
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": {
            "C1": {
                "status": "candidate",
                "selected": {
                    "runtime": {
                        "mode": "hermes",
                        "provider": "xai-oauth",
                        "model": "grok",
                    }
                },
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_signed_policy(policy_path, policy)
    config = {
        "adaptive_routing": {
            "enabled": True,
            "mode": "guarded",
            "policy_path": str(policy_path),
            "pin_path": str(tmp_path / "pins.json"),
        },
        "provider_circuits": {"enabled": False},
    }
    decision = decide_route("summarize this", "grok", "xai-oauth", config)
    route = {
        "model": "grok",
        "runtime": {"provider": "xai-oauth"},
    }
    assert apply_guarded_route(route, decision, config, pin_key="task-1") is route
    pins = json.loads((tmp_path / "pins.json").read_text(encoding="utf-8"))
    assert len(pins["pins"]) == 1


def test_c4_blocks_when_different_provider_critic_circuit_is_open(tmp_path):
    circuits = tmp_path / "circuits.json"
    record_failure(
        "openai-codex",
        "gpt-5.5",
        "rate_limit",
        retry_after_seconds=3600,
        path=circuits,
    )
    policy = {
        "mode": "shadow",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": {
            "C4": {
                "status": "candidate",
                "selected": {
                    "runtime": {
                        "mode": "hermes",
                        "provider": "xai-oauth",
                        "model": "grok",
                    }
                },
                "critic": {
                    "runtime": {
                        "mode": "hermes",
                        "provider": "openai-codex",
                        "model": "gpt-5.5",
                    }
                },
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    _write_signed_policy(policy_path, policy)
    decision = decide_route(
        "deploy this to production",
        "grok",
        "xai-oauth",
        {
            "adaptive_routing": {
                "enabled": True,
                "mode": "guarded",
                "policy_path": str(policy_path),
            },
            "provider_circuits": {
                "enabled": True,
                "state_path": str(circuits),
            },
        },
    )
    assert decision["active"] is None
    assert "C4 critic has an open provider circuit" in decision["activation_blockers"]


def test_private_decision_filters_global_fallbacks():
    filtered = filter_fallbacks_for_decision(
        [
            {"provider": "local-ollama", "model": "local"},
            {"provider": "zai", "model": "cloud"},
        ],
        {
            "private": True,
            "allowed_fallback_providers": ["local-ollama"],
        },
    )
    assert filtered == [{"provider": "local-ollama", "model": "local"}]


def test_private_resolution_failure_cannot_use_disallowed_primary():
    route = {
        "model": "cloud-primary",
        "runtime": {"provider": "xai-oauth"},
    }
    decision = {
        "mode": "guarded",
        "class": "C1",
        "private": True,
        "active": {"provider": "local-ollama", "model": "local-model"},
        "activation_blockers": [],
        "block_execution": False,
    }
    config = {
        "adaptive_routing": {
            "private_provider_allowlist": ["local-ollama"],
        },
        "provider_circuits": {"enabled": False},
    }

    def fail_resolution(**kwargs):
        raise RuntimeError("credentials unavailable")

    result = apply_guarded_route(
        route,
        decision,
        config,
        resolve_runtime=fail_resolution,
    )

    assert result is route
    assert decision["block_execution"] is True
    assert any(
        "configured primary violates" in blocker
        for blocker in decision["activation_blockers"]
    )


def test_guarded_route_preserves_logical_circuit_provider():
    route = {
        "model": "cloud-primary",
        "runtime": {"provider": "xai-oauth"},
    }
    decision = {
        "mode": "guarded",
        "class": "C1",
        "private": False,
        "active": {"provider": "local-ollama", "model": "local-model"},
        "activation_blockers": [],
        "block_execution": False,
    }
    config = {"provider_circuits": {"enabled": False}}

    result = apply_guarded_route(
        route,
        decision,
        config,
        resolve_runtime=lambda **kwargs: {
            "provider": "custom",
            "api_key": "no-key-required",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_mode": "chat_completions",
        },
    )

    assert result["runtime"]["provider"] == "custom"
    assert result["runtime"]["circuit_provider"] == "local-ollama"


def test_guarded_route_blocks_when_new_pin_cannot_be_persisted():
    route = {
        "model": "cloud-primary",
        "runtime": {"provider": "xai-oauth"},
    }
    decision = {
        "mode": "guarded",
        "class": "C1",
        "private": False,
        "active": {"provider": "local-ollama", "model": "local-model"},
        "activation_blockers": [],
        "block_execution": False,
    }
    config = {"provider_circuits": {"enabled": False}}
    with patch(
        "hermes_cli.adaptive_routing._save_route_pin",
        return_value=False,
    ):
        result = apply_guarded_route(
            route,
            decision,
            config,
            pin_key="task-1",
            resolve_runtime=lambda **kwargs: {
                "provider": "custom",
                "api_key": "no-key-required",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_mode": "chat_completions",
            },
        )
    assert result is route
    assert decision["block_execution"] is True
    assert decision["active"] is None


def test_private_pin_escalation_blocks_primary_and_filters_fallbacks(tmp_path):
    config = {
        "adaptive_routing": {
            "private_provider_allowlist": ["local-ollama"],
            "pin_path": str(tmp_path / "pins.json"),
        },
        "provider_circuits": {"enabled": True},
    }
    decision = {
        "mode": "guarded",
        "class": "C1",
        "private": False,
        "active": None,
        "activation_blockers": [],
        "block_execution": False,
        "allowed_fallback_providers": None,
    }
    route = {
        "model": "cloud-primary",
        "runtime": {"provider": "xai-oauth"},
    }
    pinned = {
        "provider": "local-ollama",
        "model": "local-model",
        "task_class": "C1",
        "private": True,
    }
    with (
        patch(
            "hermes_cli.adaptive_routing.get_route_pin",
            return_value=pinned,
        ),
        patch(
            "hermes_cli.provider_circuits.circuit_status",
            return_value={"status": "open"},
        ),
    ):
        result = apply_guarded_route(
            route,
            decision,
            config,
            pin_key="task-1",
        )

    assert result is route
    assert decision["private"] is True
    assert decision["block_execution"] is True
    assert filter_fallbacks_for_decision(
        [{"provider": "zai", "model": "cloud"}],
        decision,
    ) == []
