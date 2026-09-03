import json
from pathlib import Path
from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from hermes_cli.provider_circuits import (
    circuit_status,
    claim_probe,
    is_circuit_open,
    record_failure,
    record_success,
)


def test_rate_limit_opens_immediately_and_expires(tmp_path: Path):
    path = tmp_path / "circuits.json"
    entry = record_failure(
        "zai",
        "glm-5.2",
        FailoverReason.rate_limit,
        retry_after_seconds=600,
        path=path,
        now=1000,
    )

    assert entry["status"] == "open"
    assert is_circuit_open("zai", "glm-5.2", path=path, now=1200)
    assert circuit_status("zai", "glm-5.2", path=path, now=1601)["status"] == "probe_eligible"


def test_expired_circuit_allows_only_one_half_open_probe(tmp_path: Path):
    path = tmp_path / "circuits.json"
    record_failure(
        "zai",
        "glm-5.2",
        "rate_limit",
        retry_after_seconds=10,
        path=path,
        now=1000,
    )

    assert claim_probe("zai", "glm-5.2", path=path, now=1011)
    assert not claim_probe("zai", "glm-5.2", path=path, now=1011)
    assert circuit_status("zai", "glm-5.2", path=path, now=1011)["status"] == "open"
    assert record_success("zai", "glm-5.2", path=path, now=1012)
    assert circuit_status("zai", "glm-5.2", path=path, now=1012)["status"] == "closed"


def test_named_custom_runtime_has_stable_circuit_identity(monkeypatch):
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "_get_named_custom_provider",
        lambda requested: {
            "name": "corp-proxy",
            "base_url": "https://corp.example.test/v1",
            "api_key": "test-only",
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "_try_resolve_from_custom_pool",
        lambda *args, **kwargs: None,
    )

    runtime = runtime_provider._resolve_named_custom_runtime(
        requested_provider="corp-proxy"
    )
    assert runtime["provider"] == "custom"
    assert runtime["circuit_provider"] == "corp-proxy"


def test_transient_failure_requires_threshold(tmp_path: Path):
    path = tmp_path / "circuits.json"
    config = {"provider_circuits": {"transient_failure_threshold": 3}}

    for offset in range(2):
        entry = record_failure(
            "xai-oauth",
            "grok",
            FailoverReason.timeout,
            path=path,
            config=config,
            now=1000 + offset,
        )
        assert entry["status"] == "closed"

    entry = record_failure(
        "xai-oauth",
        "grok",
        FailoverReason.timeout,
        path=path,
        config=config,
        now=1002,
    )
    assert entry["status"] == "open"


def test_rate_limit_headers_extend_default_cooldown(tmp_path: Path):
    path = tmp_path / "circuits.json"
    bucket = SimpleNamespace(limit=100, remaining=0, remaining_seconds_now=7200)
    agent = SimpleNamespace(
        _rate_limit_state=SimpleNamespace(
            requests_min=None,
            requests_hour=bucket,
            tokens_min=None,
            tokens_hour=None,
        )
    )

    entry = record_failure(
        "openai-codex",
        "gpt-5.5",
        "rate_limit",
        agent=agent,
        path=path,
        now=1000,
    )
    assert entry["open_until_epoch"] == 8200


def test_success_closes_without_storing_sensitive_content(tmp_path: Path):
    path = tmp_path / "circuits.json"
    record_failure(
        "provider",
        "model",
        "billing",
        path=path,
        now=1000,
    )
    assert not record_success("provider", "model", path=path, now=1100)
    assert circuit_status("provider", "model", path=path, now=1100)["status"] == "open"
    assert record_success("provider", "model", path=path, now=90000)
    assert circuit_status("provider", "model", path=path, now=90000)["status"] == "closed"

    payload = path.read_text(encoding="utf-8")
    assert "prompt" not in payload.lower()
    assert "token" not in payload.lower()
    assert json.loads(payload)["circuits"]["provider/model"]["consecutive_failures"] == 0


def test_auth_failure_opens_provider_wide_until_health_success(tmp_path: Path):
    path = tmp_path / "circuits.json"
    record_failure("provider", "model-a", "auth", path=path, now=1000)

    assert circuit_status("provider", "model-b", path=path, now=1100)["status"] == "open"
    assert not record_success("provider", "model-b", path=path, now=1100)
    assert (
        circuit_status("provider", "model-b", path=path, now=999999)["status"]
        == "probe_eligible"
    )
    assert record_success("provider", "model-b", path=path, now=999999)
    assert circuit_status("provider", "model-b", path=path, now=999999)["status"] == "closed"
    assert circuit_status("provider", "model-a", path=path, now=999999)["status"] == "closed"


def test_auth_refresh_does_not_clear_quota_circuit(tmp_path: Path):
    path = tmp_path / "circuits.json"
    record_failure("provider", "model", "rate_limit", path=path, now=1000)
    assert not record_success(
        "provider",
        "*",
        path=path,
        now=1100,
        force=True,
        reasons={"auth", "auth_permanent"},
    )
    assert circuit_status("provider", "model", path=path, now=1100)["status"] == "open"


def test_unsupported_failure_does_not_create_circuit(tmp_path: Path):
    path = tmp_path / "circuits.json"
    result = record_failure(
        "provider",
        "model",
        FailoverReason.content_policy_blocked,
        path=path,
        now=1000,
    )
    assert result["status"] == "ignored"
    assert not path.exists()


def test_corrupt_state_is_preserved_on_write(tmp_path: Path):
    path = tmp_path / "circuits.json"
    path.write_text("{broken", encoding="utf-8")

    try:
        record_failure("provider", "model", "rate_limit", path=path, now=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt circuit state should block mutation")

    assert path.read_text(encoding="utf-8") == "{broken"
    assert circuit_status("provider", "model", path=path)["status"] == "unavailable"


# ── reset_seconds_from_context (quota reset → cooldown) ──────────────────────

from hermes_cli.provider_circuits import reset_seconds_from_context  # noqa: E402


def test_reset_seconds_epoch_number():
    now = 1_788_400_000.0
    assert reset_seconds_from_context({"reset_at": now + 90_000}, now=now) == 90_000


def test_reset_seconds_epoch_string_and_millis():
    now = 1_788_400_000.0
    assert reset_seconds_from_context({"reset_at": str(int(now + 3_600))}, now=now) == 3_600
    assert reset_seconds_from_context({"reset_at": (now + 7_200) * 1000}, now=now) == 7_200


def test_reset_seconds_iso_timestamp():
    now = 1_788_400_000.0
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(now + 5_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    assert reset_seconds_from_context({"reset_at": iso}, now=now) == 5_000


def test_reset_seconds_relative_small_number_and_minimum():
    now = 1_788_400_000.0
    # A small value is "seconds until reset" — clamp to the minimum probe interval.
    assert reset_seconds_from_context({"reset_at": 5}, now=now) == 60.0
    assert reset_seconds_from_context({"reset_at": 500}, now=now) == 500.0


def test_reset_seconds_past_missing_or_garbage_is_none():
    now = 1_788_400_000.0
    assert reset_seconds_from_context({"reset_at": now - 10}, now=now) is None
    assert reset_seconds_from_context({}, now=now) is None
    assert reset_seconds_from_context(None, now=now) is None
    assert reset_seconds_from_context({"reset_at": "next tuesday"}, now=now) is None
    assert reset_seconds_from_context({"reset_at": True}, now=now) is None


def test_record_failure_honours_reset_from_error_context(tmp_path: Path):
    """A 429 whose body carries resets_at keeps the circuit open until then."""
    path = tmp_path / "circuits.json"
    now = 1_788_400_000.0
    seconds = reset_seconds_from_context({"reset_at": now + 350_000}, now=now)
    record_failure("openai-codex", "gpt-5.5", FailoverReason.rate_limit,
                   retry_after_seconds=seconds, path=path, now=now)
    status = circuit_status("openai-codex", "gpt-5.5", path=path, now=now + 3_601)
    assert status["status"] == "open", "flat 1h cooldown must not reopen an exhausted quota"
    status = circuit_status("openai-codex", "gpt-5.5", path=path, now=now + 350_001)
    assert status["status"] != "open"
